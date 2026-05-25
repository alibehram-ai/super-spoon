"""Unit tests for QdrantVectorStore.

Real qdrant-client in `:memory:` mode for the happy paths (no docker, no
testcontainers). Mocks are reserved for paths that `:memory:` can't reach,
specifically: validation-before-RPC and transport-failure mapping.
"""

import math
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.app.domain.models import Chunk, Hit
from backend.app.vectorstore.base import (
    CollectionMissingError,
    QdrantUnreachableError,
)
from backend.app.vectorstore.qdrant_store import QdrantVectorStore


VECTOR_SIZE = 4


def _chunk(idx: int, *, section_title: str = "Body", text: str = "the text") -> Chunk:
    return Chunk(
        chunk_index=idx,
        section_index=0,
        chunk_index_in_section=idx,
        section_title=section_title,
        text=text,
        char_offset_start=0,
        char_offset_end=len(text),
    )


def _norm(v: list[float]) -> list[float]:
    """Unit-normalise so cosine scores in tests are easy to reason about."""
    mag = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / mag for x in v]


@pytest.fixture
def fresh_client() -> QdrantClient:
    """Per-test in-memory client — avoids cross-test collection bleed."""
    return QdrantClient(":memory:")


@pytest.fixture
def store(fresh_client: QdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(fresh_client)


class TestReset:
    def test_creates_collection_when_absent(
        self, store: QdrantVectorStore
    ) -> None:
        assert store.collection_exists("c") is False

        store.reset("c", vector_size=VECTOR_SIZE, metric="cosine")

        assert store.collection_exists("c") is True

    def test_drops_then_recreates_when_present(
        self, store: QdrantVectorStore
    ) -> None:
        # Establish state: a populated collection.
        store.reset("c", vector_size=VECTOR_SIZE)
        store.upsert(
            "c",
            chunks=[_chunk(0)],
            embeddings=[_norm([1.0, 0.0, 0.0, 0.0])],
        )
        # Sanity: the point is in there.
        hits_before = store.search("c", _norm([1.0, 0.0, 0.0, 0.0]), k=4)
        assert len(hits_before) == 1

        # Reset — drop and recreate.
        store.reset("c", vector_size=VECTOR_SIZE)

        # Collection still exists but is empty.
        assert store.collection_exists("c") is True
        hits_after = store.search("c", _norm([1.0, 0.0, 0.0, 0.0]), k=4)
        assert hits_after == []

    def test_unknown_metric_rejected(self, store: QdrantVectorStore) -> None:
        with pytest.raises(ValueError, match="unknown similarity metric"):
            store.reset("c", vector_size=VECTOR_SIZE, metric="hamming")

    def test_metric_aliases_accepted(self, store: QdrantVectorStore) -> None:
        # All three should round-trip without raising; verify by collection_exists.
        for metric in ("cosine", "euclid", "l2", "dot"):
            store.reset(f"c-{metric}", vector_size=VECTOR_SIZE, metric=metric)
            assert store.collection_exists(f"c-{metric}") is True


class TestUpsert:
    def test_length_mismatch_raises_value_error_before_any_rpc(self) -> None:
        # MagicMock — if the wrapper RPCs anything, the assertion below catches it.
        mock_client = MagicMock(spec=QdrantClient)
        store = QdrantVectorStore(mock_client)

        with pytest.raises(ValueError, match="length mismatch"):
            store.upsert(
                "c",
                chunks=[_chunk(0), _chunk(1)],
                embeddings=[_norm([1.0, 0, 0, 0])],
            )

        mock_client.upsert.assert_not_called()
        # No other client methods should have been invoked either.
        assert mock_client.method_calls == []

    def test_empty_input_is_a_noop(self, store: QdrantVectorStore) -> None:
        store.reset("c", vector_size=VECTOR_SIZE)

        # No-op: doesn't crash and doesn't add anything.
        store.upsert("c", chunks=[], embeddings=[])

        hits = store.search("c", _norm([1.0, 0, 0, 0]), k=4)
        assert hits == []

    def test_point_id_equals_chunk_index(self, store: QdrantVectorStore) -> None:
        store.reset("c", vector_size=VECTOR_SIZE)
        chunks = [_chunk(7), _chunk(42)]
        embeddings = [_norm([1.0, 0, 0, 0]), _norm([0, 1.0, 0, 0])]

        store.upsert("c", chunks=chunks, embeddings=embeddings)

        # Query by the exact vector of chunk 42 — top hit must be id=42.
        hits = store.search("c", _norm([0, 1.0, 0, 0]), k=2)
        assert hits[0].chunk_index == 42

    def test_payload_carries_every_chunk_field_per_u14(
        self, store: QdrantVectorStore
    ) -> None:
        # U14: each chunk's stored metadata includes the chunk text itself so
        # the sources[].excerpt in F8 doesn't need a second lookup.
        store.reset("c", vector_size=VECTOR_SIZE)
        chunk = Chunk(
            chunk_index=3,
            section_index=1,
            chunk_index_in_section=2,
            section_title="Light-dependent reactions",
            text="Chlorophyll absorbs visible light.",
            char_offset_start=100,
            char_offset_end=135,
        )
        vector = _norm([1.0, 0, 0, 0])

        store.upsert("c", chunks=[chunk], embeddings=[vector])

        hits = store.search("c", vector, k=1)
        assert len(hits) == 1
        # chunk_index, section_title, text are all reachable via Hit (the F8
        # source-rendering path); we sanity-check them here directly.
        assert hits[0].chunk_index == 3
        assert hits[0].section_title == "Light-dependent reactions"
        assert hits[0].text == "Chlorophyll absorbs visible light."


class TestSearch:
    def test_returns_hits_ordered_by_score_descending(
        self, store: QdrantVectorStore
    ) -> None:
        store.reset("c", vector_size=VECTOR_SIZE)
        # Three chunks with vectors at varying cosine similarity to the query.
        chunks = [_chunk(0), _chunk(1), _chunk(2)]
        embeddings = [
            _norm([1.0, 0.0, 0.0, 0.0]),  # exact match → score 1.0
            _norm([0.9, 0.1, 0.0, 0.0]),  # close
            _norm([0.0, 1.0, 0.0, 0.0]),  # orthogonal → score 0.0
        ]
        store.upsert("c", chunks=chunks, embeddings=embeddings)

        hits = store.search("c", _norm([1.0, 0.0, 0.0, 0.0]), k=3)

        assert len(hits) == 3
        # Score-desc ordering, ties broken by Qdrant internals.
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        # The chunk with the exact vector is the top hit.
        assert hits[0].chunk_index == 0

    def test_limit_k_caps_results(self, store: QdrantVectorStore) -> None:
        store.reset("c", vector_size=VECTOR_SIZE)
        chunks = [_chunk(i) for i in range(10)]
        embeddings = [_norm([1.0, float(i) / 10, 0, 0]) for i in range(10)]
        store.upsert("c", chunks=chunks, embeddings=embeddings)

        hits = store.search("c", _norm([1.0, 0, 0, 0]), k=4)

        assert len(hits) == 4

    def test_returns_empty_on_collection_with_no_points(
        self, store: QdrantVectorStore
    ) -> None:
        store.reset("c", vector_size=VECTOR_SIZE)

        hits = store.search("c", _norm([1.0, 0, 0, 0]), k=4)

        assert hits == []

    def test_missing_collection_raises_collection_missing_error(
        self, store: QdrantVectorStore
    ) -> None:
        with pytest.raises(CollectionMissingError) as exc_info:
            store.search("never_reset", _norm([1.0, 0, 0, 0]), k=4)

        assert exc_info.value.collection == "never_reset"

    def test_returns_hits_with_score_as_float(
        self, store: QdrantVectorStore
    ) -> None:
        store.reset("c", vector_size=VECTOR_SIZE)
        store.upsert(
            "c",
            chunks=[_chunk(0)],
            embeddings=[_norm([1.0, 0, 0, 0])],
        )

        hits = store.search("c", _norm([1.0, 0, 0, 0]), k=1)

        assert isinstance(hits[0], Hit)
        assert isinstance(hits[0].score, float)


class TestCollectionExists:
    def test_returns_false_for_missing_not_an_error(
        self, store: QdrantVectorStore
    ) -> None:
        assert store.collection_exists("never_made") is False

    def test_returns_true_after_reset(self, store: QdrantVectorStore) -> None:
        store.reset("c", vector_size=VECTOR_SIZE)

        assert store.collection_exists("c") is True

    def test_returns_false_after_delete_via_reset_cycle(
        self, store: QdrantVectorStore, fresh_client: QdrantClient
    ) -> None:
        store.reset("c", vector_size=VECTOR_SIZE)
        fresh_client.delete_collection("c")

        assert store.collection_exists("c") is False


class TestTransportFailures:
    """Mocks substitute for `:memory:` here — local-mode never fails with a
    transport error, so we drive the remote-mode failure paths via spec=QdrantClient.
    """

    def test_collection_exists_transport_failure_wrapped(self) -> None:
        mock_client = MagicMock(spec=QdrantClient)
        mock_client.collection_exists.side_effect = OSError("connection refused")
        store = QdrantVectorStore(mock_client)

        with pytest.raises(QdrantUnreachableError):
            store.collection_exists("c")

    def test_reset_transport_failure_wrapped(self) -> None:
        mock_client = MagicMock(spec=QdrantClient)
        mock_client.collection_exists.return_value = False
        mock_client.create_collection.side_effect = OSError("connection refused")
        store = QdrantVectorStore(mock_client)

        with pytest.raises(QdrantUnreachableError):
            store.reset("c", vector_size=VECTOR_SIZE)

    def test_upsert_transport_failure_wrapped(self) -> None:
        mock_client = MagicMock(spec=QdrantClient)
        mock_client.upsert.side_effect = OSError("connection refused")
        store = QdrantVectorStore(mock_client)

        with pytest.raises(QdrantUnreachableError):
            store.upsert(
                "c",
                chunks=[_chunk(0)],
                embeddings=[_norm([1.0, 0, 0, 0])],
            )

    def test_search_404_response_becomes_collection_missing(self) -> None:
        # Remote mode (HTTP) raises UnexpectedResponse with status_code=404 on
        # missing collection; the wrapper must convert to CollectionMissingError
        # to match the `:memory:` ValueError("Collection X not found") path.
        mock_client = MagicMock(spec=QdrantClient)
        mock_client.query_points.side_effect = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"",
            headers=None,
        )
        store = QdrantVectorStore(mock_client)

        with pytest.raises(CollectionMissingError):
            store.search("never_made", _norm([1.0, 0, 0, 0]), k=4)

    def test_search_non_404_response_becomes_unreachable(self) -> None:
        mock_client = MagicMock(spec=QdrantClient)
        mock_client.query_points.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Server Error",
            content=b"",
            headers=None,
        )
        store = QdrantVectorStore(mock_client)

        with pytest.raises(QdrantUnreachableError):
            store.search("c", _norm([1.0, 0, 0, 0]), k=4)
