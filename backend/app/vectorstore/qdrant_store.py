"""Qdrant-backed VectorStore implementation.

Wraps the official qdrant-client (sync). Owns the lifecycle of the single
`active_article` collection (DESIGN §4.4 / A7). `point.id` is the chunk's
global `chunk_index`; payload carries every Chunk field including `text` per
U14 so retrieval is a single hop.

Transport failures bubble up as QdrantUnreachableError. A search against a
collection that doesn't exist surfaces as CollectionMissingError so the chat
route can map it to the A12 `no_active_article` user-facing error rather than
a generic "vector store unavailable".
"""

from dataclasses import asdict

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException, UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.app.domain.models import Chunk, Hit
from backend.app.vectorstore.base import (
    CollectionMissingError,
    QdrantUnreachableError,
)

_METRIC_TO_DISTANCE = {
    "cosine": Distance.COSINE,
    "euclid": Distance.EUCLID,
    "l2": Distance.EUCLID,
    "dot": Distance.DOT,
}

# Local-mode (`:memory:`) raises ValueError on missing collection; remote-mode
# raises UnexpectedResponse with status 404. We match the local-mode error by
# message and the remote-mode by status.
_NOT_FOUND_MESSAGE_FRAGMENT = "not found"


class QdrantVectorStore:
    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def reset(
        self, collection: str, vector_size: int, metric: str = "cosine"
    ) -> None:
        distance = self._resolve_distance(metric)
        try:
            if self._client.collection_exists(collection):
                self._client.delete_collection(collection)
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
        except (ApiException, UnexpectedResponse, OSError) as exc:
            raise QdrantUnreachableError(
                f"reset({collection!r}) failed"
            ) from exc

    def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        # Validate BEFORE touching the client — a length mismatch is a caller
        # bug, not a transport failure, and we must not waste an RPC on it.
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        if not chunks:
            return

        points = [
            PointStruct(
                id=chunk.chunk_index,
                vector=embedding,
                payload=asdict(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        try:
            self._client.upsert(collection_name=collection, points=points)
        except (ApiException, UnexpectedResponse, OSError) as exc:
            raise QdrantUnreachableError(
                f"upsert({collection!r}) failed"
            ) from exc

    def search(
        self, collection: str, embedding: list[float], k: int
    ) -> list[Hit]:
        try:
            response = self._client.query_points(
                collection_name=collection,
                query=embedding,
                limit=k,
            )
        except ValueError as exc:
            if _NOT_FOUND_MESSAGE_FRAGMENT in str(exc).lower():
                raise CollectionMissingError(collection) from exc
            raise
        except UnexpectedResponse as exc:
            if getattr(exc, "status_code", None) == 404:
                raise CollectionMissingError(collection) from exc
            raise QdrantUnreachableError(
                f"search({collection!r}) failed"
            ) from exc
        except (ApiException, OSError) as exc:
            raise QdrantUnreachableError(
                f"search({collection!r}) failed"
            ) from exc

        return [
            Hit(
                chunk_index=int(point.id),
                section_title=(point.payload or {}).get("section_title", ""),
                text=(point.payload or {}).get("text", ""),
                score=float(point.score),
            )
            for point in response.points
        ]

    def collection_exists(self, collection: str) -> bool:
        try:
            return bool(self._client.collection_exists(collection))
        except (ApiException, UnexpectedResponse, OSError) as exc:
            raise QdrantUnreachableError(
                f"collection_exists({collection!r}) failed"
            ) from exc

    @staticmethod
    def _resolve_distance(metric: str) -> Distance:
        key = metric.lower()
        if key not in _METRIC_TO_DISTANCE:
            raise ValueError(
                f"unknown similarity metric {metric!r}; "
                f"expected one of {sorted(_METRIC_TO_DISTANCE)}"
            )
        return _METRIC_TO_DISTANCE[key]
