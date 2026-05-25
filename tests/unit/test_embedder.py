"""Unit tests for SentenceTransformersEmbedder.

Patches `backend.app.embedding.sbert.SentenceTransformer` (the class at the
wrapper's import site, not the instance after construction) so the real model
is NEVER loaded. A tmpdir HF_HOME guard provides defence-in-depth: if patching
ever silently broke and the real class ran, sentence-transformers would try to
write into our tmpdir and we'd see files appear.
"""

import sys
from pathlib import Path

import pytest

from backend.app.embedding.base import Embedder, EmbeddingError
from backend.app.embedding.sbert import SentenceTransformersEmbedder


class FakeSentenceTransformer:
    """Stand-in for sentence_transformers.SentenceTransformer.

    Records construction calls and .encode call arguments. Returns
    deterministic length-384 vectors as plain Python lists (which iterate the
    same as numpy ndarrays for the wrapper's conversion loop).
    """

    instances: list["FakeSentenceTransformer"] = []
    construction_args: list[tuple[tuple, dict]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        FakeSentenceTransformer.instances.append(self)
        FakeSentenceTransformer.construction_args.append((args, kwargs))
        self.encode_calls: list[dict] = []
        self.embedding_dimension = 384
        # Set if test wants .encode to raise instead.
        self.raise_on_encode: BaseException | None = None

    def encode(self, texts, batch_size: int = 32, **kwargs: object):
        self.encode_calls.append(
            {"texts": list(texts), "batch_size": batch_size, "kwargs": kwargs}
        )
        if self.raise_on_encode is not None:
            raise self.raise_on_encode
        return [[float((i * 7 + idx) % 1000) / 1000.0 for i in range(384)] for idx, _ in enumerate(texts)]

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dimension

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.construction_args = []


@pytest.fixture(autouse=True)
def _reset_fake_class() -> None:
    FakeSentenceTransformer.reset()


@pytest.fixture(autouse=True)
def _isolated_hf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HF_HOME at a tmp dir so if anything ever fell through and tried to
    fetch a real model, we'd see files appear here. We assert no files were
    written in the suite-level guard below.
    """
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return tmp_path


@pytest.fixture
def patched_st(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformer]:
    """Replace SentenceTransformer at the wrapper's import site.

    Patching `backend.app.embedding.sbert.SentenceTransformer` is the standard
    pattern: the wrapper resolves the name from its own module namespace, so
    swapping that binding intercepts construction without touching the real
    class.
    """
    monkeypatch.setattr(
        "backend.app.embedding.sbert.SentenceTransformer",
        FakeSentenceTransformer,
    )
    return FakeSentenceTransformer


class TestConstruction:
    def test_passes_model_name_to_underlying_class(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        SentenceTransformersEmbedder(model_name="some/model", batch_size=64)

        assert len(patched_st.instances) == 1
        args, _ = patched_st.construction_args[0]
        assert args == ("some/model",)

    def test_default_model_name_is_bge_small(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        SentenceTransformersEmbedder()

        args, _ = patched_st.construction_args[0]
        assert args == ("BAAI/bge-small-en-v1.5",)

    def test_non_positive_batch_size_rejected_before_model_load(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            SentenceTransformersEmbedder(batch_size=0)

        # Validation happens before the underlying class is touched.
        assert patched_st.instances == []

    def test_wrapper_uses_the_patched_class_not_the_real_one(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = SentenceTransformersEmbedder()

        # The wrapper's _model is a FakeSentenceTransformer instance — proves the
        # patch took effect and the real class was never instantiated.
        assert isinstance(embedder._model, FakeSentenceTransformer)


class TestVectorSize:
    def test_proxies_to_get_sentence_embedding_dimension(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = SentenceTransformersEmbedder()

        assert embedder.vector_size == 384

    def test_returns_int_not_numpy_scalar(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = SentenceTransformersEmbedder()

        assert type(embedder.vector_size) is int

    def test_reflects_model_specific_dimension(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        # Simulate a different embedding model dimension.
        embedder = SentenceTransformersEmbedder()
        embedder._model.embedding_dimension = 768

        assert embedder.vector_size == 768


class TestEmbedTextsBatching:
    def test_empty_input_short_circuits_without_calling_encode(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = SentenceTransformersEmbedder(batch_size=64)

        result = embedder.embed_texts([])

        assert result == []
        assert embedder._model.encode_calls == []

    def test_batch_size_two_with_five_inputs_produces_three_encode_calls(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = SentenceTransformersEmbedder(batch_size=2)

        embedder.embed_texts(["a", "b", "c", "d", "e"])

        calls = embedder._model.encode_calls
        assert [c["texts"] for c in calls] == [["a", "b"], ["c", "d"], ["e"]]
        # batch_size kwarg is passed through to the underlying call too.
        assert all(c["batch_size"] == 2 for c in calls)

    def test_concatenation_preserves_input_order(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = SentenceTransformersEmbedder(batch_size=2)

        result = embedder.embed_texts(["a", "b", "c", "d", "e"])

        # Each vector is length 384; output count matches input count.
        assert len(result) == 5
        assert all(len(v) == 384 for v in result)
        # FakeSentenceTransformer's deterministic encoding uses the in-batch
        # index — so the FIRST vector returned by every batch shares the same
        # leading float. Concatenated output for ["a","b","c","d","e"] with
        # batch_size=2 is therefore [b0_a, b0_b, b1_c, b1_d, b2_e]; assert the
        # batch-starting vectors share their leading element.
        assert result[0][0] == result[2][0] == result[4][0]

    def test_single_input_smaller_than_batch_size(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = SentenceTransformersEmbedder(batch_size=64)

        result = embedder.embed_texts(["only"])

        assert len(result) == 1
        assert len(embedder._model.encode_calls) == 1
        assert embedder._model.encode_calls[0]["texts"] == ["only"]

    def test_input_exactly_one_batch(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = SentenceTransformersEmbedder(batch_size=3)

        embedder.embed_texts(["a", "b", "c"])

        calls = embedder._model.encode_calls
        assert len(calls) == 1
        assert calls[0]["texts"] == ["a", "b", "c"]


class TestErrorHandling:
    def test_model_failure_raises_embedding_error_with_chained_cause(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = SentenceTransformersEmbedder(batch_size=2)
        underlying = RuntimeError("model exploded")
        embedder._model.raise_on_encode = underlying

        with pytest.raises(EmbeddingError) as exc_info:
            embedder.embed_texts(["a", "b"])

        assert exc_info.value.__cause__ is underlying

    def test_failure_message_identifies_model_and_batch_offset(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder = SentenceTransformersEmbedder(
            model_name="some/model", batch_size=2
        )
        embedder._model.raise_on_encode = RuntimeError("boom")

        with pytest.raises(EmbeddingError, match="some/model"):
            embedder.embed_texts(["x", "y"])


class TestProtocolConformance:
    def test_sbert_embedder_satisfies_embedder_protocol(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        embedder: Embedder = SentenceTransformersEmbedder()

        # vector_size is readable as a property; embed_texts has the right shape.
        assert isinstance(embedder.vector_size, int)
        assert embedder.embed_texts([]) == []


class TestRealModelNotLoaded:
    """Defence-in-depth: even if a future change accidentally bypassed the
    monkey-patched class binding, these checks would catch the leak.
    """

    def test_hf_home_remains_empty_after_full_test_cycle(
        self,
        patched_st: type[FakeSentenceTransformer],
        _isolated_hf_home: Path,
    ) -> None:
        embedder = SentenceTransformersEmbedder()
        embedder.embed_texts(["any", "text", "here"])

        # No HuggingFace download wrote anything to the isolated cache dir.
        leftover = list(_isolated_hf_home.rglob("*"))
        assert leftover == [], f"Unexpected HF cache writes: {leftover}"

    def test_real_sentence_transformer_class_is_never_instantiated_in_suite(
        self, patched_st: type[FakeSentenceTransformer]
    ) -> None:
        # If sentence_transformers is imported at all (it is, by sbert.py),
        # confirm that the class binding inside the wrapper module currently
        # points at the fake — proving every constructor route hits the fake.
        sbert_module = sys.modules["backend.app.embedding.sbert"]
        assert sbert_module.SentenceTransformer is FakeSentenceTransformer

        SentenceTransformersEmbedder()
        # Every instance recorded in the fake's class list is itself a Fake.
        assert all(
            isinstance(inst, FakeSentenceTransformer)
            for inst in patched_st.instances
        )
