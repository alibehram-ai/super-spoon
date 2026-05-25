"""Test fakes that implement the LLMClient / VectorStore / Embedder Protocols.

These are full fakes (DESIGN §7.1), not mocks-with-asserts: callers can drive
them with realistic inputs and observe recorded state. T08 introduces FakeLLM;
T09 adds FakeVectorStore and FakeEmbedder alongside.
"""

import hashlib
import math
from typing import Any

from backend.app.domain.models import Chunk, Hit
from backend.app.vectorstore.base import CollectionMissingError


class FakeLLM:
    """Records every generate() call and returns a canned response.

    Set `raise_on_generate` to an exception instance to simulate the
    OllamaClient's error paths (OllamaUnreachableError, OllamaTimeoutError,
    LLMResponseInvalidError). The fake never validates output shape itself —
    if a test wants "empty response" semantics, configure `raise_on_generate`
    to LLMResponseInvalidError (mirroring what the real client does upstream).
    """

    def __init__(
        self,
        *,
        canned_response: str = "a four to eight sentence summary",
        raise_on_generate: BaseException | None = None,
    ) -> None:
        self.canned_response = canned_response
        self.raise_on_generate = raise_on_generate
        self.calls: list[dict] = []
        self.warmup_calls: int = 0

    async def generate(
        self,
        prompt: str,
        *,
        num_ctx: int,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "num_ctx": num_ctx,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
            }
        )
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        return self.canned_response

    async def warmup(self) -> bool:
        self.warmup_calls += 1
        return True


class FakeEmbedder:
    """Deterministic SHA-256-derived vectors. Distinct texts produce distinct
    vectors; same text always produces the same vector. No model load. Records
    every embed_texts call so orchestrator tests can assert "embedder was not
    called" on the A12 path.
    """

    def __init__(self, vector_size: int = 384) -> None:
        self._vector_size = vector_size
        self.calls: list[list[str]] = []

    @property
    def vector_size(self) -> int:
        return self._vector_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector_for(t) for t in texts]

    def _vector_for(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Stretch to vector_size bytes by cycling the digest.
        material = (digest * ((self._vector_size // len(digest)) + 1))[
            : self._vector_size
        ]
        return [b / 255.0 for b in material]


class FakeVectorStore:
    """Dict-backed VectorStore with a real cosine search path AND an optional
    `preload_hits()` override that bypasses storage for tests that want to
    control the exact Hit objects returned (orchestrator tests use this so they
    don't have to fabricate vectors that produce specific scores).
    """

    def __init__(self) -> None:
        self._collections: dict[str, dict[int, tuple[list[float], dict[str, Any]]]] = {}
        self._preloaded_hits: dict[str, list[Hit]] = {}
        self.reset_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []

    def reset(
        self, collection: str, vector_size: int, metric: str = "cosine"
    ) -> None:
        self.reset_calls.append(
            {"collection": collection, "vector_size": vector_size, "metric": metric}
        )
        self._collections[collection] = {}

    def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        self.upsert_calls.append(
            {"collection": collection, "n_chunks": len(chunks)}
        )
        bucket = self._collections.setdefault(collection, {})
        for chunk, vector in zip(chunks, embeddings):
            bucket[chunk.chunk_index] = (
                list(vector),
                {
                    "chunk_index": chunk.chunk_index,
                    "section_title": chunk.section_title,
                    "text": chunk.text,
                },
            )

    def search(
        self, collection: str, embedding: list[float], k: int
    ) -> list[Hit]:
        self.search_calls.append(
            {"collection": collection, "embedding": embedding, "k": k}
        )
        if collection in self._preloaded_hits:
            return self._preloaded_hits[collection][:k]
        if collection not in self._collections:
            raise CollectionMissingError(collection)
        points = self._collections[collection]
        hits = [
            Hit(
                chunk_index=pid,
                section_title=payload.get("section_title", ""),
                text=payload.get("text", ""),
                score=_cosine(embedding, vec),
            )
            for pid, (vec, payload) in points.items()
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def collection_exists(self, collection: str) -> bool:
        return collection in self._collections or collection in self._preloaded_hits

    def preload_hits(self, collection: str, hits: list[Hit]) -> None:
        """Test-only: install Hit objects to return verbatim from search().
        Marks the collection as existing so the A12 gate passes.
        """
        self._preloaded_hits[collection] = list(hits)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
