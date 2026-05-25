"""Embedder Protocol and shared exception.

NF4 swappable abstraction: any implementation that satisfies this Protocol can
replace the SBERT one via env (`EMBEDDING_MODEL`). The Protocol is sync — DESIGN
§3 / §6.1 deliberately surfaces the CPU-bound nature at the call site so route
handlers know to offload to a thread pool rather than awaiting.
"""

from typing import Protocol


class EmbeddingError(Exception):
    """Raised when the underlying embedder model fails.

    Not expected in practice for the SBERT path; surfaces here so callers
    don't need to know about sentence-transformers exception types. T10a
    re-parents this into the DomainError hierarchy.
    """


class Embedder(Protocol):
    @property
    def vector_size(self) -> int:
        """Embedding dimensionality (e.g. 384 for bge-small-en-v1.5)."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text in input order.

        Empty input returns []. Synchronous and CPU-bound — callers offload via
        `anyio.to_thread.run_sync` to avoid blocking the event loop.
        Raises: EmbeddingError on model failure.
        """
        ...
