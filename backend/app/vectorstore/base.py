"""VectorStore Protocol and shared exceptions.

NF4 swappable abstraction: any implementation that satisfies this Protocol can
replace QdrantVectorStore via dependency injection. The Protocol is sync —
route handlers offload via `anyio.to_thread.run_sync` (DESIGN §6.1).
"""

from typing import Protocol

from backend.app.domain.models import Chunk, Hit


class QdrantUnreachableError(Exception):
    """Raised when the vector store can't be reached (connection/transport).

    Mapped to HTTP 503 (`vector_store_unavailable`) by the T10a handler.
    """


class CollectionMissingError(Exception):
    """Raised when a search targets a collection that does not exist.

    Distinct from QdrantUnreachableError — the store is reachable but the
    named collection isn't there. The chat route uses `collection_exists()`
    upstream to surface this as the A12 `no_active_article` case to users.
    """

    def __init__(self, collection: str) -> None:
        super().__init__(f"Qdrant collection {collection!r} does not exist.")
        self.collection = collection


class VectorStore(Protocol):
    def reset(self, collection: str, vector_size: int, metric: str = "cosine") -> None:
        """Drop the collection if it exists, then create it (F6 / A7).

        Raises: QdrantUnreachableError on transport failure.
        """
        ...

    def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Write chunks + embeddings as Qdrant points.

        `point.id == chunk.chunk_index`; payload carries every Chunk field
        (including `text`, per U14, so retrieval doesn't need a second hop).
        `len(chunks)` must equal `len(embeddings)` — length mismatch raises
        ValueError BEFORE any RPC.

        Raises: ValueError on length mismatch; QdrantUnreachableError on
        transport failure.
        """
        ...

    def search(self, collection: str, embedding: list[float], k: int) -> list[Hit]:
        """Return up to k hits ordered by score descending.

        Returns [] if the collection exists but has no matching points.
        Raises: CollectionMissingError if the collection does not exist;
        QdrantUnreachableError on transport failure.
        """
        ...

    def collection_exists(self, collection: str) -> bool:
        """True iff the named collection exists. Missing is not an error.

        Raises: QdrantUnreachableError on transport failure.
        """
        ...
