"""sentence-transformers wrapper implementing the Embedder Protocol.

Loads the configured model once at construction (so the FastAPI lifespan
controls when the load happens — see DESIGN §4.3 / §1 lifespan). `embed_texts`
batches the input list at `batch_size` and concatenates the per-batch outputs;
this makes the batching contract observable from tests (each batch is a
separate `.encode` call) without giving up the model's own internal batching.
"""

from sentence_transformers import SentenceTransformer

from backend.app.embedding.base import EmbeddingError


class SentenceTransformersEmbedder:
    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive (got {batch_size})")
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    @property
    def vector_size(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                vectors = self._model.encode(batch, batch_size=self._batch_size)
            except Exception as exc:
                raise EmbeddingError(
                    f"embedding model {self._model_name!r} failed on batch starting at {start}"
                ) from exc
            for v in vectors:
                out.append([float(x) for x in v])
        return out
