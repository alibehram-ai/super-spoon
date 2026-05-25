"""RAGOrchestrator per DESIGN §3 / §4.6.

Composes Embedder + VectorStore + LLMClient for the chat path. The seven-step
flow:

  1. Gate on `vector_store.collection_exists(collection)` — if False, surface
     A12 (`NoActiveArticleError`) before doing any work. The embedder and the
     LLM are NOT called in this path.
  2. Embed the question.
  3. Retrieve top-K hits.
  4. If no hits, raise `RetrievalEmptyError` — the LLM is NOT called.
  5. Build a numbered chunks block (`[1] <section>\\n<text>\\n\\n[2] ...`).
  6. Render `chat_prompt_template.format(chunks=..., question=...)` and call
     `llm.generate(prompt, num_ctx=chat_num_ctx)`.
  7. Build `Source` objects from the HITS (not the LLM output) so citations
     can't be hallucinated — F8 / OQ-5 / A17 / U8.

`chat_num_ctx` is wired via constructor (no default at the call site, mirroring
the LLMClient's required-num_ctx contract).
"""

from backend.app.api.errors import NoActiveArticleError, RetrievalEmptyError
from backend.app.domain.models import ChatResponse, Source
from backend.app.embedding.base import Embedder
from backend.app.llm.base import LLMClient
from backend.app.vectorstore.base import VectorStore

__all__ = ["NoActiveArticleError", "RAGOrchestrator", "RetrievalEmptyError"]

_EXCERPT_CHARS = 200


class RAGOrchestrator:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        llm: LLMClient,
        collection: str,
        top_k: int,
        chat_prompt_template: str,
        chat_num_ctx: int,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm = llm
        self._collection = collection
        self._top_k = top_k
        self._chat_prompt_template = chat_prompt_template
        self._chat_num_ctx = chat_num_ctx

    async def answer(self, question: str) -> ChatResponse:
        if not self._vector_store.collection_exists(self._collection):
            raise NoActiveArticleError(self._collection)

        query_vector = self._embedder.embed_texts([question])[0]
        hits = self._vector_store.search(
            self._collection, query_vector, k=self._top_k
        )
        if not hits:
            raise RetrievalEmptyError()

        chunks_block = "\n\n".join(
            f"[{i + 1}] {hit.section_title}\n{hit.text}"
            for i, hit in enumerate(hits)
        )
        prompt = self._chat_prompt_template.format(
            chunks=chunks_block, question=question
        )
        answer = await self._llm.generate(prompt, num_ctx=self._chat_num_ctx)

        # Sources are derived from the retriever's hits — NEVER from the LLM
        # output. This is what guarantees F8 citations are unhallucinable and
        # what makes A17/U8 (sources populated even when the model declines)
        # mechanically true.
        sources = [
            Source(
                section_title=hit.section_title,
                chunk_index=hit.chunk_index,
                score=hit.score,
                excerpt=hit.text[:_EXCERPT_CHARS],
            )
            for hit in hits
        ]
        return ChatResponse(answer=answer.strip(), sources=sources)
