"""Unit tests for RAGOrchestrator.

The orchestrator's job is wiring, not arithmetic — so tests verify wiring:
the A12 gate fires before any work, retrieval-empty short-circuits before the
LLM, hits flow into sources VERBATIM (not via the LLM output), and chat_num_ctx
reaches the LLM call. Fakes are full Protocol implementations so a misuse of
the contract (e.g. accidentally awaiting a sync method) fails loudly here.
"""

import pytest

from backend.app.domain.models import ChatResponse, Hit, Source
from backend.app.llm.prompts import CHAT_GROUNDED
from backend.app.rag.orchestrator import (
    NoActiveArticleError,
    RAGOrchestrator,
    RetrievalEmptyError,
)
from tests.fakes import FakeEmbedder, FakeLLM, FakeVectorStore

COLLECTION = "active_article"
TOP_K = 4
CHAT_NUM_CTX = 4096


def _build_orchestrator(
    *,
    embedder: FakeEmbedder | None = None,
    vector_store: FakeVectorStore | None = None,
    llm: FakeLLM | None = None,
    top_k: int = TOP_K,
    chat_num_ctx: int = CHAT_NUM_CTX,
) -> tuple[RAGOrchestrator, FakeEmbedder, FakeVectorStore, FakeLLM]:
    embedder = embedder or FakeEmbedder()
    vector_store = vector_store or FakeVectorStore()
    llm = llm or FakeLLM(canned_response="some grounded answer.")
    orch = RAGOrchestrator(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        collection=COLLECTION,
        top_k=top_k,
        chat_prompt_template=CHAT_GROUNDED,
        chat_num_ctx=chat_num_ctx,
    )
    return orch, embedder, vector_store, llm


def _hit(idx: int, *, section_title: str = "Body", text: str = "chunk text", score: float | None = None) -> Hit:
    return Hit(
        chunk_index=idx,
        section_title=section_title,
        text=text,
        score=score if score is not None else 1.0 - idx * 0.1,
    )


class TestA12Gate:
    async def test_missing_collection_raises_no_active_article_error(self) -> None:
        orch, _, _, _ = _build_orchestrator()
        # FakeVectorStore is fresh — no reset, no preload — so collection_exists is False.

        with pytest.raises(NoActiveArticleError) as exc_info:
            await orch.answer("anything")

        assert exc_info.value.collection == COLLECTION

    async def test_embedder_not_called_on_a12_path(self) -> None:
        orch, embedder, _, _ = _build_orchestrator()

        with pytest.raises(NoActiveArticleError):
            await orch.answer("anything")

        assert embedder.calls == []

    async def test_llm_not_called_on_a12_path(self) -> None:
        orch, _, _, llm = _build_orchestrator()

        with pytest.raises(NoActiveArticleError):
            await orch.answer("anything")

        assert llm.calls == []


class TestRetrievalEmptyPath:
    async def test_empty_hits_raises_retrieval_empty_error(self) -> None:
        store = FakeVectorStore()
        store.reset(COLLECTION, vector_size=384)  # collection exists but has zero points
        orch, _, _, _ = _build_orchestrator(vector_store=store)

        with pytest.raises(RetrievalEmptyError):
            await orch.answer("anything")

    async def test_llm_not_called_when_retrieval_is_empty(self) -> None:
        store = FakeVectorStore()
        store.reset(COLLECTION, vector_size=384)
        orch, _, _, llm = _build_orchestrator(vector_store=store)

        with pytest.raises(RetrievalEmptyError):
            await orch.answer("anything")

        assert llm.calls == []


class TestHappyPath:
    async def test_returns_chat_response_with_four_sources(self) -> None:
        store = FakeVectorStore()
        hits = [
            _hit(10, section_title="Light-dependent reactions", text="content A", score=0.9),
            _hit(11, section_title="Pigments", text="content B", score=0.7),
            _hit(12, section_title="History", text="content C", score=0.5),
            _hit(13, section_title="Reception", text="content D", score=0.3),
        ]
        store.preload_hits(COLLECTION, hits)
        orch, _, _, llm = _build_orchestrator(vector_store=store)

        response = await orch.answer("What pigment captures light?")

        assert isinstance(response, ChatResponse)
        assert response.answer == "some grounded answer."
        assert len(response.sources) == 4

    async def test_sources_preserve_hit_metadata_end_to_end(self) -> None:
        store = FakeVectorStore()
        hits = [
            _hit(10, section_title="Light-dependent reactions", text="content A", score=0.9),
            _hit(11, section_title="Pigments", text="content B", score=0.7),
            _hit(12, section_title="History", text="content C", score=0.5),
            _hit(13, section_title="Reception", text="content D", score=0.3),
        ]
        store.preload_hits(COLLECTION, hits)
        orch, _, _, _ = _build_orchestrator(vector_store=store)

        response = await orch.answer("What pigment captures light?")

        # Each Source mirrors the corresponding Hit by index — chunk_index,
        # section_title, score all flow through verbatim.
        for src, hit in zip(response.sources, hits):
            assert isinstance(src, Source)
            assert src.chunk_index == hit.chunk_index
            assert src.section_title == hit.section_title
            assert src.score == hit.score

    async def test_sources_in_score_desc_order_matching_hits(self) -> None:
        store = FakeVectorStore()
        hits = [_hit(i, score=1.0 - i * 0.1) for i in range(4)]
        store.preload_hits(COLLECTION, hits)
        orch, _, _, _ = _build_orchestrator(vector_store=store)

        response = await orch.answer("question")

        scores = [s.score for s in response.sources]
        assert scores == sorted(scores, reverse=True)

    async def test_excerpt_is_hit_text_first_200_chars(self) -> None:
        long_text = "x" * 500
        store = FakeVectorStore()
        store.preload_hits(COLLECTION, [_hit(0, text=long_text)])
        orch, _, _, _ = _build_orchestrator(
            vector_store=store, top_k=1
        )

        response = await orch.answer("question")

        assert response.sources[0].excerpt == long_text[:200]
        assert len(response.sources[0].excerpt) == 200

    async def test_excerpt_is_shorter_than_200_when_text_is_short(self) -> None:
        short_text = "short text"
        store = FakeVectorStore()
        store.preload_hits(COLLECTION, [_hit(0, text=short_text)])
        orch, _, _, _ = _build_orchestrator(vector_store=store, top_k=1)

        response = await orch.answer("question")

        assert response.sources[0].excerpt == short_text

    async def test_answer_is_stripped(self) -> None:
        store = FakeVectorStore()
        store.preload_hits(COLLECTION, [_hit(0)])
        llm = FakeLLM(canned_response="   answer with whitespace   ")
        orch, _, _, _ = _build_orchestrator(
            vector_store=store, llm=llm, top_k=1
        )

        response = await orch.answer("question")

        assert response.answer == "answer with whitespace"


class TestPromptShape:
    async def test_prompt_contains_numbered_chunks_in_score_order(self) -> None:
        store = FakeVectorStore()
        hits = [
            _hit(10, section_title="Light-dependent reactions", text="alpha alpha alpha"),
            _hit(11, section_title="Pigments", text="beta beta beta"),
            _hit(12, section_title="History", text="gamma gamma gamma"),
            _hit(13, section_title="Reception", text="delta delta delta"),
        ]
        store.preload_hits(COLLECTION, hits)
        orch, _, _, llm = _build_orchestrator(vector_store=store)

        await orch.answer("question text")

        prompt = llm.calls[0]["prompt"]
        # Numbered chunks: [1], [2], [3], [4] in score order with their titles.
        assert "[1] Light-dependent reactions\nalpha alpha alpha" in prompt
        assert "[2] Pigments\nbeta beta beta" in prompt
        assert "[3] History\ngamma gamma gamma" in prompt
        assert "[4] Reception\ndelta delta delta" in prompt
        # The numbering preserves hit order — [1] comes before [2] in the prompt.
        assert prompt.index("[1]") < prompt.index("[2]") < prompt.index("[3]") < prompt.index("[4]")

    async def test_prompt_carries_question_text_verbatim(self) -> None:
        store = FakeVectorStore()
        store.preload_hits(COLLECTION, [_hit(0)])
        orch, _, _, llm = _build_orchestrator(vector_store=store, top_k=1)

        await orch.answer("What pigment captures light in chloroplasts?")

        prompt = llm.calls[0]["prompt"]
        assert "What pigment captures light in chloroplasts?" in prompt


class TestNumCtxWiring:
    async def test_chat_num_ctx_reaches_llm_generate(self) -> None:
        store = FakeVectorStore()
        store.preload_hits(COLLECTION, [_hit(0)])
        orch, _, _, llm = _build_orchestrator(
            vector_store=store, chat_num_ctx=8192, top_k=1
        )

        await orch.answer("question")

        assert llm.calls[0]["num_ctx"] == 8192


class TestSourcesFromRetrieverNotLLM:
    """F8 / OQ-5 / A17 / U8: citations come from the retriever, not the LLM.
    These tests are the bedrock of the no-hallucinated-citations claim.
    """

    async def test_sources_populated_when_llm_says_not_found_in_article(
        self,
    ) -> None:
        store = FakeVectorStore()
        hits = [
            _hit(10, section_title="Light-dependent reactions", text="A"),
            _hit(11, section_title="Pigments", text="B"),
            _hit(12, section_title="History", text="C"),
            _hit(13, section_title="Reception", text="D"),
        ]
        store.preload_hits(COLLECTION, hits)
        llm = FakeLLM(canned_response="not found in the article")
        orch, _, _, _ = _build_orchestrator(vector_store=store, llm=llm)

        response = await orch.answer("Who wrote Hamlet?")

        # The decline is what the user sees — but sources are still all four.
        assert response.answer == "not found in the article"
        assert len(response.sources) == 4
        assert [s.chunk_index for s in response.sources] == [10, 11, 12, 13]

    async def test_sources_unaffected_by_llm_prompt_format(self) -> None:
        # Even if the LLM's response somehow contained citation tags like [1]
        # or section names, those would NOT appear in sources — sources come
        # from hits, period.
        store = FakeVectorStore()
        hits = [
            _hit(10, section_title="Section A", text="content"),
            _hit(11, section_title="Section B", text="content"),
        ]
        store.preload_hits(COLLECTION, hits)
        # Canned response mentions a totally different "section" — would never
        # leak into the sources list.
        llm = FakeLLM(canned_response="from [99] FabricatedSection: an answer")
        orch, _, _, _ = _build_orchestrator(
            vector_store=store, llm=llm, top_k=2
        )

        response = await orch.answer("question")

        # Sources reflect the hits' section_titles, not the LLM's prose.
        titles = {s.section_title for s in response.sources}
        assert titles == {"Section A", "Section B"}
        assert "FabricatedSection" not in titles


class TestTopKThreadedThrough:
    async def test_top_k_constructor_value_passed_to_search(self) -> None:
        store = FakeVectorStore()
        store.reset(COLLECTION, vector_size=384)
        # Preload nothing — search would otherwise return [], hitting
        # RetrievalEmptyError. Instead, populate via upsert so search runs the
        # cosine path AND records the k argument.
        embedder = FakeEmbedder()
        # Add 6 chunks so we can verify k caps results.
        from backend.app.domain.models import Chunk
        chunks = [
            Chunk(
                chunk_index=i,
                section_index=0,
                chunk_index_in_section=i,
                section_title=f"S{i}",
                text=f"text {i}",
                char_offset_start=0,
                char_offset_end=10,
            )
            for i in range(6)
        ]
        store.upsert(
            COLLECTION,
            chunks=chunks,
            embeddings=embedder.embed_texts([c.text for c in chunks]),
        )
        orch, _, _, _ = _build_orchestrator(
            embedder=embedder, vector_store=store, top_k=3
        )

        response = await orch.answer("question")

        assert store.search_calls[0]["k"] == 3
        assert len(response.sources) == 3
