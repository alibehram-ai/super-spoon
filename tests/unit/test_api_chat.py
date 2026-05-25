"""Unit tests for ``POST /api/chat`` — T12.

The orchestrator's own unit suite (``test_orchestrator.py``) already proves
the seven-step flow; this suite verifies the HTTP wire shape and the chat-side
F9 → §5 error mapping. ``get_orchestrator`` is overridden via
``app.dependency_overrides`` with a real ``RAGOrchestrator`` driven by the
three protocol fakes — same fakes the orchestrator suite uses, so a wiring
regression here surfaces against the same component contract.
"""

from __future__ import annotations

import httpx
from fastapi import FastAPI

from backend.app.api.deps import get_orchestrator
from backend.app.api.errors import (
    LLMResponseInvalidError,
    OllamaTimeoutError,
    OllamaUnreachableError,
    QdrantUnreachableError,
)
from backend.app.domain.models import Hit
from backend.app.llm.prompts import CHAT_GROUNDED
from backend.app.main import create_app
from backend.app.rag.orchestrator import RAGOrchestrator
from tests.fakes import FakeEmbedder, FakeLLM, FakeVectorStore

COLLECTION = "active_article"
TOP_K = 4
CHAT_NUM_CTX = 4096


def _hit(
    idx: int,
    *,
    section_title: str = "Body",
    text: str = "chunk text",
    score: float | None = None,
) -> Hit:
    return Hit(
        chunk_index=idx,
        section_title=section_title,
        text=text,
        score=score if score is not None else 1.0 - idx * 0.1,
    )


def _build_app_with_orchestrator(
    *,
    embedder: FakeEmbedder | None = None,
    vector_store: FakeVectorStore | None = None,
    llm: FakeLLM | None = None,
) -> tuple[FastAPI, FakeEmbedder, FakeVectorStore, FakeLLM]:
    """Build a fresh app and override ``get_orchestrator`` with a real
    ``RAGOrchestrator`` wired to the three protocol fakes. The lifespan is
    intentionally NOT run — the override covers the only state-reading
    dependency the route exercises.
    """
    app = create_app()

    embedder = embedder or FakeEmbedder()
    vector_store = vector_store or FakeVectorStore()
    llm = llm or FakeLLM(canned_response="a grounded answer")

    orchestrator = RAGOrchestrator(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        collection=COLLECTION,
        top_k=TOP_K,
        chat_prompt_template=CHAT_GROUNDED,
        chat_num_ctx=CHAT_NUM_CTX,
    )
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator

    return app, embedder, vector_store, llm


async def _post_chat(app: FastAPI, question: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://chat.test"
    ) as client:
        return await client.post("/api/chat", json={"question": question})


def _preloaded_store(hits: list[Hit]) -> FakeVectorStore:
    store = FakeVectorStore()
    store.preload_hits(COLLECTION, hits)
    return store


class TestHappyPath:
    async def test_returns_chat_response_with_four_sources(self) -> None:
        hits = [
            _hit(10, section_title="Light-dependent reactions", text="alpha", score=0.9),
            _hit(11, section_title="Pigments", text="beta", score=0.7),
            _hit(12, section_title="History", text="gamma", score=0.5),
            _hit(13, section_title="Reception", text="delta", score=0.3),
        ]
        app, _, _, _ = _build_app_with_orchestrator(
            vector_store=_preloaded_store(hits),
            llm=FakeLLM(canned_response="a grounded answer"),
        )

        response = await _post_chat(app, "What pigment captures light?")

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"answer", "sources"}
        assert body["answer"] == "a grounded answer"
        assert len(body["sources"]) == 4
        for src in body["sources"]:
            assert set(src) == {"section_title", "chunk_index", "score", "excerpt"}

    async def test_sources_preserve_hit_metadata_end_to_end(self) -> None:
        hits = [
            _hit(10, section_title="Light-dependent reactions", text="alpha", score=0.9),
            _hit(11, section_title="Pigments", text="beta", score=0.7),
            _hit(12, section_title="History", text="gamma", score=0.5),
            _hit(13, section_title="Reception", text="delta", score=0.3),
        ]
        app, _, _, _ = _build_app_with_orchestrator(
            vector_store=_preloaded_store(hits)
        )

        response = await _post_chat(app, "question")
        body = response.json()

        for src, hit in zip(body["sources"], hits):
            assert src["chunk_index"] == hit.chunk_index
            assert src["section_title"] == hit.section_title
            assert src["score"] == hit.score
            assert src["excerpt"] == hit.text

    async def test_llm_invoked_exactly_once(self) -> None:
        hits = [_hit(i) for i in range(4)]
        llm = FakeLLM(canned_response="answer")
        app, _, _, _ = _build_app_with_orchestrator(
            vector_store=_preloaded_store(hits), llm=llm
        )

        response = await _post_chat(app, "question")

        assert response.status_code == 200
        assert len(llm.calls) == 1


class TestF9ErrorMapping:
    """One row per F9 chat-side error code per DESIGN §5."""

    async def test_no_active_article_returns_409(self) -> None:
        # Fresh FakeVectorStore — collection_exists is False → A12 path.
        app, _, _, _ = _build_app_with_orchestrator(
            vector_store=FakeVectorStore()
        )

        response = await _post_chat(app, "question")

        assert response.status_code == 409
        assert response.json()["error"] == "no_active_article"

    async def test_retrieval_empty_returns_422(self) -> None:
        # Collection exists (reset called) but has zero points → empty search.
        store = FakeVectorStore()
        store.reset(COLLECTION, vector_size=384)
        app, _, _, _ = _build_app_with_orchestrator(vector_store=store)

        response = await _post_chat(app, "question")

        assert response.status_code == 422
        assert response.json()["error"] == "retrieval_empty"

    async def test_llm_unavailable_returns_503(self) -> None:
        hits = [_hit(i) for i in range(4)]
        llm = FakeLLM(raise_on_generate=OllamaUnreachableError())
        app, _, _, _ = _build_app_with_orchestrator(
            vector_store=_preloaded_store(hits), llm=llm
        )

        response = await _post_chat(app, "question")

        assert response.status_code == 503
        assert response.json()["error"] == "llm_unavailable"

    async def test_llm_timeout_returns_504(self) -> None:
        hits = [_hit(i) for i in range(4)]
        llm = FakeLLM(raise_on_generate=OllamaTimeoutError())
        app, _, _, _ = _build_app_with_orchestrator(
            vector_store=_preloaded_store(hits), llm=llm
        )

        response = await _post_chat(app, "question")

        assert response.status_code == 504
        assert response.json()["error"] == "llm_timeout"

    async def test_llm_response_invalid_returns_502(self) -> None:
        hits = [_hit(i) for i in range(4)]
        llm = FakeLLM(raise_on_generate=LLMResponseInvalidError())
        app, _, _, _ = _build_app_with_orchestrator(
            vector_store=_preloaded_store(hits), llm=llm
        )

        response = await _post_chat(app, "question")

        assert response.status_code == 502
        assert response.json()["error"] == "llm_response_invalid"

    async def test_qdrant_unreachable_returns_503(self) -> None:
        """``collection_exists`` raising QdrantUnreachableError surfaces the
        vector_store_unavailable row even on the gate path.
        """

        class _RaisingStore(FakeVectorStore):
            def collection_exists(self, collection: str) -> bool:
                raise QdrantUnreachableError()

        app, _, _, _ = _build_app_with_orchestrator(vector_store=_RaisingStore())

        response = await _post_chat(app, "question")

        assert response.status_code == 503
        assert response.json()["error"] == "vector_store_unavailable"


class TestSourcesFromRetrieverAtHttpLayer:
    """A17 / U8 — the "not found in the article" answer still carries four
    sources at the HTTP layer. The orchestrator suite proves this for the
    composition; this test proves the wire shape preserves it. If a future
    route refactor ever started building sources from the LLM output (or
    dropping them when the answer is a decline), this test fails.
    """

    async def test_not_found_answer_still_returns_four_sources_on_wire(
        self,
    ) -> None:
        hits = [
            _hit(10, section_title="Light-dependent reactions", text="A"),
            _hit(11, section_title="Pigments", text="B"),
            _hit(12, section_title="History", text="C"),
            _hit(13, section_title="Reception", text="D"),
        ]
        llm = FakeLLM(canned_response="not found in the article")
        app, _, _, _ = _build_app_with_orchestrator(
            vector_store=_preloaded_store(hits), llm=llm
        )

        response = await _post_chat(app, "Who wrote Hamlet?")

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "not found in the article"
        assert len(body["sources"]) == 4
        assert [s["chunk_index"] for s in body["sources"]] == [10, 11, 12, 13]
        assert [s["section_title"] for s in body["sources"]] == [
            "Light-dependent reactions",
            "Pigments",
            "History",
            "Reception",
        ]
