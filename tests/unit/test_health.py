"""Tests for the FastAPI lifespan, ``/api/health``, and ``Settings`` defaults.

The lifespan is invoked directly via the ``lifespan(app)`` async context
manager rather than through ``httpx``'s ASGITransport — ASGITransport does not
drive the ASGI lifespan protocol, and we want the warmup behaviour exercised
explicitly, not as a side-effect of the test client.

The embedder/qdrant/LLM builders in ``backend.app.main`` are monkeypatched to
return fakes so the lifespan never loads a real SBERT model or opens a real
network socket. The warmup tests drive ``OllamaClient.warmup()``'s contract
through a tiny fake — happy path, timeout, and unexpected exception.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from backend.app import main as main_module
from backend.app.config import Settings, get_settings
from backend.app.main import create_app, lifespan
from tests.fakes import FakeLLM


class _SlowWarmupLLM(FakeLLM):
    """FakeLLM whose ``warmup()`` blocks for ``delay`` seconds. Used to drive
    the lifespan's ``asyncio.wait_for`` past its ``WARMUP_TIMEOUT_S`` bound.
    """

    def __init__(self, *, delay: float) -> None:
        super().__init__()
        self._delay = delay

    async def warmup(self) -> bool:
        self.warmup_calls += 1
        await asyncio.sleep(self._delay)
        return True


class _RaisingWarmupLLM(FakeLLM):
    """FakeLLM whose ``warmup()`` raises. Real ``OllamaClient.warmup()`` is
    contracted never to raise; this drives the belt-and-braces ``except`` in
    the lifespan so a contract break wouldn't crash startup either.
    """

    async def warmup(self) -> bool:
        self.warmup_calls += 1
        raise RuntimeError("simulated warmup bug")


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Each test starts with a clean ``get_settings`` cache so env-var
    overrides via monkeypatch take effect deterministically.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def stubbed_builders(monkeypatch: pytest.MonkeyPatch):
    """Replace the three heavy builders in ``main`` with sentinels so the
    lifespan doesn't load SBERT or hit a real Qdrant. Returns a setter the
    test uses to install a specific LLM fake.
    """
    monkeypatch.setattr(main_module, "_build_embedder", lambda s: object())
    monkeypatch.setattr(main_module, "_build_qdrant_client", lambda s: object())

    def set_llm(llm) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr(main_module, "_build_llm_client", lambda s: llm)

    return set_llm


async def _health(app: FastAPI) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://lifespan.test"
    ) as client:
        return await client.get("/api/health")


class TestWarmupHappyPath:
    async def test_warmup_ok_true_when_fake_returns_true(
        self, stubbed_builders
    ) -> None:
        llm = FakeLLM()
        stubbed_builders(llm)

        app = create_app()
        async with lifespan(app):
            assert app.state.warmup_ok is True
            response = await _health(app)

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "warmup_ok": True}
        # The lifespan really did invoke warmup() — not a default.
        assert llm.warmup_calls == 1


class TestWarmupTimeout:
    async def test_timeout_sets_warmup_ok_false_app_still_serves(
        self, monkeypatch: pytest.MonkeyPatch, stubbed_builders
    ) -> None:
        # Tight timeout via env so the slow-warmup fake trips the bound.
        monkeypatch.setenv("WARMUP_TIMEOUT_S", "0.1")
        get_settings.cache_clear()

        llm = _SlowWarmupLLM(delay=1.0)
        stubbed_builders(llm)

        app = create_app()
        async with lifespan(app):
            # App came up — the WARMUP_TIMEOUT_S race did NOT crash startup.
            assert app.state.warmup_ok is False
            response = await _health(app)

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "warmup_ok": False}


class TestWarmupException:
    async def test_unexpected_exception_sets_warmup_ok_false_app_still_serves(
        self, stubbed_builders
    ) -> None:
        # Real OllamaClient.warmup() never raises (per its contract). But the
        # lifespan has a belt-and-braces except so a future refactor that
        # breaks the contract can't crash startup either.
        stubbed_builders(_RaisingWarmupLLM())

        app = create_app()
        async with lifespan(app):
            assert app.state.warmup_ok is False
            response = await _health(app)

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "warmup_ok": False}


class TestWarmupDisabled:
    async def test_warmup_skipped_when_env_disables_it(
        self, monkeypatch: pytest.MonkeyPatch, stubbed_builders
    ) -> None:
        monkeypatch.setenv("OLLAMA_WARMUP_ENABLED", "false")
        get_settings.cache_clear()

        llm = FakeLLM()
        stubbed_builders(llm)

        app = create_app()
        async with lifespan(app):
            assert app.state.warmup_ok is False
            assert llm.warmup_calls == 0  # warmup was not invoked at all


class TestStaticMountGuard:
    def test_app_constructs_without_static_directory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the static dir lookup to point at a path that doesn't exist.
        from pathlib import Path as _Path

        monkeypatch.setattr(
            main_module, "_STATIC_DIRECTORY", _Path("/tmp/silver-spoon-nope-xyz")
        )

        # create_app() must not raise — the guard is on the constructor, so a
        # missing static/ dir gracefully skips the SPA mount.
        app = create_app()
        route_paths = {getattr(r, "path", None) for r in app.routes}
        assert "/api/health" in route_paths

    async def test_health_route_works_with_no_static_directory(
        self, monkeypatch: pytest.MonkeyPatch, stubbed_builders
    ) -> None:
        from pathlib import Path as _Path

        monkeypatch.setattr(
            main_module, "_STATIC_DIRECTORY", _Path("/tmp/silver-spoon-nope-xyz")
        )
        stubbed_builders(FakeLLM())

        app = create_app()
        async with lifespan(app):
            response = await _health(app)
        assert response.status_code == 200


class TestSettingsDefaults:
    """Spot-check that ``Settings`` defaults still match DESIGN §6.4. A drift
    here would silently break the §6.4 contract that the ``.env.example`` /
    docker-compose plumbing in T15 depends on.
    """

    def test_defaults_match_design_6_4_table(self) -> None:
        # Bypass any local `.env` so the test asserts class defaults — a
        # developer with a bare-metal `.env` (localhost URLs) would otherwise
        # see this test fail through no fault of the code under test.
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        # LLM / warmup
        assert s.llm_model == "llama3.2:3b"
        assert s.ollama_base_url == "http://ollama:11434"
        assert s.ollama_timeout_s == 120
        assert s.warmup_timeout_s == 30
        assert s.summary_num_ctx == 8192
        assert s.chat_num_ctx == 4096
        # Embedding
        assert s.embedding_model == "BAAI/bge-small-en-v1.5"
        assert s.embedding_batch_size == 64
        # Vector store
        assert s.qdrant_url == "http://qdrant:6333"
        assert s.qdrant_collection == "active_article"
        assert s.retrieval_top_k == 4
        assert s.similarity_metric == "cosine"
        # Chunking
        assert s.chunk_section_ceiling_tokens == 1000
        assert s.chunk_sub_size_tokens == 512
        assert s.chunk_sub_overlap_tokens == 64
        # Article
        assert s.article_char_cap == 200_000
        assert s.article_min_chars == 500
        # Summarisation
        assert s.summary_long_threshold_tokens == 3500
        assert s.summary_per_section_tokens == 30

    def test_env_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RETRIEVAL_TOP_K", "6")
        s = Settings()
        assert s.retrieval_top_k == 6
