"""FastAPI app entrypoint — DESIGN §1 (FastAPI app) + §4.2 (warmup).

Lifespan responsibilities (in order):
  1. Build the long-lived components (Embedder, QdrantClient + VectorStore,
     OllamaClient) from ``Settings`` and stash them on ``app.state``.
  2. **Await** the warmup ping with ``asyncio.wait_for(..., WARMUP_TIMEOUT_S)``
     — DESIGN §4.2's "no silent ``create_task``" intent. The earlier draft
     was fire-and-forget; this is awaited-with-bound. On timeout or any
     exception, log a WARNING and set ``warmup_ok=False`` — the app still
     serves traffic (graceful degradation per A16).
  3. Yield. Lifecycle teardown is a no-op; the clients are GC-cleaned.

Static mount is best-effort: if ``static/`` is absent (bare-metal dev without
a frontend build), we log an INFO line and skip the mount rather than crash.
``StaticFiles(directory=...)`` raises at construction when the directory is
missing, so the guard wraps the constructor itself.

The single ``DomainError`` -> HTTP mapping is registered here so every route
benefits without per-route exception plumbing (DESIGN §5).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.api.errors import DomainError, domain_error_handler
from backend.app.api.health import router as health_router
from backend.app.api.ingest import router as ingest_router
from backend.app.config import Settings, get_settings
from backend.app.embedding.sbert import SentenceTransformersEmbedder
from backend.app.llm.ollama import OllamaClient
from backend.app.vectorstore.qdrant_store import QdrantVectorStore
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

_STATIC_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    app.state.settings = settings
    app.state.embedder = _build_embedder(settings)
    app.state.qdrant_client = _build_qdrant_client(settings)
    app.state.vector_store = QdrantVectorStore(app.state.qdrant_client)
    app.state.llm_client = _build_llm_client(settings)

    # DESIGN §4.2 — await with bound. The earlier draft used create_task and
    # was effectively useless: the app declared ready before the model was
    # loaded, and the first request paid the cold-start tax anyway.
    app.state.warmup_ok = await _run_warmup(app.state.llm_client, settings)

    yield


async def _run_warmup(llm_client: OllamaClient, settings: Settings) -> bool:
    if not settings.ollama_warmup_enabled:
        logger.info("warmup disabled via OLLAMA_WARMUP_ENABLED; skipping")
        return False
    try:
        return await asyncio.wait_for(
            llm_client.warmup(), timeout=settings.warmup_timeout_s
        )
    except asyncio.TimeoutError:
        logger.warning(
            "warmup ping did not complete within %.1fs; serving cold",
            settings.warmup_timeout_s,
        )
        return False
    except Exception:
        # warmup() is contracted to never raise; if a future refactor breaks
        # that, the lifespan must still come up. Log loud and degrade.
        logger.exception("warmup ping raised; serving cold")
        return False


def _build_embedder(settings: Settings) -> SentenceTransformersEmbedder:
    return SentenceTransformersEmbedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )


def _build_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def _build_llm_client(settings: Settings) -> OllamaClient:
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        timeout_s=settings.ollama_timeout_s,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="silver-spoon2", lifespan=lifespan)

    # Central DESIGN §5 mapping — one registration, every route benefits.
    app.add_exception_handler(DomainError, domain_error_handler)

    app.include_router(health_router)
    app.include_router(ingest_router)

    # Guarded static mount — bare-metal dev without a frontend build must
    # still come up. StaticFiles(directory=...) raises at construction when
    # the path is missing, so the check is on the constructor itself.
    if _STATIC_DIRECTORY.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(_STATIC_DIRECTORY), html=True),
            name="static",
        )
    else:
        logger.info(
            "static directory %s not found; skipping SPA mount", _STATIC_DIRECTORY
        )

    return app


app = create_app()
