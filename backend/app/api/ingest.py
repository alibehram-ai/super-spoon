"""``POST /api/ingest`` — DESIGN §2 ingest flow.

Composes the pipeline:

  validate_url -> fetcher.fetch -> cleaner.clean -> chunker.chunk
    -> embedder.embed_texts (to_thread) -> vector_store.reset+upsert (to_thread)
    -> summariser.summarise

Every component is injected via FastAPI ``Depends(...)`` so tests can override
exactly the pieces they need with fakes (the wiring T11 verifies, not the
component internals — those already have their own unit tests).

CPU / sync I/O pieces (the embedder and the vector store) run through
``anyio.to_thread.run_sync`` per DESIGN §6.1 so they never block the event
loop. Wrapping the vector store is "for safety" — single-article scale makes
the calls fast in practice, but the route shouldn't have to know that.

Errors flow through the central ``DomainError`` handler registered in
``main.py`` (T10a) — every F9 ingest-side row in DESIGN §5 lands on the right
status/code without per-route exception plumbing.
"""

from __future__ import annotations

import anyio
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.app.api.deps import (
    get_app_settings,
    get_article_cleaner,
    get_chunker,
    get_embedder,
    get_summariser,
    get_vector_store,
    get_wikipedia_fetcher,
)
from backend.app.chunking.chunker import Chunker
from backend.app.config import Settings
from backend.app.embedding.base import Embedder
from backend.app.rag.summariser import Summariser
from backend.app.vectorstore.base import VectorStore
from backend.app.wikipedia.cleaner import ArticleCleaner
from backend.app.wikipedia.fetcher import WikipediaFetcher
from backend.app.wikipedia.validator import validate_url

router = APIRouter()


class IngestRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    summary: str
    section_count: int
    chunk_count: int
    truncated: bool
    char_count: int


@router.post("/api/ingest", response_model=IngestResponse)
async def ingest(
    payload: IngestRequest,
    settings: Settings = Depends(get_app_settings),
    fetcher: WikipediaFetcher = Depends(get_wikipedia_fetcher),
    cleaner: ArticleCleaner = Depends(get_article_cleaner),
    chunker: Chunker = Depends(get_chunker),
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
    summariser: Summariser = Depends(get_summariser),
) -> IngestResponse:
    validated = validate_url(payload.url)
    article = await fetcher.fetch(validated.article_title)
    cleaned = cleaner.clean(article)
    chunks = chunker.chunk(cleaned)

    chunk_texts = [c.text for c in chunks]
    vectors = await anyio.to_thread.run_sync(embedder.embed_texts, chunk_texts)

    collection = settings.qdrant_collection
    await anyio.to_thread.run_sync(
        vector_store.reset,
        collection,
        embedder.vector_size,
        settings.similarity_metric,
    )
    await anyio.to_thread.run_sync(
        vector_store.upsert, collection, chunks, vectors
    )

    summary = await summariser.summarise(cleaned)

    return IngestResponse(
        summary=summary.strip(),
        section_count=len(cleaned.sections),
        chunk_count=len(chunks),
        truncated=cleaned.truncated,
        char_count=cleaned.total_chars,
    )
