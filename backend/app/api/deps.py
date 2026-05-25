"""FastAPI dependency providers — DESIGN §1, §4 wiring.

Each ``get_*`` function below is the single construction site for one
component. Process-scoped (``@lru_cache``) where the component is stateless
and expensive to build (Embedder loads SBERT; Chunker loads the SBERT
tokenizer; QdrantVectorStore wraps a connection-pooled client). Per-request
where the orchestrator composes other process-scoped pieces — its construction
is cheap but giving it FastAPI ``Depends`` wiring is what lets tests override
its inputs via the standard ``app.dependency_overrides`` pathway.

The orchestrator's LLM client and embedder come from ``app.state`` rather
than being constructed fresh per request. The lifespan in ``main.py`` builds
them once at startup; the providers below read them back. This is the same
pattern the warmup uses — single instance, single lifecycle.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Request
from qdrant_client import QdrantClient

from backend.app.chunking.chunker import Chunker
from backend.app.chunking.tokenizer import get_sbert_tokenizer
from backend.app.config import Settings, get_settings
from backend.app.embedding.base import Embedder
from backend.app.embedding.sbert import SentenceTransformersEmbedder
from backend.app.llm.base import LLMClient
from backend.app.llm.ollama import OllamaClient
from backend.app.llm.prompts import CHAT_GROUNDED, SUMMARY_LONG, SUMMARY_SHORT
from backend.app.rag.orchestrator import RAGOrchestrator
from backend.app.rag.summariser import Summariser
from backend.app.vectorstore.base import VectorStore
from backend.app.vectorstore.qdrant_store import QdrantVectorStore
from backend.app.wikipedia.cleaner import ArticleCleaner
from backend.app.wikipedia.fetcher import WikipediaFetcher


# --- Process-scoped factories (constructed once, reused across requests) ---


@lru_cache(maxsize=1)
def _build_embedder(settings: Settings) -> SentenceTransformersEmbedder:
    return SentenceTransformersEmbedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )


@lru_cache(maxsize=1)
def _build_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


@lru_cache(maxsize=1)
def _build_llm_client(settings: Settings) -> OllamaClient:
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        timeout_s=settings.ollama_timeout_s,
    )


@lru_cache(maxsize=1)
def _build_tokenizer(model_name: str):  # type: ignore[no-untyped-def]
    return get_sbert_tokenizer(model_name)


# --- FastAPI providers ------------------------------------------------------
#
# The Embedder and LLMClient providers read app.state — set by the lifespan
# at startup. Reading from state (rather than constructing in the provider)
# means tests can override these with fakes via the standard
# ``app.dependency_overrides`` pathway without monkeypatching the lifespan.


def get_app_settings(request: Request) -> Settings:
    """Process-singleton Settings — read once at startup; stable for the
    lifetime of the app. Per-request access stays cheap.
    """
    return get_settings()


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_wikipedia_fetcher(
    settings: Settings = Depends(get_app_settings),
) -> WikipediaFetcher:
    return WikipediaFetcher(
        base_url=settings.wikipedia_api_base,
        user_agent=settings.wikipedia_user_agent,
        timeout_s=settings.wikipedia_fetch_timeout_s,
    )


def get_article_cleaner(
    settings: Settings = Depends(get_app_settings),
) -> ArticleCleaner:
    return ArticleCleaner(
        min_chars=settings.article_min_chars,
        char_cap=settings.article_char_cap,
    )


def get_chunker(
    settings: Settings = Depends(get_app_settings),
) -> Chunker:
    tokenizer = _build_tokenizer(settings.embedding_model)
    return Chunker(
        section_ceiling_tokens=settings.chunk_section_ceiling_tokens,
        sub_size_tokens=settings.chunk_sub_size_tokens,
        sub_overlap_tokens=settings.chunk_sub_overlap_tokens,
        tokenizer=tokenizer,
    )


def get_summariser(
    settings: Settings = Depends(get_app_settings),
    llm: LLMClient = Depends(get_llm_client),
) -> Summariser:
    tokenizer = _build_tokenizer(settings.embedding_model)
    return Summariser(
        llm,
        num_ctx=settings.summary_num_ctx,
        long_threshold_tokens=settings.summary_long_threshold_tokens,
        per_section_tokens=settings.summary_per_section_tokens,
        tokenizer=tokenizer,
        prompt_template_short=SUMMARY_SHORT,
        prompt_template_long=SUMMARY_LONG,
    )


def get_orchestrator(
    settings: Settings = Depends(get_app_settings),
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
    llm: LLMClient = Depends(get_llm_client),
) -> RAGOrchestrator:
    return RAGOrchestrator(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        collection=settings.qdrant_collection,
        top_k=settings.retrieval_top_k,
        chat_prompt_template=CHAT_GROUNDED,
        chat_num_ctx=settings.chat_num_ctx,
    )
