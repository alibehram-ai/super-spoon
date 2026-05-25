"""Unit tests for ``POST /api/ingest`` — T11.

The route's job is composition, not arithmetic — every component already has
its own unit test, so this suite verifies wiring, response shape, and the
F9 -> §5 ingest-side error mapping.

All seven ``Depends`` providers are substituted via ``app.dependency_overrides``
per the T11 spec — no real SBERT load, no real Qdrant client, no real Ollama
HTTP. The fetcher is a fake; the cleaner and chunker run the real code with
a deterministic fake tokenizer; the embedder, vector store, and the LLM behind
the summariser are full Protocol fakes from ``tests/fakes.py``.
"""

from __future__ import annotations

import httpx
from fastapi import FastAPI

from backend.app.api.deps import (
    get_article_cleaner,
    get_chunker,
    get_embedder,
    get_summariser,
    get_vector_store,
    get_wikipedia_fetcher,
)
from backend.app.api.errors import (
    ArticleDisambiguationError,
    ArticleNotFoundError,
    LLMResponseInvalidError,
    OllamaTimeoutError,
    OllamaUnreachableError,
    QdrantUnreachableError,
    WikipediaUnavailableError,
)
from backend.app.chunking.chunker import Chunker
from backend.app.domain.models import RawSection, WikipediaArticle
from backend.app.llm.prompts import SUMMARY_LONG, SUMMARY_SHORT
from backend.app.main import create_app
from backend.app.rag.summariser import Summariser
from backend.app.wikipedia.cleaner import ArticleCleaner
from tests.fakes import FakeEmbedder, FakeLLM, FakeVectorStore

VALID_URL = "https://en.wikipedia.org/wiki/Photosynthesis"


def _fake_tokenizer(text: str) -> list[int]:
    """One token per whitespace-separated word — same shape T04/T08 use so
    chunker and summariser behaviour is deterministic without SBERT.
    """
    return list(range(len(text.split())))


def _make_article(
    *,
    title: str = "Photosynthesis",
    lede: str | None = None,
    sections: list[RawSection] | None = None,
) -> WikipediaArticle:
    """Synthetic WikipediaArticle large enough to clear ``min_chars=500`` and
    produce multiple chunks under the fake tokenizer.
    """
    if lede is None:
        lede = "Photosynthesis is the process plants use to convert light. " * 6
    if sections is None:
        sections = [
            RawSection(
                title="Overview",
                paragraphs=[
                    "Photosynthesis takes place in the chloroplasts of green plants. " * 4,
                    "Light energy is captured by chlorophyll molecules in the thylakoid. " * 4,
                ],
            ),
            RawSection(
                title="Pigments",
                paragraphs=[
                    "Chlorophyll is the dominant pigment of photosynthesis. " * 4,
                ],
            ),
        ]
    return WikipediaArticle(
        canonical_title=title,
        lede=lede,
        sections=sections,
        is_disambiguation=False,
    )


class FakeFetcher:
    """Records the requested title; returns a canned article or raises a
    pre-configured exception.
    """

    def __init__(
        self,
        article: WikipediaArticle | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self.article = article
        self.raise_exc = raise_exc
        self.calls: list[str] = []

    async def fetch(self, article_title: str) -> WikipediaArticle:
        self.calls.append(article_title)
        if self.raise_exc is not None:
            raise self.raise_exc
        assert self.article is not None
        return self.article


class _RaisingVectorStore(FakeVectorStore):
    """VectorStore whose ``reset`` raises — used to drive the
    ``vector_store_unavailable`` mapping row.
    """

    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self._exc = exc

    def reset(self, collection: str, vector_size: int, metric: str = "cosine") -> None:
        raise self._exc


class _Fakes:
    def __init__(
        self,
        *,
        fetcher: FakeFetcher,
        cleaner: ArticleCleaner,
        chunker: Chunker,
        embedder: FakeEmbedder,
        vector_store: FakeVectorStore,
        llm: FakeLLM,
        summariser: Summariser,
    ) -> None:
        self.fetcher = fetcher
        self.cleaner = cleaner
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.summariser = summariser


def _build_app_with_fakes(
    *,
    fetcher: FakeFetcher | None = None,
    cleaner: ArticleCleaner | None = None,
    chunker: Chunker | None = None,
    embedder: FakeEmbedder | None = None,
    vector_store: FakeVectorStore | None = None,
    llm: FakeLLM | None = None,
    article: WikipediaArticle | None = None,
) -> tuple[FastAPI, _Fakes]:
    """Build a fresh app with every Depends overridden by a fake.

    The lifespan is intentionally NOT run — overrides cover every dependency,
    so app.state never needs the real embedder/qdrant client/LLM that the
    production lifespan builds.
    """
    app = create_app()

    fetcher = fetcher or FakeFetcher(article=article or _make_article())
    cleaner = cleaner or ArticleCleaner(min_chars=500, char_cap=200_000)
    chunker = chunker or Chunker(
        section_ceiling_tokens=1000,
        sub_size_tokens=512,
        sub_overlap_tokens=64,
        tokenizer=_fake_tokenizer,
    )
    embedder = embedder or FakeEmbedder()
    vector_store = vector_store or FakeVectorStore()
    llm = llm or FakeLLM(canned_response="a short canned summary")
    summariser = Summariser(
        llm,
        num_ctx=8192,
        long_threshold_tokens=3500,
        per_section_tokens=30,
        tokenizer=_fake_tokenizer,
        prompt_template_short=SUMMARY_SHORT,
        prompt_template_long=SUMMARY_LONG,
    )

    app.dependency_overrides[get_wikipedia_fetcher] = lambda: fetcher
    app.dependency_overrides[get_article_cleaner] = lambda: cleaner
    app.dependency_overrides[get_chunker] = lambda: chunker
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_summariser] = lambda: summariser

    return app, _Fakes(
        fetcher=fetcher,
        cleaner=cleaner,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        summariser=summariser,
    )


async def _post_ingest(app: FastAPI, url: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://ingest.test"
    ) as client:
        return await client.post("/api/ingest", json={"url": url})


class TestHappyPath:
    async def test_returns_expected_response_shape(self) -> None:
        app, _ = _build_app_with_fakes()

        response = await _post_ingest(app, VALID_URL)

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {
            "summary",
            "section_count",
            "chunk_count",
            "truncated",
            "char_count",
        }
        assert body["summary"] == "a short canned summary"
        assert body["truncated"] is False
        assert body["chunk_count"] > 0
        assert body["section_count"] >= 1
        assert body["char_count"] > 0

    async def test_pipeline_steps_invoked_in_order(self) -> None:
        app, fakes = _build_app_with_fakes()

        response = await _post_ingest(app, VALID_URL)
        assert response.status_code == 200

        # Validator passed the article title to the fetcher.
        assert fakes.fetcher.calls == ["Photosynthesis"]
        # Vector store was reset+upserted against the active collection.
        assert len(fakes.vector_store.reset_calls) == 1
        assert fakes.vector_store.reset_calls[0]["collection"] == "active_article"
        assert (
            fakes.vector_store.reset_calls[0]["vector_size"]
            == fakes.embedder.vector_size
        )
        assert fakes.vector_store.reset_calls[0]["metric"] == "cosine"
        assert len(fakes.vector_store.upsert_calls) == 1
        assert fakes.vector_store.upsert_calls[0]["collection"] == "active_article"
        # Summariser fed the LLM exactly once.
        assert len(fakes.llm.calls) == 1

    async def test_embedder_called_with_chunk_texts(self) -> None:
        app, fakes = _build_app_with_fakes()

        response = await _post_ingest(app, VALID_URL)
        body = response.json()

        # embed_texts called exactly once with N == chunk_count texts.
        assert len(fakes.embedder.calls) == 1
        assert len(fakes.embedder.calls[0]) == body["chunk_count"]
        # upsert received the same number of chunks.
        assert (
            fakes.vector_store.upsert_calls[0]["n_chunks"] == body["chunk_count"]
        )


class TestTruncatedArticle:
    async def test_truncated_flag_propagates_to_response(self) -> None:
        # Force truncation: char_cap below the synthetic article body length
        # so the cleaner drops trailing sections and sets truncated=True.
        cleaner = ArticleCleaner(min_chars=100, char_cap=600)
        article = _make_article(
            lede="The lede goes here. " * 20,
            sections=[
                RawSection(title="One", paragraphs=["Paragraph one. " * 30]),
                RawSection(title="Two", paragraphs=["Paragraph two. " * 30]),
                RawSection(title="Three", paragraphs=["Paragraph three. " * 30]),
            ],
        )
        app, _ = _build_app_with_fakes(cleaner=cleaner, article=article)

        response = await _post_ingest(app, VALID_URL)

        assert response.status_code == 200
        body = response.json()
        assert body["truncated"] is True
        assert body["char_count"] <= 600


class TestF9ErrorMapping:
    """One row per F9 ingest-side error code. Each fake is configured to raise
    the exception that DESIGN §5 says owns the row; the response is then
    asserted against the §5 (status, code) pair.
    """

    async def test_url_malformed_returns_400(self) -> None:
        app, _ = _build_app_with_fakes()
        response = await _post_ingest(app, "not-a-url")
        assert response.status_code == 400
        assert response.json()["error"] == "url_malformed"

    async def test_url_non_wikipedia_returns_400(self) -> None:
        app, _ = _build_app_with_fakes()
        response = await _post_ingest(
            app, "https://example.com/wiki/Photosynthesis"
        )
        assert response.status_code == 400
        assert response.json()["error"] == "url_non_wikipedia"

    async def test_url_non_english_returns_400(self) -> None:
        app, _ = _build_app_with_fakes()
        response = await _post_ingest(
            app, "https://fr.wikipedia.org/wiki/Photosynthese"
        )
        assert response.status_code == 400
        assert response.json()["error"] == "url_non_english"

    async def test_url_namespace_returns_400(self) -> None:
        app, _ = _build_app_with_fakes()
        response = await _post_ingest(
            app, "https://en.wikipedia.org/wiki/Special:Random"
        )
        assert response.status_code == 400
        assert response.json()["error"] == "url_namespace"

    async def test_url_curid_returns_400(self) -> None:
        app, _ = _build_app_with_fakes()
        response = await _post_ingest(
            app, "https://en.wikipedia.org/?curid=12345"
        )
        assert response.status_code == 400
        assert response.json()["error"] == "url_curid"

    async def test_article_not_found_returns_404(self) -> None:
        fetcher = FakeFetcher(raise_exc=ArticleNotFoundError())
        app, _ = _build_app_with_fakes(fetcher=fetcher)
        response = await _post_ingest(app, VALID_URL)
        assert response.status_code == 404
        assert response.json()["error"] == "article_not_found"

    async def test_disambiguation_returns_422(self) -> None:
        fetcher = FakeFetcher(raise_exc=ArticleDisambiguationError())
        app, _ = _build_app_with_fakes(fetcher=fetcher)
        response = await _post_ingest(app, VALID_URL)
        assert response.status_code == 422
        assert response.json()["error"] == "article_disambiguation"

    async def test_wikipedia_unavailable_returns_502(self) -> None:
        fetcher = FakeFetcher(raise_exc=WikipediaUnavailableError())
        app, _ = _build_app_with_fakes(fetcher=fetcher)
        response = await _post_ingest(app, VALID_URL)
        assert response.status_code == 502
        assert response.json()["error"] == "wikipedia_unavailable"

    async def test_article_too_short_returns_422(self) -> None:
        # Real cleaner with min_chars high enough that the synthetic article
        # fails the floor.
        cleaner = ArticleCleaner(min_chars=10**6, char_cap=200_000)
        app, _ = _build_app_with_fakes(cleaner=cleaner)
        response = await _post_ingest(app, VALID_URL)
        assert response.status_code == 422
        assert response.json()["error"] == "article_too_short"

    async def test_llm_unavailable_returns_503(self) -> None:
        llm = FakeLLM(raise_on_generate=OllamaUnreachableError())
        app, _ = _build_app_with_fakes(llm=llm)
        response = await _post_ingest(app, VALID_URL)
        assert response.status_code == 503
        assert response.json()["error"] == "llm_unavailable"

    async def test_llm_timeout_returns_504(self) -> None:
        llm = FakeLLM(raise_on_generate=OllamaTimeoutError())
        app, _ = _build_app_with_fakes(llm=llm)
        response = await _post_ingest(app, VALID_URL)
        assert response.status_code == 504
        assert response.json()["error"] == "llm_timeout"

    async def test_llm_response_invalid_returns_502(self) -> None:
        llm = FakeLLM(raise_on_generate=LLMResponseInvalidError())
        app, _ = _build_app_with_fakes(llm=llm)
        response = await _post_ingest(app, VALID_URL)
        assert response.status_code == 502
        assert response.json()["error"] == "llm_response_invalid"

    async def test_qdrant_unreachable_returns_503(self) -> None:
        store = _RaisingVectorStore(QdrantUnreachableError())
        app, _ = _build_app_with_fakes(vector_store=store)
        response = await _post_ingest(app, VALID_URL)
        assert response.status_code == 503
        assert response.json()["error"] == "vector_store_unavailable"


class TestPipelineShortCircuits:
    """A failure upstream of the summariser must NOT result in an LLM call —
    the "fail fast and cheap" invariant DESIGN §2's flow implies.
    """

    async def test_disambiguation_short_circuits_before_llm(self) -> None:
        fetcher = FakeFetcher(raise_exc=ArticleDisambiguationError())
        llm = FakeLLM()
        app, fakes = _build_app_with_fakes(fetcher=fetcher, llm=llm)

        response = await _post_ingest(app, VALID_URL)

        assert response.status_code == 422
        assert fakes.llm.calls == []
        assert fakes.embedder.calls == []
        assert fakes.vector_store.reset_calls == []
        assert fakes.vector_store.upsert_calls == []

    async def test_qdrant_failure_short_circuits_before_summariser(self) -> None:
        store = _RaisingVectorStore(QdrantUnreachableError())
        llm = FakeLLM()
        app, fakes = _build_app_with_fakes(vector_store=store, llm=llm)

        response = await _post_ingest(app, VALID_URL)

        assert response.status_code == 503
        # Embedder runs before the vector store, so its calls is non-empty —
        # but the summariser sits after the upsert and must not have fired.
        assert fakes.llm.calls == []
