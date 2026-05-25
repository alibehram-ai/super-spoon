import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from backend.app.api.errors import (
    ArticleDisambiguationError,
    ArticleNotFoundError,
    WikipediaUnavailableError,
)
from backend.app.domain.models import WikipediaArticle
from backend.app.wikipedia.fetcher import WikipediaFetcher

BASE_URL = "https://test.example/api/rest_v1"
USER_AGENT = "silver-spoon-test/0.0 (https://example.invalid; contact: test@example.invalid)"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _load(name: str) -> dict[str, Any]:
    with (FIXTURE_DIR / name).open() as f:
        return json.load(f)


def _build_fetcher(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    max_retries: int = 1,
) -> tuple[WikipediaFetcher, list[httpx.Request]]:
    recorded: list[httpx.Request] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return handler(request)

    transport = httpx.MockTransport(recording_handler)
    fetcher = WikipediaFetcher(
        base_url=BASE_URL,
        user_agent=USER_AGENT,
        timeout_s=5.0,
        max_retries=max_retries,
        retry_backoff_s=0.0,  # keep tests fast
        transport=transport,
    )
    return fetcher, recorded


class TestErrorPaths:
    async def test_404_from_summary_raises_not_found(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/page/summary/" in str(request.url)
            return httpx.Response(404, json={"detail": "Page not found"})

        fetcher, recorded = _build_fetcher(handler)

        with pytest.raises(ArticleNotFoundError):
            await fetcher.fetch("Missing")
        # mobile-sections must not be called when summary 404s.
        assert all("/page/summary/" in str(r.url) for r in recorded)

    async def test_disambiguation_short_circuits(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/page/summary/" in url:
                return httpx.Response(
                    200,
                    json={
                        "type": "disambiguation",
                        "title": "Mercury",
                        "titles": {
                            "canonical": "Mercury",
                            "normalized": "Mercury",
                            "display": "Mercury",
                        },
                    },
                )
            pytest.fail(f"mobile-sections must not be called on disambiguation; got {url}")

        fetcher, recorded = _build_fetcher(handler)

        with pytest.raises(ArticleDisambiguationError):
            await fetcher.fetch("Mercury")
        assert len(recorded) == 1
        assert "/page/summary/Mercury" in str(recorded[0].url)

    async def test_transient_after_double_503(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"detail": "Service unavailable"})

        fetcher, recorded = _build_fetcher(handler)

        with pytest.raises(WikipediaUnavailableError):
            await fetcher.fetch("Anything")
        # initial attempt + one retry = 2 summary calls; mobile-sections never reached.
        assert len(recorded) == 2
        assert all("/page/summary/" in str(r.url) for r in recorded)

    async def test_404_from_mobile_sections_also_raises_not_found(self) -> None:
        # Defensive branch: summary succeeded but the body endpoint 404s. Not
        # enumerated in DESIGN §5, but treating it as not_found is the only
        # behaviour that doesn't leak a partially-formed article downstream.
        summary_body = _load("sample_summary.json")

        def handler(request: httpx.Request) -> httpx.Response:
            if "/page/summary/" in str(request.url):
                return httpx.Response(200, json=summary_body)
            return httpx.Response(404, json={"detail": "missing"})

        fetcher, _ = _build_fetcher(handler)

        with pytest.raises(ArticleNotFoundError):
            await fetcher.fetch("Sample_Article")

    async def test_timeout_raises_timeout_reason(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("simulated read timeout")

        fetcher, recorded = _build_fetcher(handler)

        with pytest.raises(WikipediaUnavailableError):
            await fetcher.fetch("Anything")
        # Timeout also retries once before raising — same single-retry policy as 5xx.
        # Per DESIGN §5, transient 5xx and read-timeout collapse to the same
        # user-facing case (wikipedia_unavailable), hence the same exception.
        assert len(recorded) == 2


class TestRetryRecovery:
    async def test_503_then_200_recovers(self) -> None:
        summary_calls = {"n": 0}
        summary_body = _load("sample_summary.json")
        mobile_body = _load("sample_mobile_sections.json")

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/page/summary/" in url:
                summary_calls["n"] += 1
                if summary_calls["n"] == 1:
                    return httpx.Response(503, json={"detail": "transient"})
                return httpx.Response(200, json=summary_body)
            if "/page/mobile-sections/" in url:
                return httpx.Response(200, json=mobile_body)
            raise AssertionError(f"unexpected URL {url}")

        fetcher, recorded = _build_fetcher(handler)
        result = await fetcher.fetch("Sample_Article")

        assert isinstance(result, WikipediaArticle)
        # 2 summary calls (one 503 + one 200) + 1 mobile-sections call.
        assert summary_calls["n"] == 2
        assert len(recorded) == 3
        assert sum("/page/mobile-sections/" in str(r.url) for r in recorded) == 1


class TestHeaders:
    async def test_user_agent_on_every_recorded_request(self) -> None:
        summary_body = _load("sample_summary.json")
        mobile_body = _load("sample_mobile_sections.json")

        def handler(request: httpx.Request) -> httpx.Response:
            if "/page/summary/" in str(request.url):
                return httpx.Response(200, json=summary_body)
            return httpx.Response(200, json=mobile_body)

        fetcher, recorded = _build_fetcher(handler)
        await fetcher.fetch("Sample_Article")

        assert len(recorded) == 2
        for request in recorded:
            assert request.headers.get("user-agent") == USER_AGENT


class TestRedirectCanonicalReflection:
    async def test_canonical_title_from_summary_used_for_mobile_sections(self) -> None:
        summary_body = _load("sample_summary.json")
        # Force a redirect: requested "Sample Article" → canonical "Sample_Article".
        # (sample_summary fixture already encodes that distinction.)
        mobile_body = _load("sample_mobile_sections.json")

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "/page/summary/" in url:
                return httpx.Response(200, json=summary_body)
            if "/page/mobile-sections/" in url:
                return httpx.Response(200, json=mobile_body)
            raise AssertionError(f"unexpected URL {url}")

        fetcher, recorded = _build_fetcher(handler)
        # Request via the spaced-form (the validator's article_title shape).
        result = await fetcher.fetch("Sample Article")

        assert result.canonical_title == "Sample_Article"

        summary_calls = [r for r in recorded if "/page/summary/" in str(r.url)]
        mobile_calls = [r for r in recorded if "/page/mobile-sections/" in str(r.url)]
        assert len(summary_calls) == 1
        assert len(mobile_calls) == 1

        # Summary was called with the requested (URL-encoded) title.
        assert "/page/summary/Sample%20Article" in str(summary_calls[0].url)
        # Mobile-sections was called with the *canonical* title from summary.
        assert "/page/mobile-sections/Sample_Article" in str(mobile_calls[0].url)


class TestHappyPathAgainstFixtures:
    async def test_returns_structured_wikipedia_article(self) -> None:
        summary_body = _load("sample_summary.json")
        mobile_body = _load("sample_mobile_sections.json")

        def handler(request: httpx.Request) -> httpx.Response:
            if "/page/summary/" in str(request.url):
                return httpx.Response(200, json=summary_body)
            return httpx.Response(200, json=mobile_body)

        fetcher, _ = _build_fetcher(handler)
        result = await fetcher.fetch("Sample_Article")

        assert result.canonical_title == "Sample_Article"
        assert result.is_disambiguation is False
        # Lede: two <p>s joined with blank line; citation [1] retained as text
        # (it gets stripped by the cleaner downstream, not by the fetcher).
        assert "Sample Article is a tiny stub" in result.lede
        assert "[1]" in result.lede
        assert "MockTransport assertions easy" in result.lede

        section_titles = [s.title for s in result.sections]
        assert section_titles == ["Overview", "Details", "References"]

        overview = result.sections[0]
        assert overview.paragraphs == [
            "The overview opens with a topical sentence.",
            "It continues with a second paragraph that survives extraction.",
        ]

        # References section has no <p> tags in the fixture (only <ol>);
        # paragraph extraction yields an empty list. The cleaner is what
        # drops the section by title — the fetcher must still include it.
        references = result.sections[-1]
        assert references.title == "References"
        assert references.paragraphs == []

    async def test_html_paragraph_extraction_strips_outer_whitespace(self) -> None:
        summary_body = _load("sample_summary.json")
        mobile_body = {
            "lead": {"sections": [{"id": 0, "text": "  <p>  hello world  </p>  "}]},
            "remaining": {"sections": []},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            if "/page/summary/" in str(request.url):
                return httpx.Response(200, json=summary_body)
            return httpx.Response(200, json=mobile_body)

        fetcher, _ = _build_fetcher(handler)
        result = await fetcher.fetch("Sample_Article")
        assert result.lede == "hello world"

    async def test_empty_lead_sections_yields_empty_lede(self) -> None:
        summary_body = _load("sample_summary.json")
        mobile_body = {"lead": {"sections": []}, "remaining": {"sections": []}}

        def handler(request: httpx.Request) -> httpx.Response:
            if "/page/summary/" in str(request.url):
                return httpx.Response(200, json=summary_body)
            return httpx.Response(200, json=mobile_body)

        fetcher, _ = _build_fetcher(handler)
        result = await fetcher.fetch("Sample_Article")
        assert result.lede == ""
        assert result.sections == []
