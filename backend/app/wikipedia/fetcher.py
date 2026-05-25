import asyncio
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote

import httpx

from backend.app.api.errors import (
    ArticleDisambiguationError,
    ArticleNotFoundError,
    WikipediaUnavailableError,
)
from backend.app.domain.models import RawSection, WikipediaArticle


_HEADING_TAGS = frozenset({"h2", "h3", "h4", "h5", "h6"})


class WikipediaFetcher:
    """Two-call fetcher per DESIGN §4.10.

    Order: ``/page/summary/{title}`` first (cheap; gates disambiguation, 404,
    redirect-target resolution) then ``/page/html/{title}`` for the sectioned
    body. Single retry with exponential backoff on 5xx / timeout.

    Migrated from the retired ``/page/mobile-sections/`` endpoint mid-Phase-5
    (see NOTES.md "Migrate off /page/mobile-sections/" — Wikimedia retired
    the legacy REST endpoint sooner than its deprecation notice suggested).
    The body call now returns Parsoid HTML; the parser splits sections at
    heading boundaries (``<h2>``..``<h6>``) and collects ``<p>`` content
    within each, preserving the same ``RawSection(title, paragraphs)`` shape
    the rest of the pipeline expects.
    """

    def __init__(
        self,
        *,
        base_url: str,
        user_agent: str,
        timeout_s: float,
        max_retries: int = 1,
        retry_backoff_s: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._user_agent = user_agent
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._transport = transport

    async def fetch(self, article_title: str) -> WikipediaArticle:
        async with httpx.AsyncClient(
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            summary = await self._fetch_summary(client, article_title)

            if summary.get("type") == "disambiguation":
                raise ArticleDisambiguationError()

            canonical_title = (
                summary.get("titles", {}).get("canonical")
                or summary.get("title")
                or article_title
            )

            html = await self._fetch_html(client, canonical_title)

        lede_paragraphs, raw_sections = _parse_parsoid_html(html)
        lede = "\n\n".join(lede_paragraphs)
        sections = [
            RawSection(title=title, paragraphs=paragraphs)
            for title, paragraphs in raw_sections
        ]

        return WikipediaArticle(
            canonical_title=canonical_title,
            lede=lede,
            sections=sections,
            is_disambiguation=False,
        )

    async def _fetch_summary(
        self, client: httpx.AsyncClient, title: str
    ) -> dict[str, Any]:
        url = f"{self._base_url}/page/summary/{quote(title, safe='')}"
        response = await self._get_with_retry(client, url)
        if response.status_code == 404:
            raise ArticleNotFoundError()
        return response.json()

    async def _fetch_html(
        self, client: httpx.AsyncClient, title: str
    ) -> str:
        url = f"{self._base_url}/page/html/{quote(title, safe='')}"
        response = await self._get_with_retry(client, url)
        if response.status_code == 404:
            raise ArticleNotFoundError()
        return response.text

    async def _get_with_retry(
        self, client: httpx.AsyncClient, url: str
    ) -> httpx.Response:
        # Per DESIGN §5, both transient 5xx and read-timeout collapse to the
        # single user-facing `wikipedia_unavailable` case (502).
        headers = {"User-Agent": self._user_agent}
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.get(url, headers=headers)
            except httpx.TimeoutException as exc:
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_backoff_s * (2**attempt))
                    continue
                raise WikipediaUnavailableError() from exc

            if response.status_code >= 500:
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_backoff_s * (2**attempt))
                    continue
                raise WikipediaUnavailableError()

            return response

        # Loop always either returns or raises; this line is for the type checker.
        raise WikipediaUnavailableError()  # pragma: no cover


class _ParsoidHtmlParser(HTMLParser):
    """Single-pass parser over Parsoid ``/page/html`` output.

    Splits sections at any heading tag (``<h2>``..``<h6>``) — matching the
    flattened section list the legacy mobile-sections endpoint returned. ``<p>``
    text is the only body content collected; figures, tables, lists, and
    edit-section spans are dropped (same posture as the legacy parser, which
    also captured only ``<p>`` content from each section's ``text`` field).

    Paragraphs encountered before the first heading are the lede; everything
    after lives under its enclosing heading.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lede_paragraphs: list[str] = []
        self.sections: list[tuple[str, list[str]]] = []
        self._cur_title: str | None = None
        self._cur_paragraphs: list[str] = []
        self._p_depth = 0
        self._p_buf: list[str] = []
        self._h_depth = 0
        self._h_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "p":
            if self._p_depth == 0:
                self._p_buf = []
            self._p_depth += 1
        elif tag in _HEADING_TAGS:
            if self._h_depth == 0:
                self._h_buf = []
            self._h_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._p_depth > 0:
            self._p_depth -= 1
            if self._p_depth == 0:
                text = "".join(self._p_buf).strip()
                if text:
                    if self._cur_title is None:
                        self.lede_paragraphs.append(text)
                    else:
                        self._cur_paragraphs.append(text)
        elif tag in _HEADING_TAGS and self._h_depth > 0:
            self._h_depth -= 1
            if self._h_depth == 0:
                # Finalise the previous section before starting the new one;
                # this also handles consecutive headings (empty paragraph list).
                if self._cur_title is not None:
                    self.sections.append((self._cur_title, self._cur_paragraphs))
                self._cur_title = "".join(self._h_buf).strip()
                self._cur_paragraphs = []

    def handle_data(self, data: str) -> None:
        if self._p_depth > 0:
            self._p_buf.append(data)
        elif self._h_depth > 0:
            self._h_buf.append(data)

    def finalize(self) -> None:
        if self._cur_title is not None:
            self.sections.append((self._cur_title, self._cur_paragraphs))


def _parse_parsoid_html(html: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    parser = _ParsoidHtmlParser()
    parser.feed(html)
    parser.close()
    parser.finalize()
    return parser.lede_paragraphs, parser.sections
