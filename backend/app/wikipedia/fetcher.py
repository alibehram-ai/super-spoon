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


class WikipediaFetcher:
    """Two-call fetcher per DESIGN §4.10.

    Order: /page/summary/{title} first (cheap; gates disambiguation, 404,
    redirect-target resolution) then /page/mobile-sections/{title} for the
    sectioned body. Single retry with exponential backoff on 5xx / timeout.

    NOTE: /page/mobile-sections/ is on Wikimedia's long-term deprecation
    track; the Parsoid migration is logged in NOTES.md.
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

            mobile = await self._fetch_mobile_sections(client, canonical_title)

        lead_sections = mobile.get("lead", {}).get("sections", []) or []
        lede_html = lead_sections[0].get("text", "") if lead_sections else ""
        lede_paragraphs = _html_to_paragraphs(lede_html)
        lede = "\n\n".join(lede_paragraphs)

        sections: list[RawSection] = []
        for sec in mobile.get("remaining", {}).get("sections", []) or []:
            title = sec.get("line", "")
            paragraphs = _html_to_paragraphs(sec.get("text", ""))
            sections.append(RawSection(title=title, paragraphs=paragraphs))

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

    async def _fetch_mobile_sections(
        self, client: httpx.AsyncClient, title: str
    ) -> dict[str, Any]:
        url = f"{self._base_url}/page/mobile-sections/{quote(title, safe='')}"
        response = await self._get_with_retry(client, url)
        if response.status_code == 404:
            raise ArticleNotFoundError()
        return response.json()

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


class _ParagraphExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._buf: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "p":
            if self._depth == 0:
                self._buf = []
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._depth > 0:
            self._depth -= 1
            if self._depth == 0:
                text = "".join(self._buf).strip()
                if text:
                    self.paragraphs.append(text)

    def handle_data(self, data: str) -> None:
        if self._depth > 0:
            self._buf.append(data)


def _html_to_paragraphs(html: str) -> list[str]:
    if not html.strip():
        return []
    parser = _ParagraphExtractor()
    parser.feed(html)
    parser.close()
    return parser.paragraphs
