import re

from backend.app.api.errors import ArticleTooShortError
from backend.app.domain.models import (
    CleanedArticle,
    Section,
    WikipediaArticle,
)

__all__ = ["ArticleCleaner", "ArticleTooShortError"]


_DEFAULT_DROP = frozenset(
    {"References", "External links", "Further reading", "See also", "Notes"}
)
_CITATION_PATTERN = re.compile(r"\[\d+\]")
_SECTION_SEPARATOR = "\n\n"


class ArticleCleaner:
    """Drop low-signal sections, strip citation markers, enforce min/max body size.

    Pure function over its input (DESIGN §3). Contract:
      - Drop sections whose normalised title is in dropped_section_titles
        (case-insensitive, whitespace-trimmed). Reassigns section_index so the
        output is contiguous from 0.
      - Strip inline citation markers ([1], [12], ...) from lede + section
        paragraphs. Non-numeric bracketed text ([sic], [citation needed]) is
        preserved by design.
      - total_chars is the length of the linearised body
        (lede + "\n\n" + section text + "\n\n" + ...) on the retained content.
      - total_chars < min_chars  -> ArticleTooShortError.
      - total_chars > char_cap   -> drop trailing sections until the running
        total fits; never partially truncate a section (A9/U13). truncated=True.
    """

    def __init__(
        self,
        *,
        dropped_section_titles: frozenset[str] = _DEFAULT_DROP,
        min_chars: int = 500,
        char_cap: int = 200_000,
    ) -> None:
        self._dropped = frozenset(
            t.strip().lower() for t in dropped_section_titles
        )
        self._min_chars = min_chars
        self._char_cap = char_cap

    def clean(self, article: WikipediaArticle) -> CleanedArticle:
        clean_lede = _strip_citations(article.lede)

        body_length = len(clean_lede)
        any_part_emitted = bool(clean_lede)

        kept_sections: list[Section] = []
        new_index = 0

        for raw in article.sections:
            if raw.title.strip().lower() in self._dropped:
                continue

            cleaned_paragraphs = [_strip_citations(p) for p in raw.paragraphs]
            section_text = _SECTION_SEPARATOR.join(cleaned_paragraphs)

            if section_text:
                if any_part_emitted:
                    body_length += len(_SECTION_SEPARATOR)
                start = body_length
                body_length += len(section_text)
                end = body_length
                any_part_emitted = True
            else:
                start = body_length
                end = body_length

            kept_sections.append(
                Section(
                    section_index=new_index,
                    title=raw.title,
                    paragraphs=cleaned_paragraphs,
                    char_offset_start=start,
                    char_offset_end=end,
                )
            )
            new_index += 1

        total_chars = body_length

        if total_chars < self._min_chars:
            raise ArticleTooShortError(total_chars)

        truncated = False
        if total_chars > self._char_cap:
            while kept_sections and kept_sections[-1].char_offset_end > self._char_cap:
                kept_sections.pop()
            truncated = True
            total_chars = (
                kept_sections[-1].char_offset_end
                if kept_sections
                else len(clean_lede)
            )

        return CleanedArticle(
            title=article.canonical_title,
            lede=clean_lede,
            sections=kept_sections,
            total_chars=total_chars,
            truncated=truncated,
        )


def _strip_citations(text: str) -> str:
    return _CITATION_PATTERN.sub("", text)
