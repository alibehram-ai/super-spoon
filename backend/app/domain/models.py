from dataclasses import dataclass


@dataclass(frozen=True)
class ValidatedUrl:
    canonical_url: str
    article_title: str


@dataclass(frozen=True)
class RawSection:
    title: str
    paragraphs: list[str]


@dataclass(frozen=True)
class WikipediaArticle:
    canonical_title: str
    lede: str
    sections: list[RawSection]
    is_disambiguation: bool


@dataclass(frozen=True)
class Section:
    section_index: int
    title: str
    paragraphs: list[str]
    char_offset_start: int
    char_offset_end: int


@dataclass(frozen=True)
class CleanedArticle:
    title: str
    lede: str
    sections: list[Section]
    total_chars: int
    truncated: bool


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    section_index: int
    chunk_index_in_section: int
    section_title: str
    text: str
    char_offset_start: int
    char_offset_end: int
