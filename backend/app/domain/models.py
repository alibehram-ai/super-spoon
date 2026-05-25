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
