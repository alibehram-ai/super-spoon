from dataclasses import dataclass


@dataclass(frozen=True)
class ValidatedUrl:
    canonical_url: str
    article_title: str
