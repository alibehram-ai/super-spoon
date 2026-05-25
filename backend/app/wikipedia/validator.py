import re
from urllib.parse import quote, unquote, urlparse

from backend.app.api.errors import (
    UrlCuridError,
    UrlMalformedError,
    UrlNamespaceError,
    UrlNonEnglishError,
    UrlNonWikipediaError,
)
from backend.app.domain.models import ValidatedUrl

_HOST_ACCEPT = frozenset({"en.wikipedia.org", "en.m.wikipedia.org"})
_HOST_LANG_PATTERN = re.compile(r"^[a-z]{2,3}(\.m)?\.wikipedia\.org$")
_NAMESPACE_PREFIXES = (
    "Special:",
    "Talk:",
    "Category:",
    "File:",
    "Help:",
    "User:",
    "Portal:",
)
_PATH_PATTERN = re.compile(r"^/wiki/(?P<title>.+)$")
_CANONICAL_HOST = "en.wikipedia.org"


def validate_url(raw_url: str) -> ValidatedUrl:
    """Validate and canonicalise a Wikipedia article URL.

    Pipeline order matches DESIGN §4.9 / §3 step-for-step; each rejection
    raises the §5-mapped DomainError subclass directly so the central handler
    in ``backend.app.api.errors`` can dispatch on class identity (T10a):

      1. parse + scheme check                       -> UrlMalformedError
      2. host accept vs lang.wikipedia.org pattern  -> UrlNonEnglishError
      3. host wikipedia gate                        -> UrlNonWikipediaError
      4. query-string curid gate                    -> UrlCuridError
      5. path shape /wiki/<title>                   -> UrlMalformedError
      6. namespace prefix gate                      -> UrlNamespaceError
      7. title non-empty after URL-decode           -> UrlMalformedError
      8. normalise host (en.m. -> en.), strip fragment, re-encode title.
    """
    try:
        parsed = urlparse(raw_url)
    except ValueError as exc:
        raise UrlMalformedError() from exc

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise UrlMalformedError()

    host = (parsed.hostname or "").lower()

    if host not in _HOST_ACCEPT:
        if _HOST_LANG_PATTERN.match(host):
            raise UrlNonEnglishError()
        raise UrlNonWikipediaError()

    if "curid" in parsed.query.lower():
        raise UrlCuridError()

    path_match = _PATH_PATTERN.match(parsed.path)
    if not path_match:
        raise UrlMalformedError()

    raw_title = path_match.group("title")
    article_title = unquote(raw_title)

    matched_namespace = next(
        (p for p in _NAMESPACE_PREFIXES if article_title.startswith(p)),
        None,
    )
    if matched_namespace is not None:
        raise UrlNamespaceError(namespace=matched_namespace.rstrip(":"))

    if not article_title.strip():
        raise UrlMalformedError()

    canonical_url = f"https://{_CANONICAL_HOST}/wiki/{quote(article_title, safe='')}"
    return ValidatedUrl(canonical_url=canonical_url, article_title=article_title)
