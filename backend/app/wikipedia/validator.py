import re
from urllib.parse import quote, unquote, urlparse

from backend.app.domain.models import ValidatedUrl


class UrlValidationError(Exception):
    """Raised by validate_url with a stable reason code.

    T10 will re-parent this to the DomainError hierarchy; the reason codes
    here are the dispatch keys for that re-parenting.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


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

    Pipeline order matches DESIGN §4.9 / §3 step-for-step:
      1. parse + scheme check                       -> 'malformed'
      2. host accept vs lang.wikipedia.org pattern  -> 'non_english'
      3. host wikipedia gate                        -> 'non_wikipedia'
      4. query-string curid gate                    -> 'curid'
      5. path shape /wiki/<title>                   -> 'malformed'
      6. namespace prefix gate                      -> 'namespace'
      7. title non-empty after URL-decode           -> 'malformed'
      8. normalise host (en.m. -> en.), strip fragment, re-encode title.
    """
    try:
        parsed = urlparse(raw_url)
    except ValueError as exc:
        raise UrlValidationError("malformed") from exc

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise UrlValidationError("malformed")

    host = (parsed.hostname or "").lower()

    if host not in _HOST_ACCEPT:
        if _HOST_LANG_PATTERN.match(host):
            raise UrlValidationError("non_english")
        raise UrlValidationError("non_wikipedia")

    if "curid" in parsed.query.lower():
        raise UrlValidationError("curid")

    path_match = _PATH_PATTERN.match(parsed.path)
    if not path_match:
        raise UrlValidationError("malformed")

    raw_title = path_match.group("title")
    article_title = unquote(raw_title)

    if article_title.startswith(_NAMESPACE_PREFIXES):
        raise UrlValidationError("namespace")

    if not article_title.strip():
        raise UrlValidationError("malformed")

    canonical_url = f"https://{_CANONICAL_HOST}/wiki/{quote(article_title, safe='')}"
    return ValidatedUrl(canonical_url=canonical_url, article_title=article_title)
