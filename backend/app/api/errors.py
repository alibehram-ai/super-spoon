"""DomainError hierarchy and FastAPI exception handler — DESIGN §5.

Each F9 row in DESIGN §5 maps to exactly one DomainError subclass that
declares its ``(code, message, status)`` as class-level attributes. The
central handler dispatches on class identity and returns the §5 JSON shape
``{"error": code, "message": message}`` with the matching HTTP status. Domain
code raises typed exceptions and never builds HTTP responses itself.

``CollectionMissingError`` deliberately is NOT a DomainError — it stays an
internal exception (DESIGN §5: never user-facing). The orchestrator gates on
``collection_exists()`` upstream and raises the user-facing
``NoActiveArticleError`` instead.

``EmbeddingError`` likewise stays out of this hierarchy: it doesn't appear in
F9, and the SBERT path doesn't surface it in practice.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class DomainError(Exception):
    """Base class for every user-facing error in F9 / DESIGN §5.

    Subclasses override ``code``, ``message``, and ``status`` at the class
    level so the central handler can build the response from class identity
    alone. State-bearing subclasses (e.g. ``UrlNamespaceError`` carrying the
    matched namespace prefix) may override ``message`` on the instance; the
    class-level default stays available for no-arg construction (which is
    what ``tests/unit/test_errors.py`` uses to drive the §5 mapping table).
    """

    code: str = "internal_error"
    message: str = "An unexpected error occurred."
    status: int = 500

    def __init__(self, message: str | None = None) -> None:
        if message is not None:
            self.message = message
        super().__init__(self.message)


# --- URL validation (DESIGN §4.9 / §5) ---------------------------------------


class UrlMalformedError(DomainError):
    code = "url_malformed"
    status = 400
    message = "That doesn't look like a valid URL."


class UrlNonWikipediaError(DomainError):
    code = "url_non_wikipedia"
    status = 400
    message = "Please paste a Wikipedia article URL."


class UrlNonEnglishError(DomainError):
    code = "url_non_english"
    status = 400
    message = "Only English Wikipedia (en.wikipedia.org) is supported."


class UrlNamespaceError(DomainError):
    code = "url_namespace"
    status = 400
    message = "That looks like a non-article Wikipedia page, not an article."

    def __init__(self, namespace: str | None = None) -> None:
        if namespace:
            super().__init__(
                f"That looks like a {namespace} page, not an article."
            )
        else:
            super().__init__()
        self.namespace = namespace


class UrlCuridError(DomainError):
    code = "url_curid"
    status = 400
    message = "Use the regular article URL, not a curid link."


# --- Wikipedia fetch (DESIGN §5) --------------------------------------------


class ArticleNotFoundError(DomainError):
    code = "article_not_found"
    status = 404
    message = "Wikipedia doesn't have an article at that URL."


class ArticleDisambiguationError(DomainError):
    code = "article_disambiguation"
    status = 422
    message = (
        "That URL points to a disambiguation page — "
        "pick one of the linked articles."
    )


class WikipediaUnavailableError(DomainError):
    code = "wikipedia_unavailable"
    status = 502
    message = "Wikipedia didn't respond. Try again."


# --- Article cleanup (DESIGN §5) --------------------------------------------


class ArticleTooShortError(DomainError):
    code = "article_too_short"
    status = 422
    message = "This article is too short to be useful (under 500 characters)."

    def __init__(self, char_count: int | None = None) -> None:
        super().__init__()
        self.char_count = char_count


# --- LLM (DESIGN §5) --------------------------------------------------------


class OllamaUnreachableError(DomainError):
    code = "llm_unavailable"
    status = 503
    message = "The local LLM isn't responding. Is Ollama running?"


class OllamaTimeoutError(DomainError):
    code = "llm_timeout"
    status = 504
    message = "The local LLM took too long. Try again or check Ollama load."


class LLMResponseInvalidError(DomainError):
    code = "llm_response_invalid"
    status = 502
    message = (
        "The model returned an unparseable response. "
        "Try rephrasing or check Ollama logs."
    )


# --- Vector store (DESIGN §5) -----------------------------------------------


class QdrantUnreachableError(DomainError):
    code = "vector_store_unavailable"
    status = 503
    message = "The vector database isn't responding."


# --- RAG (DESIGN §5) --------------------------------------------------------


class RetrievalEmptyError(DomainError):
    code = "retrieval_empty"
    status = 422
    message = "No relevant content found in the article for that question."


class NoActiveArticleError(DomainError):
    code = "no_active_article"
    status = 409
    message = "Ingest a Wikipedia article first, then ask questions about it."

    def __init__(self, collection: str | None = None) -> None:
        super().__init__()
        self.collection = collection


# --- FastAPI handler --------------------------------------------------------


async def domain_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Single registration point for the F9 / §5 mapping.

    Returns the §5 JSON shape with the subclass-declared ``status``. The
    handler signature uses ``Exception`` (rather than ``DomainError``) so it
    matches FastAPI's ``exception_handler`` typing; the dispatch decision is
    on subclass identity via attribute lookup, which would fail loudly if a
    non-DomainError ever reached this handler — and the registration in
    ``main.py`` (T10b) only binds it to ``DomainError``, so that can't happen
    in production.
    """

    assert isinstance(exc, DomainError)
    return JSONResponse(
        status_code=exc.status,
        content={"error": exc.code, "message": exc.message},
    )
