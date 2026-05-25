"""Tests for the DomainError → (status, code, message) mapping in DESIGN §5.

The mapping table IS the test: one parametrized row per F9 case. The fixture
spins up a throwaway FastAPI app with a single ``POST /raise/{exception_name}``
route that resolves the named class out of ``backend.app.api.errors`` and
raises it with no constructor args. The central ``domain_error_handler`` then
turns that into the §5 JSON shape, and each row asserts the exact triple.

If you add a row to DESIGN §5, add a row here. If you change a default message
or status in ``api/errors.py``, this test is what fails first.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from backend.app.api import errors as errors_module
from backend.app.api.errors import DomainError, domain_error_handler


@pytest.fixture(scope="session")
def error_app() -> FastAPI:
    """Throwaway app: one route that raises any DomainError by class name.

    Session-scoped because the app is immutable across the parametrized
    suite — building one per row would be wasteful. The handler under test
    is the single registration the production app (T10b) will also use.
    """
    app = FastAPI()
    app.add_exception_handler(DomainError, domain_error_handler)

    @app.post("/raise/{exception_name}")
    async def raise_route(exception_name: str) -> None:
        cls = getattr(errors_module, exception_name)
        raise cls()

    return app


@pytest.fixture(scope="session")
async def client(error_app: FastAPI):
    transport = httpx.ASGITransport(app=error_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://errors.test"
    ) as ac:
        yield ac


# DESIGN §5 — one row per F9 case. Tuple order: (class_name, status, code, message).
# CollectionMissingError is INTENTIONALLY ABSENT — DESIGN §5 keeps it as an
# internal exception that never reaches users.
DESIGN_5_TABLE: list[tuple[str, int, str, str]] = [
    # URL validation (§4.9 / §5)
    (
        "UrlMalformedError",
        400,
        "url_malformed",
        "That doesn't look like a valid URL.",
    ),
    (
        "UrlNonWikipediaError",
        400,
        "url_non_wikipedia",
        "Please paste a Wikipedia article URL.",
    ),
    (
        "UrlNonEnglishError",
        400,
        "url_non_english",
        "Only English Wikipedia (en.wikipedia.org) is supported.",
    ),
    (
        "UrlNamespaceError",
        400,
        "url_namespace",
        # No-arg construction → class-level default (the {namespace} placeholder
        # is filled in by the validator when it knows which prefix matched).
        "That looks like a non-article Wikipedia page, not an article.",
    ),
    (
        "UrlCuridError",
        400,
        "url_curid",
        "Use the regular article URL, not a curid link.",
    ),
    # Wikipedia fetch (§5)
    (
        "ArticleNotFoundError",
        404,
        "article_not_found",
        "Wikipedia doesn't have an article at that URL.",
    ),
    (
        "ArticleDisambiguationError",
        422,
        "article_disambiguation",
        "That URL points to a disambiguation page — pick one of the linked articles.",
    ),
    (
        "WikipediaUnavailableError",
        502,
        "wikipedia_unavailable",
        "Wikipedia didn't respond. Try again.",
    ),
    # Article cleanup (§5)
    (
        "ArticleTooShortError",
        422,
        "article_too_short",
        "This article is too short to be useful (under 500 characters).",
    ),
    # LLM (§5)
    (
        "OllamaUnreachableError",
        503,
        "llm_unavailable",
        "The local LLM isn't responding. Is Ollama running?",
    ),
    (
        "OllamaTimeoutError",
        504,
        "llm_timeout",
        "The local LLM took too long. Try again or check Ollama load.",
    ),
    (
        "LLMResponseInvalidError",
        502,
        "llm_response_invalid",
        "The model returned an unparseable response. Try rephrasing or check Ollama logs.",
    ),
    # Vector store (§5)
    (
        "QdrantUnreachableError",
        503,
        "vector_store_unavailable",
        "The vector database isn't responding.",
    ),
    # RAG (§5)
    (
        "RetrievalEmptyError",
        422,
        "retrieval_empty",
        "No relevant content found in the article for that question.",
    ),
    (
        "NoActiveArticleError",
        409,
        "no_active_article",
        "Ingest a Wikipedia article first, then ask questions about it.",
    ),
]


@pytest.mark.parametrize(
    ("exception_name", "expected_status", "expected_code", "expected_message"),
    DESIGN_5_TABLE,
    ids=[row[0] for row in DESIGN_5_TABLE],
)
async def test_design_5_mapping_row(
    client: httpx.AsyncClient,
    exception_name: str,
    expected_status: int,
    expected_code: str,
    expected_message: str,
) -> None:
    response = await client.post(f"/raise/{exception_name}")

    assert response.status_code == expected_status
    assert response.json() == {
        "error": expected_code,
        "message": expected_message,
    }


class TestHandlerInvariants:
    """Properties of the handler that hold across every row above."""

    async def test_response_is_json(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/raise/UrlMalformedError")
        assert response.headers["content-type"].startswith("application/json")

    async def test_payload_has_exactly_two_keys(
        self, client: httpx.AsyncClient
    ) -> None:
        # §5 commits to the shape {"error": <code>, "message": <user-facing>}.
        # No extra debug fields, no stack traces.
        response = await client.post("/raise/OllamaUnreachableError")
        assert set(response.json().keys()) == {"error", "message"}

    async def test_all_handled_classes_subclass_domain_error(self) -> None:
        # Catches a future row added to errors.py that forgets to inherit
        # from DomainError — the central handler is keyed on DomainError, so
        # a non-subclass would bypass the §5 mapping entirely.
        for class_name, *_ in DESIGN_5_TABLE:
            cls = getattr(errors_module, class_name)
            assert issubclass(cls, DomainError), (
                f"{class_name} must subclass DomainError to be routed by the "
                f"central handler."
            )

    async def test_collection_missing_error_is_not_a_domain_error(self) -> None:
        # DESIGN §5: CollectionMissingError stays an internal exception.
        # If someone re-parents it, the chat route's A12 path would start
        # leaking the internal "collection missing" message to users.
        from backend.app.vectorstore.base import CollectionMissingError

        assert not issubclass(CollectionMissingError, DomainError)
