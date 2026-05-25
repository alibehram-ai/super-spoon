"""Unit tests for OllamaClient.

Uses httpx.MockTransport to drive responses and exceptions — no real Ollama,
no network. Every test asserts the request body shape (`stream=False`,
options carried) alongside the response or exception assertions, since the
non-streaming contract is just as load-bearing as the error mapping.
"""

import json
from collections.abc import Callable

import httpx
import pytest

from backend.app.llm.base import (
    LLMResponseInvalidError,
    OllamaTimeoutError,
    OllamaUnreachableError,
)
from backend.app.llm.ollama import OllamaClient

BASE_URL = "http://ollama:11434"
MODEL = "llama3.2:3b"


def _build_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[OllamaClient, list[httpx.Request]]:
    recorded: list[httpx.Request] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return handler(request)

    transport = httpx.MockTransport(recording_handler)
    client = OllamaClient(
        base_url=BASE_URL,
        model=MODEL,
        timeout_s=5.0,
        transport=transport,
    )
    return client, recorded


def _request_body(request: httpx.Request) -> dict:
    return json.loads(request.content.decode())


class TestGenerateHappyPath:
    async def test_returns_stripped_response_text(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "  hello world  ", "done": True})

        client, _ = _build_client(handler)

        result = await client.generate("any prompt", num_ctx=4096)

        assert result == "hello world"

    async def test_request_body_includes_stream_false(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "x"})

        client, recorded = _build_client(handler)

        await client.generate("any prompt", num_ctx=4096)

        assert _request_body(recorded[0])["stream"] is False

    async def test_request_body_includes_model_and_prompt(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "x"})

        client, recorded = _build_client(handler)

        await client.generate("specific prompt text", num_ctx=4096)

        body = _request_body(recorded[0])
        assert body["model"] == MODEL
        assert body["prompt"] == "specific prompt text"

    async def test_request_body_carries_options_with_num_ctx_and_num_predict(
        self,
    ) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "x"})

        client, recorded = _build_client(handler)

        await client.generate(
            "prompt",
            num_ctx=4096,
            max_tokens=256,
            temperature=0.5,
        )

        options = _request_body(recorded[0])["options"]
        assert options["num_ctx"] == 4096
        # Ollama's API uses num_predict for max-tokens.
        assert options["num_predict"] == 256
        assert options["temperature"] == 0.5

    async def test_stop_sequences_omitted_when_none(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "x"})

        client, recorded = _build_client(handler)

        await client.generate("prompt", num_ctx=4096)

        options = _request_body(recorded[0])["options"]
        assert "stop" not in options

    async def test_stop_sequences_passed_through_when_provided(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "x"})

        client, recorded = _build_client(handler)

        await client.generate(
            "prompt", num_ctx=4096, stop=["\n\n", "END"]
        )

        options = _request_body(recorded[0])["options"]
        assert options["stop"] == ["\n\n", "END"]

    async def test_posts_to_api_generate(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "x"})

        client, recorded = _build_client(handler)

        await client.generate("prompt", num_ctx=4096)

        assert str(recorded[0].url) == f"{BASE_URL}/api/generate"
        assert recorded[0].method == "POST"


class TestInvalidResponses:
    async def test_empty_response_string_raises_invalid(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": ""})

        client, _ = _build_client(handler)

        with pytest.raises(LLMResponseInvalidError):
            await client.generate("prompt", num_ctx=4096)

    async def test_whitespace_only_response_raises_invalid(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "   \n  "})

        client, _ = _build_client(handler)

        with pytest.raises(LLMResponseInvalidError):
            await client.generate("prompt", num_ctx=4096)

    async def test_missing_response_key_raises_invalid(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"done": True})

        client, _ = _build_client(handler)

        with pytest.raises(LLMResponseInvalidError):
            await client.generate("prompt", num_ctx=4096)


class TestTransportErrorMapping:
    async def test_connect_error_maps_to_unreachable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("DNS failure")

        client, _ = _build_client(handler)

        with pytest.raises(OllamaUnreachableError):
            await client.generate("prompt", num_ctx=4096)

    async def test_connect_timeout_maps_to_unreachable_not_timeout(self) -> None:
        # ConnectTimeout inherits TimeoutException but represents a failed
        # connect — per DESIGN §5, this is Unreachable, NOT a read-timeout.
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("connect timed out")

        client, _ = _build_client(handler)

        with pytest.raises(OllamaUnreachableError):
            await client.generate("prompt", num_ctx=4096)

    async def test_remote_protocol_error_maps_to_unreachable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.RemoteProtocolError("server hung up")

        client, _ = _build_client(handler)

        with pytest.raises(OllamaUnreachableError):
            await client.generate("prompt", num_ctx=4096)

    async def test_read_timeout_maps_to_timeout(self) -> None:
        # ReadTimeout = exchange after connection established exceeded budget.
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("read timed out")

        client, _ = _build_client(handler)

        with pytest.raises(OllamaTimeoutError):
            await client.generate("prompt", num_ctx=4096)

    async def test_non_200_response_maps_to_unreachable(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "internal"})

        client, _ = _build_client(handler)

        with pytest.raises(OllamaUnreachableError):
            await client.generate("prompt", num_ctx=4096)


class TestNumCtxIsRequired:
    async def test_num_ctx_is_keyword_only_and_required(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "x"})

        client, _ = _build_client(handler)

        with pytest.raises(TypeError):
            # No num_ctx supplied; the signature must reject this at call time
            # rather than silently defaulting (DESIGN revision-1 cleanup).
            await client.generate("prompt")  # type: ignore[call-arg]


class TestWarmup:
    async def test_warmup_returns_true_on_success(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "pong"})

        client, recorded = _build_client(handler)

        result = await client.warmup()

        assert result is True
        # Warmup uses num_predict=1 and a small num_ctx — bounded to keep the
        # call cheap regardless of how the lifespan times it.
        options = _request_body(recorded[0])["options"]
        assert options["num_predict"] == 1

    async def test_warmup_returns_false_on_unreachable_never_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("nope")

        client, _ = _build_client(handler)

        result = await client.warmup()

        assert result is False

    async def test_warmup_returns_false_on_read_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow")

        client, _ = _build_client(handler)

        result = await client.warmup()

        assert result is False

    async def test_warmup_returns_false_on_invalid_response(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": ""})

        client, _ = _build_client(handler)

        result = await client.warmup()

        assert result is False

    async def test_warmup_returns_false_on_non_200(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": "model loading"})

        client, _ = _build_client(handler)

        result = await client.warmup()

        assert result is False

    async def test_warmup_never_raises_on_unexpected_exception(self) -> None:
        # If a non-httpx exception ever bubbled (e.g. a bug in our payload
        # building), warmup must still degrade gracefully — the lifespan
        # depends on it.
        def handler(request: httpx.Request) -> httpx.Response:
            raise RuntimeError("unexpected")

        client, _ = _build_client(handler)

        result = await client.warmup()

        assert result is False


class TestPromptTemplates:
    """The prompts module is part of T07. Sanity-check the placeholders so
    a future edit can't silently break the f-string contract that T08
    (Summariser) and T09 (RAGOrchestrator) depend on.
    """

    def test_summary_short_has_body_placeholder(self) -> None:
        from backend.app.llm.prompts import SUMMARY_SHORT

        rendered = SUMMARY_SHORT.format(body="some article body")
        assert "some article body" in rendered

    def test_summary_long_has_outline_placeholder(self) -> None:
        from backend.app.llm.prompts import SUMMARY_LONG

        rendered = SUMMARY_LONG.format(outline="some structured outline")
        assert "some structured outline" in rendered

    def test_chat_grounded_has_chunks_and_question_placeholders(self) -> None:
        from backend.app.llm.prompts import CHAT_GROUNDED

        rendered = CHAT_GROUNDED.format(
            chunks="[1] heading\nbody text", question="What is X?"
        )
        assert "[1] heading" in rendered
        assert "body text" in rendered
        assert "What is X?" in rendered

    def test_chat_grounded_instructs_decline_on_unknown(self) -> None:
        # F7 grounding contract: model must reply exactly with the magic
        # phrase when chunks don't contain the answer. The integration test
        # (T16) asserts this end-to-end; here we just confirm the instruction
        # is present in the template the LLM sees.
        from backend.app.llm.prompts import CHAT_GROUNDED

        assert "not found in the article" in CHAT_GROUNDED
