"""Ollama HTTP client wrapping /api/generate (non-streaming per OQ-7).

Maps httpx transport errors to the two distinct user-facing failure modes:
  - connect-phase failures (ConnectError, ConnectTimeout, ProtocolError,
    DNS, anything else under TransportError) -> OllamaUnreachableError;
  - read/write timeouts AFTER the connection was established -> OllamaTimeoutError.
The order of the except clauses is load-bearing: ConnectTimeout is a
TimeoutException in httpx but should map to Unreachable, so we catch it
before the generic TimeoutException clause. Non-200 responses also become
Unreachable.

`max_tokens` is sent as Ollama's `num_predict`. `stream=False` is always
included in the request body — non-streaming is part of the contract.
"""

from typing import Any

import httpx

from backend.app.llm.base import (
    LLMResponseInvalidError,
    OllamaTimeoutError,
    OllamaUnreachableError,
)

_GENERATE_PATH = "/api/generate"
_WARMUP_PROMPT = "ping"
_WARMUP_NUM_CTX = 512


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._transport = transport

    async def generate(
        self,
        prompt: str,
        *,
        num_ctx: int,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        options: dict[str, Any] = {
            "num_ctx": num_ctx,
            "num_predict": max_tokens,
            "temperature": temperature,
        }
        if stop is not None:
            options["stop"] = stop

        body = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        async with httpx.AsyncClient(
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            try:
                response = await client.post(
                    f"{self._base_url}{_GENERATE_PATH}", json=body
                )
            # Order is load-bearing: ConnectTimeout inherits TimeoutException
            # but maps to Unreachable, so it must be caught first.
            except httpx.ConnectTimeout as exc:
                raise OllamaUnreachableError("ollama connect timeout") from exc
            except (
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                httpx.NetworkError,
                httpx.ProtocolError,
            ) as exc:
                raise OllamaUnreachableError("ollama unreachable") from exc
            except httpx.TimeoutException as exc:
                # ReadTimeout, WriteTimeout, PoolTimeout — connection established,
                # but the exchange exceeded OLLAMA_TIMEOUT_S.
                raise OllamaTimeoutError("ollama read/write timeout") from exc
            except httpx.TransportError as exc:
                raise OllamaUnreachableError("ollama transport error") from exc

        if response.status_code != 200:
            raise OllamaUnreachableError(
                f"ollama returned status {response.status_code}"
            )

        payload = response.json()
        if "response" not in payload:
            raise LLMResponseInvalidError(
                "ollama response missing 'response' field"
            )
        text = str(payload["response"]).strip()
        if not text:
            raise LLMResponseInvalidError("ollama returned empty response")
        return text

    async def warmup(self) -> bool:
        """Total: returns False on every failure path, never raises.

        Issues a minimal generate to load the model into VRAM. The lifespan
        in T10b awaits this with a WARMUP_TIMEOUT_S bound and sets
        app.state.warmup_ok = result.
        """
        try:
            await self.generate(
                _WARMUP_PROMPT, max_tokens=1, num_ctx=_WARMUP_NUM_CTX
            )
        except BaseException:
            return False
        return True
