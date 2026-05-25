"""LLMClient Protocol and shared exceptions.

NF4 swappable abstraction. Async Protocol because the production transport
(httpx.AsyncClient) is async; tests substitute a FakeLLM that implements the
same signatures and is awaited by callers.

`num_ctx` is required on `generate` — DESIGN revision-1 cleanup. The chat and
summary call sites wire their own `CHAT_NUM_CTX` / `SUMMARY_NUM_CTX` via
constructor args so per-call-type context windows are explicit and never
silently hardcoded.
"""

from typing import Protocol


class OllamaUnreachableError(Exception):
    """Connection refused, DNS failure, connect-timeout, protocol error, or 5xx
    after retry. T10a maps this to HTTP 503 (`llm_unavailable`).
    """


class OllamaTimeoutError(Exception):
    """Read/write timeout after the connection was established — Ollama is up
    but the request hit the configured `OLLAMA_TIMEOUT_S` budget. Distinct
    from unreachable because the user-facing remedy differs (the user can
    retry; the operator might lower `num_predict` or check Ollama load).
    T10a maps this to HTTP 504 (`llm_timeout`).
    """


class LLMResponseInvalidError(Exception):
    """The HTTP exchange succeeded but the response payload was unusable
    (empty string in `response`, or the `response` key was missing entirely).
    T10a maps this to HTTP 502 (`llm_response_invalid`).
    """


class LLMClient(Protocol):
    async def generate(
        self,
        prompt: str,
        *,
        num_ctx: int,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        """Return the LLM's completion text with leading/trailing whitespace
        stripped. `num_ctx` is REQUIRED — the call site decides chat vs summary.

        Raises:
            OllamaUnreachableError - connection refused / DNS / 5xx after retry.
            OllamaTimeoutError     - read/write timeout after connection.
            LLMResponseInvalidError - empty string or missing `response` field.
        """
        ...

    async def warmup(self) -> bool:
        """One-shot no-op generation to load the model into VRAM. Best-effort
        per A16: returns True on success, False on any failure, NEVER raises.
        The lifespan in T10b relies on this totality.
        """
        ...
