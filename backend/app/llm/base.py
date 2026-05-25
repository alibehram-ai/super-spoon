"""LLMClient Protocol.

NF4 swappable abstraction. Async Protocol because the production transport
(httpx.AsyncClient) is async; tests substitute a FakeLLM that implements the
same signatures and is awaited by callers.

`num_ctx` is required on `generate` — DESIGN revision-1 cleanup. The chat and
summary call sites wire their own `CHAT_NUM_CTX` / `SUMMARY_NUM_CTX` via
constructor args so per-call-type context windows are explicit and never
silently hardcoded.

The three LLM error classes (``OllamaUnreachableError``, ``OllamaTimeoutError``,
``LLMResponseInvalidError``) live in ``backend.app.api.errors`` as DomainError
subclasses (T10a) and are re-exported here so existing callers can keep their
imports.
"""

from typing import Protocol

from backend.app.api.errors import (
    LLMResponseInvalidError,
    OllamaTimeoutError,
    OllamaUnreachableError,
)

__all__ = [
    "LLMClient",
    "LLMResponseInvalidError",
    "OllamaTimeoutError",
    "OllamaUnreachableError",
]


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
