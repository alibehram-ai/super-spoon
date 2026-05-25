"""Test fakes that implement the LLMClient / VectorStore / Embedder Protocols.

These are full fakes (DESIGN §7.1), not mocks-with-asserts: callers can drive
them with realistic inputs and observe recorded state. T08 introduces FakeLLM;
T09 adds FakeVectorStore and FakeEmbedder alongside.
"""


class FakeLLM:
    """Records every generate() call and returns a canned response.

    Set `raise_on_generate` to an exception instance to simulate the
    OllamaClient's error paths (OllamaUnreachableError, OllamaTimeoutError,
    LLMResponseInvalidError). The fake never validates output shape itself —
    if a test wants "empty response" semantics, configure `raise_on_generate`
    to LLMResponseInvalidError (mirroring what the real client does upstream).
    """

    def __init__(
        self,
        *,
        canned_response: str = "a four to eight sentence summary",
        raise_on_generate: BaseException | None = None,
    ) -> None:
        self.canned_response = canned_response
        self.raise_on_generate = raise_on_generate
        self.calls: list[dict] = []
        self.warmup_calls: int = 0

    async def generate(
        self,
        prompt: str,
        *,
        num_ctx: int,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "num_ctx": num_ctx,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
            }
        )
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        return self.canned_response

    async def warmup(self) -> bool:
        self.warmup_calls += 1
        return True
