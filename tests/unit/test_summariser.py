"""Unit tests for Summariser.

Fake tokenizer (one-token-per-whitespace-word, deterministic, no model load)
mirrors the chunker's tests so we exercise the same numeric contracts. FakeLLM
records every call so we can assert single-call semantics and the exact prompt
shape per A4 / §4.7.
"""

import pytest

from backend.app.domain.models import CleanedArticle, Section
from backend.app.llm.base import (
    LLMResponseInvalidError,
    OllamaTimeoutError,
    OllamaUnreachableError,
)
from backend.app.llm.prompts import SUMMARY_LONG, SUMMARY_SHORT
from backend.app.rag.summariser import Summariser
from tests.fakes import FakeLLM


def fake_tokenizer(s: str) -> list[int]:
    return list(range(len(s.split())))


def _make_section(
    *,
    title: str,
    paragraphs: list[str],
    section_index: int = 0,
    char_offset_start: int = 0,
) -> Section:
    body_len = sum(len(p) for p in paragraphs) + max(0, (len(paragraphs) - 1) * 2)
    return Section(
        section_index=section_index,
        title=title,
        paragraphs=paragraphs,
        char_offset_start=char_offset_start,
        char_offset_end=char_offset_start + body_len,
    )


def _make_article(*, lede: str, sections: list[Section]) -> CleanedArticle:
    return CleanedArticle(
        title="Test Article",
        lede=lede,
        sections=sections,
        total_chars=0,  # not validated by Summariser
        truncated=False,
    )


def _build_summariser(
    llm: FakeLLM,
    *,
    long_threshold_tokens: int = 3500,
    per_section_tokens: int = 30,
    num_ctx: int = 8192,
) -> Summariser:
    return Summariser(
        llm,
        num_ctx=num_ctx,
        long_threshold_tokens=long_threshold_tokens,
        per_section_tokens=per_section_tokens,
        tokenizer=fake_tokenizer,
        prompt_template_short=SUMMARY_SHORT,
        prompt_template_long=SUMMARY_LONG,
    )


class TestShortPath:
    async def test_below_threshold_calls_llm_exactly_once_with_short_template(
        self,
    ) -> None:
        llm = FakeLLM(canned_response="four to eight sentences here.")
        # 5 tokens total in body — under threshold=100.
        article = _make_article(
            lede="alpha beta gamma",
            sections=[
                _make_section(title="Body", paragraphs=["delta epsilon"])
            ],
        )
        summariser = _build_summariser(llm, long_threshold_tokens=100)

        result = await summariser.summarise(article)

        assert result == "four to eight sentences here."
        assert len(llm.calls) == 1

    async def test_short_path_prompt_contains_full_body_verbatim(self) -> None:
        llm = FakeLLM()
        article = _make_article(
            lede="lede words here",
            sections=[
                _make_section(title="History", paragraphs=["history para text"]),
                _make_section(title="Biology", paragraphs=["biology para text"]),
            ],
        )
        summariser = _build_summariser(llm, long_threshold_tokens=100)

        await summariser.summarise(article)

        prompt = llm.calls[0]["prompt"]
        assert "lede words here" in prompt
        assert "history para text" in prompt
        assert "biology para text" in prompt

    async def test_short_path_does_not_emit_outline_headings(self) -> None:
        # Sanity: short-path uses the body verbatim, NOT the outline format.
        # Make sure we didn't accidentally route through _build_outline.
        llm = FakeLLM()
        article = _make_article(
            lede="lede words here",
            sections=[
                _make_section(title="History", paragraphs=["history text"])
            ],
        )
        summariser = _build_summariser(llm, long_threshold_tokens=100)

        await summariser.summarise(article)

        # The long-path template emits "## <heading>"; short path does not.
        prompt = llm.calls[0]["prompt"]
        assert "## History" not in prompt

    async def test_token_count_counts_lede_once_not_twice(self) -> None:
        # Guard against a regression where the linearised body might
        # accidentally double-count the lede (e.g. by treating it as both the
        # lede string AND a synthetic section). The token threshold should
        # reflect actual content, not a doubled fixture.
        llm = FakeLLM()
        article = _make_article(
            lede="one two three four five",  # 5 tokens
            sections=[
                _make_section(title="Body", paragraphs=["six seven eight"])  # 3 tokens
            ],
        )
        # Threshold sits between 8 (single count) and 13 (doubled count).
        # If the lede were counted twice, we'd land on the long path; with
        # correct single-counting we land on the short path.
        summariser = _build_summariser(llm, long_threshold_tokens=10)

        await summariser.summarise(article)

        prompt = llm.calls[0]["prompt"]
        # Short-path template — not the outline format.
        assert "## Body" not in prompt


class TestLongPath:
    async def test_above_threshold_calls_llm_exactly_once_with_long_template(
        self,
    ) -> None:
        llm = FakeLLM()
        long_section = _make_section(
            title="Long",
            paragraphs=[" ".join(f"w{i}" for i in range(200))],
        )
        article = _make_article(lede="short lede", sections=[long_section])
        summariser = _build_summariser(llm, long_threshold_tokens=50)

        await summariser.summarise(article)

        assert len(llm.calls) == 1

    async def test_long_path_outline_contains_lede_verbatim(self) -> None:
        llm = FakeLLM()
        lede = "lede sentence one. lede sentence two."
        article = _make_article(
            lede=lede,
            sections=[
                _make_section(
                    title="History",
                    paragraphs=[" ".join(f"h{i}" for i in range(100))],
                ),
            ],
        )
        summariser = _build_summariser(llm, long_threshold_tokens=10)

        await summariser.summarise(article)

        prompt = llm.calls[0]["prompt"]
        assert lede in prompt

    async def test_long_path_outline_contains_every_section_heading(self) -> None:
        llm = FakeLLM()
        article = _make_article(
            lede="lede",
            sections=[
                _make_section(
                    title="Light-dependent reactions",
                    paragraphs=[" ".join(f"a{i}" for i in range(60))],
                ),
                _make_section(
                    title="Light-independent reactions",
                    paragraphs=[" ".join(f"b{i}" for i in range(60))],
                ),
                _make_section(
                    title="Pigments",
                    paragraphs=[" ".join(f"c{i}" for i in range(60))],
                ),
            ],
        )
        summariser = _build_summariser(llm, long_threshold_tokens=10)

        await summariser.summarise(article)

        prompt = llm.calls[0]["prompt"]
        assert "## Light-dependent reactions" in prompt
        assert "## Light-independent reactions" in prompt
        assert "## Pigments" in prompt

    async def test_long_path_caps_first_sentence_at_per_section_tokens(
        self,
    ) -> None:
        # 50-word first sentence (no punctuation) -> cap at 30 tokens.
        first_para = " ".join(f"w{i}" for i in range(50))
        llm = FakeLLM()
        article = _make_article(
            lede="lede",
            sections=[
                _make_section(title="Long", paragraphs=[first_para]),
                # Force long-path routing by making the body large.
                _make_section(
                    title="Filler",
                    paragraphs=[" ".join(f"f{i}" for i in range(200))],
                ),
            ],
        )
        per_section_tokens = 30
        summariser = _build_summariser(
            llm,
            long_threshold_tokens=10,
            per_section_tokens=per_section_tokens,
        )

        await summariser.summarise(article)

        prompt = llm.calls[0]["prompt"]
        # The snippet under "## Long" is the first per_section_tokens words.
        expected_snippet = " ".join(f"w{i}" for i in range(per_section_tokens))
        long_heading_idx = prompt.index("## Long")
        # The snippet appears immediately after the heading (separated by \n).
        assert expected_snippet in prompt[long_heading_idx:]
        # Word #30 (which would exceed the budget) MUST NOT appear under the
        # snippet — but it does appear in the filler section, so we slice the
        # prompt to just the Long section's window.
        filler_heading_idx = prompt.index("## Filler")
        long_section_window = prompt[long_heading_idx:filler_heading_idx]
        assert "w30" not in long_section_window

    async def test_long_path_first_sentence_is_first_sentence_not_first_paragraph(
        self,
    ) -> None:
        # First paragraph has two sentences; only the first should be sampled
        # (then capped). Under the cap, the cap doesn't bite.
        first_para = "Photosynthesis converts light energy into chemical energy. " \
                     "The process occurs in chloroplasts via a sequence of reactions."
        llm = FakeLLM()
        article = _make_article(
            lede="lede",
            sections=[
                _make_section(title="Process", paragraphs=[first_para]),
                _make_section(
                    title="Filler",
                    paragraphs=[" ".join(f"f{i}" for i in range(200))],
                ),
            ],
        )
        summariser = _build_summariser(
            llm, long_threshold_tokens=10, per_section_tokens=30
        )

        await summariser.summarise(article)

        prompt = llm.calls[0]["prompt"]
        # The first sentence is included under "## Process".
        process_idx = prompt.index("## Process")
        filler_idx = prompt.index("## Filler")
        process_window = prompt[process_idx:filler_idx]
        assert "Photosynthesis converts light energy into chemical energy." in process_window
        # The second sentence MUST NOT leak into the snippet.
        assert "chloroplasts via a sequence of reactions" not in process_window

    async def test_long_path_section_with_empty_paragraphs_emits_heading_only(
        self,
    ) -> None:
        llm = FakeLLM()
        article = _make_article(
            lede="lede",
            sections=[
                _make_section(title="Empty", paragraphs=[]),
                _make_section(
                    title="Filler",
                    paragraphs=[" ".join(f"f{i}" for i in range(200))],
                ),
            ],
        )
        summariser = _build_summariser(llm, long_threshold_tokens=10)

        await summariser.summarise(article)

        prompt = llm.calls[0]["prompt"]
        # Heading still appears (so the outline is complete), with no snippet.
        assert "## Empty" in prompt


class TestNumCtxWiring:
    async def test_passes_constructor_num_ctx_to_llm_generate(self) -> None:
        llm = FakeLLM()
        article = _make_article(lede="short", sections=[])
        summariser = _build_summariser(llm, num_ctx=8192)

        await summariser.summarise(article)

        assert llm.calls[0]["num_ctx"] == 8192

    async def test_num_ctx_is_used_on_both_paths(self) -> None:
        # Short and long path must both wire the constructor num_ctx — neither
        # path should bypass it with a hardcoded default.
        for body_words in (5, 500):  # short path, then long path
            llm = FakeLLM()
            article = _make_article(
                lede="lede",
                sections=[
                    _make_section(
                        title="Body",
                        paragraphs=[" ".join(f"w{i}" for i in range(body_words))],
                    )
                ],
            )
            summariser = _build_summariser(llm, long_threshold_tokens=100, num_ctx=12345)

            await summariser.summarise(article)

            assert llm.calls[0]["num_ctx"] == 12345


class TestErrorPropagation:
    async def test_llm_response_invalid_propagates_unchanged(self) -> None:
        underlying = LLMResponseInvalidError("empty response")
        llm = FakeLLM(raise_on_generate=underlying)
        article = _make_article(lede="lede words", sections=[])
        summariser = _build_summariser(llm, long_threshold_tokens=100)

        with pytest.raises(LLMResponseInvalidError) as exc_info:
            await summariser.summarise(article)

        # Identity check: the exact instance the FakeLLM raised reached the caller.
        assert exc_info.value is underlying

    async def test_ollama_timeout_propagates_unchanged(self) -> None:
        underlying = OllamaTimeoutError("slow")
        llm = FakeLLM(raise_on_generate=underlying)
        article = _make_article(lede="lede words", sections=[])
        summariser = _build_summariser(llm, long_threshold_tokens=100)

        with pytest.raises(OllamaTimeoutError) as exc_info:
            await summariser.summarise(article)

        assert exc_info.value is underlying

    async def test_ollama_unreachable_propagates_unchanged(self) -> None:
        underlying = OllamaUnreachableError("down")
        llm = FakeLLM(raise_on_generate=underlying)
        article = _make_article(lede="lede words", sections=[])
        summariser = _build_summariser(llm, long_threshold_tokens=100)

        with pytest.raises(OllamaUnreachableError) as exc_info:
            await summariser.summarise(article)

        assert exc_info.value is underlying
