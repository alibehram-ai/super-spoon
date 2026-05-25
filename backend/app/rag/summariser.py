"""Summariser per A4 / DESIGN §4.7.

Single LLM call always. Branches on cleaned-body token count (lede + sections
joined, counted once — not lede counted separately and again inside sections):

  - tokens <= long_threshold_tokens -> short path: render
    `prompt_template_short.format(body=<full linearised body>)`.

  - tokens >  long_threshold_tokens -> long path: build a structured outline
    that is `lede verbatim + per-section "## <title>\\n<first sentence capped
    at per_section_tokens>"`, then render `prompt_template_long.format(outline=...)`.

Always calls `llm.generate(prompt, num_ctx=self._num_ctx)` exactly once.
Propagates LLMResponseInvalidError / OllamaUnreachableError / OllamaTimeoutError
unchanged — error mapping for the API layer is T10a's concern.

First-sentence cap detail: the tokenizer Protocol is encode-only (no decode),
so we approximate the cap by binary-searching the largest word prefix whose
token count stays within budget. Under the fake one-token-per-word tokenizer
this is exact; under SBERT's subword tokenizer it's a tight underestimate
(slightly fewer subword tokens than the budget allows), which matches A4's
"~30 tokens" wording.
"""

import re
from collections.abc import Callable

from backend.app.domain.models import CleanedArticle, Section
from backend.app.llm.base import LLMClient

_PARA_SEP = "\n\n"
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


class Summariser:
    def __init__(
        self,
        llm: LLMClient,
        *,
        num_ctx: int,
        long_threshold_tokens: int = 3500,
        per_section_tokens: int = 30,
        tokenizer: Callable[[str], list[int]],
        prompt_template_short: str,
        prompt_template_long: str,
    ) -> None:
        self._llm = llm
        self._num_ctx = num_ctx
        self._long_threshold = long_threshold_tokens
        self._per_section_tokens = per_section_tokens
        self._tokenizer = tokenizer
        self._prompt_template_short = prompt_template_short
        self._prompt_template_long = prompt_template_long

    async def summarise(self, article: CleanedArticle) -> str:
        body = self._linearise_body(article)
        body_token_count = len(self._tokenizer(body))

        if body_token_count <= self._long_threshold:
            prompt = self._prompt_template_short.format(body=body)
        else:
            outline = self._build_outline(article)
            prompt = self._prompt_template_long.format(outline=outline)

        return await self._llm.generate(prompt, num_ctx=self._num_ctx)

    @staticmethod
    def _linearise_body(article: CleanedArticle) -> str:
        """Reconstruct the linearised body the cleaner's char-offset math
        assumes: lede + section_1_text + section_2_text + ..., joined with
        \\n\\n. Lede counted once; not double-counted inside sections.
        """
        parts: list[str] = []
        if article.lede:
            parts.append(article.lede)
        for section in article.sections:
            section_text = _PARA_SEP.join(section.paragraphs)
            if section_text:
                parts.append(section_text)
        return _PARA_SEP.join(parts)

    def _build_outline(self, article: CleanedArticle) -> str:
        parts: list[str] = []
        if article.lede:
            parts.append(article.lede)
        for section in article.sections:
            snippet = self._first_sentence_capped(section)
            heading = f"## {section.title}"
            parts.append(f"{heading}\n{snippet}" if snippet else heading)
        return _PARA_SEP.join(parts)

    def _first_sentence_capped(self, section: Section) -> str:
        if not section.paragraphs:
            return ""
        first_paragraph = section.paragraphs[0]
        if not first_paragraph.strip():
            return ""
        first_sentence = _SENTENCE_END_RE.split(first_paragraph, maxsplit=1)[0]
        return self._truncate_to_token_budget(
            first_sentence, self._per_section_tokens
        )

    def _truncate_to_token_budget(self, text: str, budget: int) -> str:
        """Largest whitespace-bounded prefix of `text` whose token count
        is <= budget. Binary search on word slices so cost is logarithmic.
        """
        if len(self._tokenizer(text)) <= budget:
            return text
        words = text.split()
        if not words:
            return text
        lo, hi = 0, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = " ".join(words[:mid])
            if len(self._tokenizer(candidate)) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return " ".join(words[:lo])
