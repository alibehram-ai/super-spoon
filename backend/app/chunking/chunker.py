"""Section-aware chunker per DESIGN §4.5 / A10.

For each cleaned section (the lede is treated as a synthetic first section so
its content is retrievable — see "lede handling" below):
  - If section token count <= section_ceiling_tokens: emit one chunk verbatim.
  - Else: walk paragraphs accumulating tokens until >= sub_size_tokens,
    emit a chunk, rewind to the latest paragraph boundary that gives at least
    sub_overlap_tokens of overlap with the just-emitted chunk, continue.

Chunk boundaries always coincide with paragraph boundaries (we never split
mid-paragraph). chunk_index is globally monotonic across the article;
chunk_index_in_section resets per section.

Lede handling (resolves the DESIGN §3 gap where the lede was never chunked):
the chunker prepends a synthetic Section for CleanedArticle.lede with
section_index=0 and section_title=lede_section_title (default "Introduction").
Real sections keep their cleaner-assigned section_index — section_index=0
may therefore appear twice in chunk metadata, distinguished by section_title.
chunk_index remains globally unique. The summariser still reads
CleanedArticle.lede directly for the verbatim outline lede; nothing about
that path changes.
"""

from collections.abc import Callable, Iterator

from backend.app.domain.models import Chunk, CleanedArticle, Section

_PARA_SEP = "\n\n"
_PARA_SEP_LEN = len(_PARA_SEP)
_DEFAULT_LEDE_TITLE = "Introduction"
_LEDE_SECTION_INDEX = 0


class Chunker:
    def __init__(
        self,
        *,
        section_ceiling_tokens: int = 1000,
        sub_size_tokens: int = 512,
        sub_overlap_tokens: int = 64,
        tokenizer: Callable[[str], list[int]],
        lede_section_title: str = _DEFAULT_LEDE_TITLE,
    ) -> None:
        if sub_overlap_tokens >= sub_size_tokens:
            raise ValueError(
                "sub_overlap_tokens must be < sub_size_tokens "
                f"(got overlap={sub_overlap_tokens}, size={sub_size_tokens})"
            )
        self._ceiling = section_ceiling_tokens
        self._sub_size = sub_size_tokens
        self._sub_overlap = sub_overlap_tokens
        self._tokenizer = tokenizer
        self._lede_title = lede_section_title

    def chunk(self, article: CleanedArticle) -> list[Chunk]:
        chunks: list[Chunk] = []
        global_idx = 0

        for section in self._iter_sections(article):
            if not section.paragraphs:
                continue

            section_text = _PARA_SEP.join(section.paragraphs)
            section_tokens = len(self._tokenizer(section_text))

            if section_tokens <= self._ceiling:
                chunks.append(
                    Chunk(
                        chunk_index=global_idx,
                        section_index=section.section_index,
                        chunk_index_in_section=0,
                        section_title=section.title,
                        text=section_text,
                        char_offset_start=section.char_offset_start,
                        char_offset_end=section.char_offset_end,
                    )
                )
                global_idx += 1
                continue

            for chunk in self._sub_chunk_section(section, global_idx):
                chunks.append(chunk)
                global_idx += 1

        return chunks

    def _iter_sections(self, article: CleanedArticle) -> Iterator[Section]:
        """Yield the lede as a synthetic Section (if present) then the cleaner's
        sections. Lede paragraphs are recovered by splitting on `\\n\\n`, which
        round-trips against the cleaner's lede join. Char offsets place the
        lede at [0, len(lede)] — the cleaner reserves that range and offsets
        every real section past the `\\n\\n` separator that follows the lede.
        """
        if article.lede:
            lede_paragraphs = [p for p in article.lede.split(_PARA_SEP) if p]
            if lede_paragraphs:
                yield Section(
                    section_index=_LEDE_SECTION_INDEX,
                    title=self._lede_title,
                    paragraphs=lede_paragraphs,
                    char_offset_start=0,
                    char_offset_end=len(article.lede),
                )
        yield from article.sections

    def _sub_chunk_section(
        self, section: Section, starting_global_idx: int
    ) -> list[Chunk]:
        paragraphs = section.paragraphs
        n = len(paragraphs)
        para_tokens = [len(self._tokenizer(p)) for p in paragraphs]
        para_starts = self._paragraph_char_starts(section)

        out: list[Chunk] = []
        local_idx = 0
        global_idx = starting_global_idx
        start_idx = 0

        while start_idx < n:
            total = 0
            end_idx = start_idx
            while end_idx < n:
                total += para_tokens[end_idx]
                end_idx += 1
                if total >= self._sub_size:
                    break

            text = _PARA_SEP.join(paragraphs[start_idx:end_idx])
            char_start = para_starts[start_idx]
            char_end = char_start + len(text)
            out.append(
                Chunk(
                    chunk_index=global_idx,
                    section_index=section.section_index,
                    chunk_index_in_section=local_idx,
                    section_title=section.title,
                    text=text,
                    char_offset_start=char_start,
                    char_offset_end=char_end,
                )
            )
            local_idx += 1
            global_idx += 1

            if end_idx >= n:
                break

            new_start = self._rewind_to_overlap_boundary(
                end_idx, start_idx, para_tokens
            )
            start_idx = new_start

        return out

    def _rewind_to_overlap_boundary(
        self, end_idx: int, start_idx: int, para_tokens: list[int]
    ) -> int:
        """Find new start index such that paragraphs[new_start:end_idx] has
        at least sub_overlap_tokens of tokens. Walks backward from end_idx.

        Guarantees forward progress (new_start > start_idx) so the outer loop
        cannot stall, even when a single paragraph exceeds sub_overlap_tokens.
        """
        overlap = 0
        new_start = end_idx
        while new_start > 0 and overlap < self._sub_overlap:
            new_start -= 1
            overlap += para_tokens[new_start]

        if new_start <= start_idx:
            new_start = start_idx + 1
        return new_start

    @staticmethod
    def _paragraph_char_starts(section: Section) -> list[int]:
        """Char offset of each paragraph's first byte within the linearised body.

        Paragraphs are joined with `\\n\\n` and the section starts at
        `section.char_offset_start` (per ArticleCleaner). This mirrors the
        cleaner's offset math so chunk char_offset_* round-trip exactly into
        cleaned_body[start:end] == chunk.text.
        """
        starts = [section.char_offset_start]
        for i in range(1, len(section.paragraphs)):
            starts.append(
                starts[i - 1] + len(section.paragraphs[i - 1]) + _PARA_SEP_LEN
            )
        return starts
