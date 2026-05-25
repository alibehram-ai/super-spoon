import pytest

from backend.app.chunking.chunker import Chunker
from backend.app.domain.models import CleanedArticle, Section


def fake_tokenizer(s: str) -> list[int]:
    """One token per whitespace-separated word (deterministic, no model load)."""
    return list(range(len(s.split())))


_PARA_SEP = "\n\n"
_PARA_SEP_LEN = len(_PARA_SEP)


def make_section(
    *,
    title: str,
    paragraphs: list[str],
    section_index: int,
    char_offset_start: int,
) -> Section:
    """Construct a Section whose char offsets match the cleaner's contract:
    char_offset_end = char_offset_start + len("\\n\\n".join(paragraphs)).
    """
    if paragraphs:
        body_len = sum(len(p) for p in paragraphs) + _PARA_SEP_LEN * (len(paragraphs) - 1)
    else:
        body_len = 0
    return Section(
        section_index=section_index,
        title=title,
        paragraphs=paragraphs,
        char_offset_start=char_offset_start,
        char_offset_end=char_offset_start + body_len,
    )


def make_article(sections: list[Section], *, lede: str = "") -> CleanedArticle:
    if sections:
        total = sections[-1].char_offset_end
    else:
        total = len(lede)
    return CleanedArticle(
        title="Test Article",
        lede=lede,
        sections=sections,
        total_chars=total,
        truncated=False,
    )


def linearised_body(article: CleanedArticle) -> str:
    """Reconstruct the linearised body string in the same order the cleaner uses,
    so chunk char-offset round-trips can be asserted against a single string.
    """
    parts: list[str] = []
    if article.lede:
        parts.append(article.lede)
    for s in article.sections:
        if s.paragraphs:
            parts.append(_PARA_SEP.join(s.paragraphs))
    return _PARA_SEP.join(parts)


class TestSectionUnderCeiling:
    def test_emits_one_chunk_with_full_section_text(self) -> None:
        words = " ".join(f"w{i}" for i in range(50))  # 50 tokens via fake tokenizer
        section = make_section(
            title="Body",
            paragraphs=[words],
            section_index=0,
            char_offset_start=100,
        )
        # No lede here — this test isolates the single-section-under-ceiling path.
        # Lede chunking is exercised in TestLedeChunking.
        article = make_article([section])
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].chunk_index_in_section == 0
        assert chunks[0].section_index == 0
        assert chunks[0].section_title == "Body"
        assert chunks[0].text == words
        assert chunks[0].char_offset_start == 100
        assert chunks[0].char_offset_end == 100 + len(words)

    def test_multi_paragraph_section_under_ceiling_joins_with_double_newline(self) -> None:
        p1 = "alpha beta gamma"
        p2 = "delta epsilon zeta"
        section = make_section(
            title="Body",
            paragraphs=[p1, p2],
            section_index=0,
            char_offset_start=0,
        )
        article = make_article([section])
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        assert len(chunks) == 1
        assert chunks[0].text == f"{p1}{_PARA_SEP}{p2}"


class TestSectionOverCeilingSubChunks:
    def _build_long_section(self, *, n_paragraphs: int, tokens_per_para: int) -> Section:
        # Each paragraph: `tokens_per_para` whitespace-separated tokens, prefixed with
        # a paragraph marker so paragraphs are distinguishable (matters for the
        # "boundary alignment" test).
        paragraphs = [
            f"P{i} " + " ".join(f"t{i}-{j}" for j in range(tokens_per_para - 1))
            for i in range(n_paragraphs)
        ]
        return make_section(
            title="Long",
            paragraphs=paragraphs,
            section_index=0,
            char_offset_start=0,
        )

    def test_long_section_produces_multiple_chunks(self) -> None:
        section = self._build_long_section(n_paragraphs=12, tokens_per_para=100)
        article = make_article([section])
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        assert len(chunks) >= 3

    def test_consecutive_subchunks_overlap_by_at_least_sub_overlap_tokens(self) -> None:
        section = self._build_long_section(n_paragraphs=12, tokens_per_para=100)
        article = make_article([section])
        sub_overlap = 64
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=sub_overlap,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        for prev, curr in zip(chunks, chunks[1:]):
            # Both come from the same section; overlap is computed in token space.
            prev_tokens = set(prev.text.split())
            curr_tokens = set(curr.text.split())
            overlap_tokens = prev_tokens & curr_tokens
            assert len(overlap_tokens) >= sub_overlap, (
                f"Overlap {len(overlap_tokens)} < sub_overlap {sub_overlap}"
            )

    def test_subchunk_boundaries_align_with_paragraph_starts(self) -> None:
        section = self._build_long_section(n_paragraphs=12, tokens_per_para=100)
        article = make_article([section])
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        # Each chunk's first characters must match one of the section's paragraph
        # openings (paragraphs in this fixture start with the unique "P{i} " marker).
        paragraph_openings = {p[: p.index(" ") + 1] for p in section.paragraphs}
        for chunk in chunks:
            head = chunk.text[: chunk.text.index(" ") + 1]
            assert head in paragraph_openings, f"Chunk does not start at a paragraph boundary: {chunk.text[:30]!r}"

    def test_subchunk_char_offsets_round_trip_into_section_text(self) -> None:
        section = self._build_long_section(n_paragraphs=12, tokens_per_para=100)
        article = make_article([section])
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)
        body = linearised_body(article)

        for chunk in chunks:
            assert body[chunk.char_offset_start : chunk.char_offset_end] == chunk.text


class TestGlobalChunkIndexAcrossSections:
    def test_chunk_index_is_monotonic_across_sections(self) -> None:
        # Two short sections (each emitted as one chunk) + one long section
        # (sub-chunked into several). Global index should be 0, 1, 2, ..., contiguous.
        short_a = make_section(
            title="A",
            paragraphs=["alpha beta gamma"],
            section_index=0,
            char_offset_start=0,
        )
        # Build the long section AFTER short_a, so offsets stack correctly.
        long_paragraphs = [
            f"P{i} " + " ".join(f"t{i}-{j}" for j in range(99)) for i in range(12)
        ]
        long_section_start = short_a.char_offset_end + _PARA_SEP_LEN
        long_section = make_section(
            title="Long",
            paragraphs=long_paragraphs,
            section_index=1,
            char_offset_start=long_section_start,
        )
        short_c_start = long_section.char_offset_end + _PARA_SEP_LEN
        short_c = make_section(
            title="C",
            paragraphs=["xi omicron pi"],
            section_index=2,
            char_offset_start=short_c_start,
        )
        article = make_article([short_a, long_section, short_c])
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        # Three sections, with the middle producing >1 chunk.
        assert len(chunks) >= 4
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chunk_index_in_section_resets_per_section(self) -> None:
        # Each section is short → one chunk each → chunk_index_in_section is 0 every time.
        sections = []
        offset = 0
        for i in range(3):
            section = make_section(
                title=f"S{i}",
                paragraphs=[f"only paragraph {i} here"],
                section_index=i,
                char_offset_start=offset,
            )
            sections.append(section)
            offset = section.char_offset_end + _PARA_SEP_LEN
        article = make_article(sections)
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        assert [c.chunk_index_in_section for c in chunks] == [0, 0, 0]
        assert [c.section_index for c in chunks] == [0, 1, 2]

    def test_chunk_index_in_section_increments_within_a_subchunked_section(self) -> None:
        long_paragraphs = [
            f"P{i} " + " ".join(f"t{i}-{j}" for j in range(99)) for i in range(12)
        ]
        section = make_section(
            title="Long",
            paragraphs=long_paragraphs,
            section_index=0,
            char_offset_start=0,
        )
        article = make_article([section])
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        assert [c.chunk_index_in_section for c in chunks] == list(range(len(chunks)))


class TestF5MetadataFields:
    def test_every_chunk_has_all_required_metadata(self) -> None:
        section = make_section(
            title="Body",
            paragraphs=["alpha beta gamma delta"],
            section_index=0,
            char_offset_start=42,
        )
        article = make_article([section])
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        chunk = chunks[0]
        # F5 metadata fields, per REQUIREMENTS §5: section_title, section_index,
        # chunk_index_in_section, char_offset_start, char_offset_end, text.
        assert chunk.section_title == "Body"
        assert chunk.section_index == 0
        assert chunk.chunk_index_in_section == 0
        assert chunk.char_offset_start == 42
        assert chunk.char_offset_end == 42 + len("alpha beta gamma delta")
        assert chunk.text == "alpha beta gamma delta"

    def test_section_title_propagates_to_subchunks_in_a_long_section(self) -> None:
        long_paragraphs = [
            f"P{i} " + " ".join(f"t{i}-{j}" for j in range(99)) for i in range(12)
        ]
        section = make_section(
            title="Light-dependent reactions",
            paragraphs=long_paragraphs,
            section_index=3,
            char_offset_start=0,
        )
        article = make_article([section])
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        assert {c.section_title for c in chunks} == {"Light-dependent reactions"}
        assert {c.section_index for c in chunks} == {3}


class TestEdgeCases:
    def test_one_section_one_paragraph_emits_exactly_one_chunk(self) -> None:
        section = make_section(
            title="Solo",
            paragraphs=["the only paragraph here is short and sweet"],
            section_index=0,
            char_offset_start=0,
        )
        article = make_article([section])
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        assert len(chunks) == 1

    def test_empty_paragraphs_in_section_are_skipped_silently(self) -> None:
        # A section with no paragraphs should not crash and should emit no chunks.
        empty_section = make_section(
            title="Empty",
            paragraphs=[],
            section_index=0,
            char_offset_start=0,
        )
        kept_section = make_section(
            title="Real",
            paragraphs=["non empty content here"],
            section_index=1,
            char_offset_start=0,
        )
        article = make_article([empty_section, kept_section])
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        assert len(chunks) == 1
        assert chunks[0].section_title == "Real"

    def test_single_paragraph_above_subsize_is_emitted_whole(self) -> None:
        # One paragraph alone, larger than sub_size. We cannot split mid-paragraph,
        # so the chunker emits one chunk containing the whole paragraph.
        para = " ".join(f"t{i}" for i in range(800))  # 800 tokens, one paragraph
        section = make_section(
            title="Giant",
            paragraphs=[para],
            section_index=0,
            char_offset_start=0,
        )
        article = make_article([section])
        chunker = Chunker(
            section_ceiling_tokens=200,  # forces sub-chunk path
            sub_size_tokens=300,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        assert len(chunks) == 1
        assert chunks[0].text == para

    def test_overlap_not_less_than_size_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            Chunker(
                section_ceiling_tokens=1000,
                sub_size_tokens=512,
                sub_overlap_tokens=512,
                tokenizer=fake_tokenizer,
            )


class TestLedeChunking:
    """The chunker must produce chunks for the lede so RAG can retrieve from
    introductory content — DESIGN §3 left this implicit. Lede chunks carry
    section_index=0 and section_title="Introduction" by default; real sections
    keep their cleaner-assigned section_index (so a section_index=0 collision
    with the first real section is possible and acceptable — chunk_index is
    the unique key).
    """

    def test_lede_emits_a_chunk_with_introduction_title(self) -> None:
        lede = "alpha beta gamma delta"  # 4 tokens
        section = make_section(
            title="Body",
            paragraphs=["body content here"],
            section_index=0,
            char_offset_start=len(lede) + _PARA_SEP_LEN,
        )
        article = make_article([section], lede=lede)
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        assert len(chunks) == 2
        assert chunks[0].section_title == "Introduction"
        assert chunks[0].text == lede
        assert chunks[0].section_index == 0
        assert chunks[0].chunk_index == 0
        assert chunks[0].chunk_index_in_section == 0
        assert chunks[1].section_title == "Body"
        assert chunks[1].chunk_index == 1

    def test_lede_chunk_char_offsets_round_trip_into_linearised_body(self) -> None:
        lede = "alpha beta gamma\n\ndelta epsilon zeta"
        section = make_section(
            title="Body",
            paragraphs=["body paragraph here"],
            section_index=0,
            char_offset_start=len(lede) + _PARA_SEP_LEN,
        )
        article = make_article([section], lede=lede)
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)
        body = linearised_body(article)

        for chunk in chunks:
            assert body[chunk.char_offset_start : chunk.char_offset_end] == chunk.text

    def test_empty_lede_emits_no_lede_chunk(self) -> None:
        section = make_section(
            title="Body",
            paragraphs=["body content here"],
            section_index=0,
            char_offset_start=0,
        )
        article = make_article([section], lede="")
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        assert len(chunks) == 1
        assert chunks[0].section_title == "Body"

    def test_long_lede_is_sub_chunked(self) -> None:
        # 12 lede paragraphs × 100 tokens each = 1200 tokens, joined with \n\n.
        lede_paragraphs = [
            f"L{i} " + " ".join(f"u{i}-{j}" for j in range(99)) for i in range(12)
        ]
        lede = _PARA_SEP.join(lede_paragraphs)
        article = CleanedArticle(
            title="Test Article",
            lede=lede,
            sections=[],
            total_chars=len(lede),
            truncated=False,
        )
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        assert len(chunks) >= 3
        assert {c.section_title for c in chunks} == {"Introduction"}
        # chunk_index_in_section increments within the lede (which is its own section).
        assert [c.chunk_index_in_section for c in chunks] == list(range(len(chunks)))
        # chunk_index is globally monotonic from 0.
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_lede_title_is_configurable(self) -> None:
        lede = "alpha beta gamma"
        article = CleanedArticle(
            title="Test",
            lede=lede,
            sections=[],
            total_chars=len(lede),
            truncated=False,
        )
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
            lede_section_title="Lead",
        )

        chunks = chunker.chunk(article)

        assert chunks[0].section_title == "Lead"

    def test_lede_only_article_still_produces_chunks(self) -> None:
        # Resolves the "never returns []" contract gap: CleanedArticle with a
        # 500-char lede and no sections (allowed by the cleaner) now produces
        # at least one chunk for the lede.
        lede = " ".join(f"w{i}" for i in range(100))
        article = CleanedArticle(
            title="Stub",
            lede=lede,
            sections=[],
            total_chars=len(lede),
            truncated=False,
        )
        chunker = Chunker(
            section_ceiling_tokens=1000,
            sub_size_tokens=512,
            sub_overlap_tokens=64,
            tokenizer=fake_tokenizer,
        )

        chunks = chunker.chunk(article)

        assert len(chunks) >= 1
        assert chunks[0].section_title == "Introduction"


class TestSentenceTransformersNotLoadedDuringUnitTests:
    def test_tokenizer_module_does_not_bind_sentence_transformers_at_module_scope(self) -> None:
        # The tokenizer wrapper must keep its sentence-transformers import lazy
        # (inside `get_sbert_tokenizer`), so the chunker can be unit-tested with
        # a fake tokenizer and downstream modules can import tokenizer.py
        # without pulling the heavy model dep in. We assert structurally:
        # neither the `SentenceTransformer` class nor the `sentence_transformers`
        # module appears in the tokenizer module's own namespace.
        import importlib

        mod = importlib.import_module("backend.app.chunking.tokenizer")
        namespace = vars(mod)
        assert "SentenceTransformer" not in namespace
        assert "sentence_transformers" not in namespace
