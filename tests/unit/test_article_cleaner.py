import pytest

from backend.app.domain.models import RawSection, WikipediaArticle
from backend.app.wikipedia.cleaner import (
    ArticleCleaner,
    ArticleTooShortError,
)


def _article(
    *,
    sections: list[RawSection],
    lede: str = "",
    title: str = "Photosynthesis",
) -> WikipediaArticle:
    return WikipediaArticle(
        canonical_title=title,
        lede=lede,
        sections=sections,
        is_disambiguation=False,
    )


def _para(n_chars: int, *, prefix: str = "p") -> str:
    body = "x" * max(0, n_chars - len(prefix))
    return f"{prefix}{body}"


class TestDroppedSections:
    def test_default_drop_list_removes_known_low_signal_sections(self) -> None:
        article = _article(
            lede="Intro lede paragraph of moderate length. " * 20,
            sections=[
                RawSection(title="History", paragraphs=["History body paragraph. " * 30]),
                RawSection(title="References", paragraphs=["Cited works. " * 10]),
                RawSection(title="Biology", paragraphs=["Biology body paragraph. " * 30]),
                RawSection(title="External links", paragraphs=["External link. " * 10]),
                RawSection(title="See also", paragraphs=["Related. " * 10]),
                RawSection(title="Further reading", paragraphs=["Read on. " * 10]),
                RawSection(title="Notes", paragraphs=["Note. " * 10]),
            ],
        )

        result = ArticleCleaner().clean(article)

        titles = [s.title for s in result.sections]
        assert titles == ["History", "Biology"]

    def test_section_index_is_reassigned_contiguously_after_drops(self) -> None:
        article = _article(
            lede="Some lede content. " * 30,
            sections=[
                RawSection(title="Intro", paragraphs=["Intro body. " * 30]),
                RawSection(title="References", paragraphs=["Refs body. " * 10]),
                RawSection(title="Body", paragraphs=["Body content. " * 30]),
                RawSection(title="External links", paragraphs=["Links. " * 10]),
            ],
        )

        result = ArticleCleaner().clean(article)

        assert [s.section_index for s in result.sections] == [0, 1]
        assert [s.title for s in result.sections] == ["Intro", "Body"]

    @pytest.mark.parametrize(
        "title",
        ["references", "REFERENCES", "References ", " References", "ReFeReNcEs"],
    )
    def test_drop_list_match_is_case_insensitive_and_trimmed(self, title: str) -> None:
        article = _article(
            lede="Some lede content. " * 30,
            sections=[
                RawSection(title="Kept", paragraphs=["Kept body. " * 30]),
                RawSection(title=title, paragraphs=["Refs body. " * 10]),
            ],
        )

        result = ArticleCleaner().clean(article)

        assert [s.title for s in result.sections] == ["Kept"]

    def test_custom_drop_list_overrides_default(self) -> None:
        article = _article(
            lede="Lede content. " * 30,
            sections=[
                RawSection(title="References", paragraphs=["Refs. " * 30]),
                RawSection(title="Custom", paragraphs=["Custom. " * 30]),
            ],
        )

        cleaner = ArticleCleaner(dropped_section_titles=frozenset({"Custom"}))
        result = cleaner.clean(article)

        # With the default drop list overridden, References is now retained
        # and Custom is dropped.
        assert [s.title for s in result.sections] == ["References"]


class TestCitationStripping:
    def test_inline_numeric_citation_markers_are_removed(self) -> None:
        article = _article(
            lede="Photosynthesis is a process[1] used by plants[2][3].",
            sections=[
                RawSection(
                    title="History",
                    paragraphs=[
                        "Early work[12] established the field[3].",
                        "Padding paragraph that boosts overall length so the article exceeds the min_chars floor. "
                        * 8,
                    ],
                ),
            ],
        )

        result = ArticleCleaner().clean(article)

        assert "[1]" not in result.lede
        assert "[2]" not in result.lede
        assert "[3]" not in result.lede
        assert result.lede == "Photosynthesis is a process used by plants."
        joined = "\n\n".join(result.sections[0].paragraphs)
        assert "[12]" not in joined
        assert "Early work established the field." in joined

    def test_non_numeric_bracketed_text_is_preserved(self) -> None:
        article = _article(
            lede="A claim[sic] survives. A [citation needed] survives too.",
            sections=[
                RawSection(
                    title="Body",
                    paragraphs=[
                        "Padding paragraph that boosts overall length so the article exceeds the min_chars floor. "
                        * 8,
                    ],
                ),
            ],
        )

        result = ArticleCleaner().clean(article)

        assert "[sic]" in result.lede
        assert "[citation needed]" in result.lede


class TestTotalCharsAndOffsets:
    def test_total_chars_matches_linearised_body_length(self) -> None:
        article = _article(
            lede="L" * 200,
            sections=[
                RawSection(title="A", paragraphs=["a" * 300]),
                RawSection(title="B", paragraphs=["b" * 400]),
            ],
        )

        result = ArticleCleaner().clean(article)

        # body = lede + "\n\n" + section_a + "\n\n" + section_b
        expected = 200 + 2 + 300 + 2 + 400
        assert result.total_chars == expected

    def test_section_offsets_round_trip_within_linearised_body(self) -> None:
        lede = "L" * 200
        a_para = "a" * 300
        b_para = "b" * 400
        article = _article(
            lede=lede,
            sections=[
                RawSection(title="A", paragraphs=[a_para]),
                RawSection(title="B", paragraphs=[b_para]),
            ],
        )

        result = ArticleCleaner().clean(article)

        body = "\n\n".join([result.lede, *[
            "\n\n".join(s.paragraphs) for s in result.sections
        ]])

        for section in result.sections:
            section_text = "\n\n".join(section.paragraphs)
            assert body[section.char_offset_start : section.char_offset_end] == section_text

    def test_offsets_are_monotonic_across_sections(self) -> None:
        article = _article(
            lede="L" * 200,
            sections=[
                RawSection(title=f"S{i}", paragraphs=[f"body{i} " * 30])
                for i in range(5)
            ],
        )

        result = ArticleCleaner().clean(article)

        for prev, curr in zip(result.sections, result.sections[1:]):
            assert prev.char_offset_end < curr.char_offset_start


class TestMinCharsFloor:
    def test_below_min_chars_raises(self) -> None:
        article = _article(
            lede="too short " * 5,  # ~50 chars
            sections=[
                RawSection(title="Body", paragraphs=["a few more words"]),
            ],
        )

        with pytest.raises(ArticleTooShortError) as exc_info:
            ArticleCleaner().clean(article)

        assert exc_info.value.char_count < 500

    def test_exactly_at_min_chars_passes(self) -> None:
        # Build a body of exactly 500 chars: 500-char lede, no sections.
        article = _article(lede="x" * 500, sections=[])

        result = ArticleCleaner().clean(article)

        assert result.total_chars == 500
        assert result.truncated is False


class TestTruncationAtSectionBoundary:
    def test_250k_body_truncates_to_cap_with_last_section_fully_present(self) -> None:
        # 50 sections of ~5000 chars each + ~500-char lede ≈ 250,500 chars.
        section_para = "s" * 5000
        article = _article(
            lede="L" * 500,
            sections=[
                RawSection(title=f"Section {i}", paragraphs=[section_para])
                for i in range(50)
            ],
        )

        cleaner = ArticleCleaner(char_cap=200_000)
        result = cleaner.clean(article)

        assert result.truncated is True
        assert result.total_chars <= 200_000
        # Last retained section is fully present: its end offset == total_chars
        # (no half-section / mid-paragraph truncation).
        assert result.sections[-1].char_offset_end == result.total_chars
        last_text = "\n\n".join(result.sections[-1].paragraphs)
        assert len(last_text) == 5000

    def test_truncation_drops_only_trailing_sections(self) -> None:
        article = _article(
            lede="L" * 500,
            sections=[
                RawSection(title=f"S{i}", paragraphs=["x" * 5000])
                for i in range(50)
            ],
        )

        cleaner = ArticleCleaner(char_cap=200_000)
        result = cleaner.clean(article)

        # Some sections retained, some dropped, retained are a strict prefix.
        retained_titles = [s.title for s in result.sections]
        assert len(retained_titles) < 50
        assert len(retained_titles) > 0
        assert retained_titles == [f"S{i}" for i in range(len(retained_titles))]
        # Indices are still contiguous from 0 (no holes).
        assert [s.section_index for s in result.sections] == list(range(len(retained_titles)))

    def test_under_cap_returns_truncated_false(self) -> None:
        article = _article(
            lede="L" * 1000,
            sections=[RawSection(title="Body", paragraphs=["b" * 1000])],
        )

        result = ArticleCleaner(char_cap=200_000).clean(article)

        assert result.truncated is False
        assert result.total_chars < 200_000


class TestPassThroughFields:
    def test_title_carries_canonical_title_from_input(self) -> None:
        article = _article(
            title="Photosynthesis",
            lede="L" * 600,
            sections=[],
        )

        result = ArticleCleaner().clean(article)

        assert result.title == "Photosynthesis"
