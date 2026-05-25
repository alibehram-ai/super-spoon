import pytest

from backend.app.domain.models import ValidatedUrl
from backend.app.wikipedia.validator import UrlValidationError, validate_url


class TestRejectMalformed:
    @pytest.mark.parametrize(
        "url",
        [
            "hello",
            "ftp://en.wikipedia.org/wiki/X",
            "",
            "https://",
            "wikipedia.org/wiki/X",
            "http://[invalid",  # urlparse raises ValueError on malformed IPv6 brackets.
        ],
    )
    def test_unparseable_or_wrong_scheme(self, url: str) -> None:
        with pytest.raises(UrlValidationError) as exc_info:
            validate_url(url)
        assert exc_info.value.reason == "malformed"


class TestRejectNonEnglish:
    @pytest.mark.parametrize(
        "url",
        [
            "https://fr.wikipedia.org/wiki/Soleil",
            "https://de.m.wikipedia.org/wiki/Sonne",
            "https://es.wikipedia.org/wiki/Sol",
            "https://zh.wikipedia.org/wiki/X",
        ],
    )
    def test_other_language_wikipedia(self, url: str) -> None:
        with pytest.raises(UrlValidationError) as exc_info:
            validate_url(url)
        assert exc_info.value.reason == "non_english"


class TestRejectNonWikipedia:
    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/wiki/X",
            "https://en.wikipedia.com/wiki/X",
            "https://wikipedia.org/wiki/X",
            "https://www.google.com/wiki/X",
        ],
    )
    def test_non_wikipedia_host(self, url: str) -> None:
        with pytest.raises(UrlValidationError) as exc_info:
            validate_url(url)
        assert exc_info.value.reason == "non_wikipedia"


class TestRejectCurid:
    def test_curid_query_rejected(self) -> None:
        with pytest.raises(UrlValidationError) as exc_info:
            validate_url("https://en.wikipedia.org/?curid=12345")
        assert exc_info.value.reason == "curid"

    def test_curid_in_query_with_article_path_rejected(self) -> None:
        # curid wins over path-shape because it indicates an intent the user got wrong.
        with pytest.raises(UrlValidationError) as exc_info:
            validate_url("https://en.wikipedia.org/wiki/Photosynthesis?curid=12345")
        assert exc_info.value.reason == "curid"


class TestRejectNamespace:
    @pytest.mark.parametrize(
        "prefix",
        ["Special:", "Talk:", "Category:", "File:", "Help:", "User:", "Portal:"],
    )
    def test_each_blocked_namespace(self, prefix: str) -> None:
        url = f"https://en.wikipedia.org/wiki/{prefix}Something"
        with pytest.raises(UrlValidationError) as exc_info:
            validate_url(url)
        assert exc_info.value.reason == "namespace"

    def test_percent_encoded_namespace_separator(self) -> None:
        # %3A is the percent-encoded ':' — must still be detected after decode.
        url = "https://en.wikipedia.org/wiki/Special%3ARandom"
        with pytest.raises(UrlValidationError) as exc_info:
            validate_url(url)
        assert exc_info.value.reason == "namespace"


class TestHappyPath:
    def test_canonical_url_returned(self) -> None:
        result = validate_url("https://en.wikipedia.org/wiki/Photosynthesis")
        assert result == ValidatedUrl(
            canonical_url="https://en.wikipedia.org/wiki/Photosynthesis",
            article_title="Photosynthesis",
        )

    def test_mobile_host_normalises_to_canonical(self) -> None:
        result = validate_url("https://en.m.wikipedia.org/wiki/Photosynthesis")
        assert result.canonical_url == "https://en.wikipedia.org/wiki/Photosynthesis"
        assert result.article_title == "Photosynthesis"

    def test_fragment_stripped(self) -> None:
        result = validate_url("https://en.wikipedia.org/wiki/Photosynthesis#History")
        assert result == ValidatedUrl(
            canonical_url="https://en.wikipedia.org/wiki/Photosynthesis",
            article_title="Photosynthesis",
        )

    def test_percent_encoded_title_round_trips(self) -> None:
        result = validate_url("https://en.wikipedia.org/wiki/Carbon%20dioxide")
        assert result.article_title == "Carbon dioxide"
        assert result.canonical_url == "https://en.wikipedia.org/wiki/Carbon%20dioxide"

    def test_underscore_preserved_in_title(self) -> None:
        # Wikipedia treats underscores as spaces in URLs but we keep them literal.
        result = validate_url("https://en.wikipedia.org/wiki/Carbon_dioxide")
        assert result.article_title == "Carbon_dioxide"
        assert result.canonical_url == "https://en.wikipedia.org/wiki/Carbon_dioxide"

    def test_uppercase_host_lowered(self) -> None:
        result = validate_url("https://EN.WIKIPEDIA.ORG/wiki/Photosynthesis")
        assert result.canonical_url == "https://en.wikipedia.org/wiki/Photosynthesis"

    def test_mobile_with_fragment_and_encoded_title(self) -> None:
        result = validate_url(
            "https://en.m.wikipedia.org/wiki/Carbon%20dioxide#Properties"
        )
        assert result.article_title == "Carbon dioxide"
        assert result.canonical_url == "https://en.wikipedia.org/wiki/Carbon%20dioxide"


class TestEmptyTitleAfterDecode:
    @pytest.mark.parametrize(
        "url",
        [
            "https://en.wikipedia.org/wiki/%20",
            "https://en.wikipedia.org/wiki/%20%20",
        ],
    )
    def test_whitespace_only_title_is_malformed(self, url: str) -> None:
        with pytest.raises(UrlValidationError) as exc_info:
            validate_url(url)
        assert exc_info.value.reason == "malformed"

    def test_trailing_slash_with_no_title_is_malformed(self) -> None:
        with pytest.raises(UrlValidationError) as exc_info:
            validate_url("https://en.wikipedia.org/wiki/")
        assert exc_info.value.reason == "malformed"
