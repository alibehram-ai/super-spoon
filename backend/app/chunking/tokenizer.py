"""Lazy SBERT tokenizer wrapper.

The Chunker depends on a `Callable[[str], list[int]]` (DESIGN §3) — kept abstract
so unit tests pass a deterministic fake and don't pull in the sentence-transformers
dep. This module supplies the production callable; importing the module does NOT
load the model. The first call constructs the SentenceTransformer and caches it.

`sentence-transformers` is added in T05; until then, calling
`get_sbert_tokenizer()` will raise ImportError. The Chunker class itself never
imports this module, so the rest of the system stays runnable.
"""

from collections.abc import Callable

_cached_tokenizer: Callable[[str], list[int]] | None = None


def get_sbert_tokenizer(
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> Callable[[str], list[int]]:
    """Return a callable that tokenizes a string into token ids.

    Lazy: the SentenceTransformer is loaded on first call and cached. Subsequent
    calls (with the same default model) reuse the cached callable. The model name
    is fixed across a process — passing a different name after the first call
    still returns the cached tokenizer, by design.
    """
    global _cached_tokenizer
    if _cached_tokenizer is None:
        from sentence_transformers import SentenceTransformer  # noqa: I001

        model = SentenceTransformer(model_name)
        tokenizer = model.tokenizer

        def _encode(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

        _cached_tokenizer = _encode
    return _cached_tokenizer
