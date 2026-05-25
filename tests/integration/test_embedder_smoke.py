"""Real ``BAAI/bge-small-en-v1.5`` smoke test (DESIGN §7.2).

Confirms the SBERT abstraction binds to the real model correctly. The unit
suite (``tests/unit/test_embedder.py``) patches ``SentenceTransformer`` to
keep the default run hermetic; this test is the real-model counterpart
under the ``integration`` marker.

First run downloads the model (~133 MB) into the local HuggingFace cache
under ``~/.cache/huggingface/``; subsequent runs reuse it.
"""

from __future__ import annotations

import math

import pytest

from backend.app.embedding.sbert import SentenceTransformersEmbedder


pytestmark = pytest.mark.integration


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def test_real_embedder_loads_and_embeds() -> None:
    embedder = SentenceTransformersEmbedder(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=8,
    )

    assert embedder.vector_size == 384

    vectors = embedder.embed_texts(
        ["Photosynthesis converts light to chemical energy.", "The quick brown fox."]
    )
    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)
    # Distinct prose -> distinct embeddings.
    assert _cosine(vectors[0], vectors[1]) < 1.0

    trio = embedder.embed_texts(["the", "cat", "sat"])
    assert len(trio) == 3
    for v in trio:
        assert len(v) == 384
        magnitude = math.sqrt(sum(x * x for x in v))
        # bge-small returns L2-normalised vectors; allow generous tolerance
        # so this doesn't false-fail across minor model revisions.
        assert math.isclose(magnitude, 1.0, abs_tol=0.05), magnitude
