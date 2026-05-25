"""Application settings — one ``Settings`` object reading every env var from
DESIGN §6.4.

Defaults match the §6.4 table verbatim. ``.env`` is read when present (gitignored
per NF2); environment variables override the file. ``Settings`` is the single
source of truth for runtime configuration — deps.py wires its fields into
component constructors so the wiring is centralised and testable.

The class is excluded from coverage per pyproject ``[tool.coverage.run] omit``
because it's pure declarative configuration with no behaviour to test beyond
"defaults match the table", which ``tests/unit/test_health.py`` spot-checks.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every env var from DESIGN §6.4, grouped by component."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM / Ollama (DESIGN §4.1, §4.2) ---
    llm_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://ollama:11434"
    ollama_timeout_s: float = 120.0
    ollama_warmup_enabled: bool = True
    warmup_timeout_s: float = 30.0
    summary_num_ctx: int = 8192
    chat_num_ctx: int = 4096

    # --- Embedding (DESIGN §4.3, §12 Challenge 1) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64

    # --- Vector store (DESIGN §4.4, A2/A7) ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "active_article"
    retrieval_top_k: int = 4
    similarity_metric: str = "cosine"

    # --- Chunking (DESIGN §4.5, A10) ---
    chunk_section_ceiling_tokens: int = 1000
    chunk_sub_size_tokens: int = 512
    chunk_sub_overlap_tokens: int = 64

    # --- Article (DESIGN §4.5, A9/U13, F9) ---
    article_char_cap: int = 200_000
    article_min_chars: int = 500

    # --- Summarisation (DESIGN §4.7, A4) ---
    summary_long_threshold_tokens: int = 3500
    summary_per_section_tokens: int = 30

    # --- Wikipedia REST API (DESIGN §3, A13/U6) ---
    wikipedia_api_base: str = "https://en.wikipedia.org/api/rest_v1"
    wikipedia_user_agent: str = (
        "silver-spoon/0.1 (https://github.com/<user>/silver-spoon2; contact: <email>)"
    )
    wikipedia_fetch_timeout_s: float = 15.0

    # --- Logging ---
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-singleton ``Settings``. ``lru_cache`` makes this safe to call
    from every ``Depends(...)`` provider without re-parsing env each request.
    Tests that need to override env can clear the cache via
    ``get_settings.cache_clear()``.
    """
    return Settings()
