"""Integration-suite fixtures.

The integration suite assumes the docker compose stack is already running
(per DESIGN §7.2 / TASKS T16). This conftest reads the three endpoint URLs
from env (with localhost defaults that match T15's compose port mapping)
and exposes a single session-scoped readiness gate, ``wait_for_app_healthy``,
that every integration test depends on. The gate polls ``/api/health`` until
it returns 200; ``warmup_ok`` may be either ``true`` or ``false`` — both
mean the app is up and serving (DESIGN §4.2's "graceful degradation"
contract: a cold start is still a healthy app).

The ``integration`` marker itself is registered in ``pyproject.toml`` so
default ``pytest`` invocations skip this suite. Run it explicitly with
``uv run pytest tests/integration -v`` after ``docker compose up -d``.
"""

from __future__ import annotations

import os
import time

import httpx
import pytest


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@pytest.fixture(scope="session")
def qdrant_url() -> str:
    return _env("QDRANT_URL", "http://localhost:6333")


@pytest.fixture(scope="session")
def ollama_base_url() -> str:
    return _env("OLLAMA_BASE_URL", "http://localhost:11434")


@pytest.fixture(scope="session")
def app_base_url() -> str:
    return _env("APP_BASE_URL", "http://localhost:8000").rstrip("/")


@pytest.fixture(scope="session")
def wait_for_app_healthy(app_base_url: str) -> str:
    """Block until ``GET ${APP_BASE_URL}/api/health`` returns 200.

    Polls every 1 s; times out after 90 s with an error pointing at T15's
    compose-level app healthcheck (whose ``start_period: 30s`` + warmup
    budget puts the app firmly inside this window on any reasonable host).
    Returns ``app_base_url`` so consumers can chain it.
    """
    deadline = time.monotonic() + 90.0
    last_error: str | None = None
    health_url = f"{app_base_url}/api/health"

    while time.monotonic() < deadline:
        try:
            response = httpx.get(health_url, timeout=2.0)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200 and "warmup_ok" in response.json():
                return app_base_url
            last_error = (
                f"HTTP {response.status_code} body={response.text[:200]!r}"
            )
        time.sleep(1.0)

    pytest.fail(
        "App did not become healthy within 90 s polling "
        f"{health_url}. Last error: {last_error}. "
        "Is `docker compose --profile containerised-ollama up -d` running? "
        "See T15's app healthcheck (start_period 30s + WARMUP_TIMEOUT_S=30)."
    )
