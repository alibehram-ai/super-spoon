"""Regenerate the Photosynthesis fixture used by ``tests/integration``.

Hits the live Wikipedia REST API (``/page/summary/{title}`` then
``/page/html/{title}``) and writes the responses to
``tests/fixtures/photosynthesis_rest_api.json`` as a ``{summary, html}``
envelope. Idempotent — re-runs overwrite the file.

Endpoint note: an earlier draft of TASKS.md referenced
``/page/mobile-sections/``; Wikimedia retired that endpoint in 2026 (see
NOTES.md "Incidents"). The fetcher migrated to ``/page/html/`` in commit
0481b71, so the fixture envelope here mirrors that — ``summary`` is the
parsed JSON object, ``html`` is the raw Parsoid HTML string.

Run by hand. Do not wire into CI — this script hits live Wikipedia and is
intentionally non-idempotent across time (article drifts). See README's
"Maintaining fixtures" section (T18) for when to regenerate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import httpx

_ARTICLE_TITLE = "Photosynthesis"
_DEFAULT_BASE = "https://en.wikipedia.org/api/rest_v1"
_DEFAULT_UA = (
    "silver-spoon/0.1 (https://github.com/<user>/silver-spoon2; contact: <email>)"
)
_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "photosynthesis_rest_api.json"
)


def main() -> int:
    base_url = os.environ.get("WIKIPEDIA_API_BASE", _DEFAULT_BASE).rstrip("/")
    user_agent = os.environ.get("WIKIPEDIA_USER_AGENT", _DEFAULT_UA)
    headers = {"User-Agent": user_agent}
    timeout = float(os.environ.get("WIKIPEDIA_FETCH_TIMEOUT_S", "30"))

    summary_url = f"{base_url}/page/summary/{quote(_ARTICLE_TITLE, safe='')}"

    with httpx.Client(timeout=timeout, headers=headers) as client:
        print(f"GET {summary_url}", file=sys.stderr)
        summary_response = client.get(summary_url)
        summary_response.raise_for_status()
        summary = summary_response.json()

        canonical_title = (
            summary.get("titles", {}).get("canonical")
            or summary.get("title")
            or _ARTICLE_TITLE
        )
        html_url = f"{base_url}/page/html/{quote(canonical_title, safe='')}"

        print(f"GET {html_url}", file=sys.stderr)
        html_response = client.get(html_url)
        html_response.raise_for_status()
        html = html_response.text

    envelope = {"summary": summary, "html": html}
    _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FIXTURE_PATH.write_text(
        json.dumps(envelope, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    section_count = html.lower().count("<h2")
    title = (
        summary.get("titles", {}).get("display")
        or summary.get("title")
        or _ARTICLE_TITLE
    )
    print(f"wrote {_FIXTURE_PATH}")
    print(f"  title:         {title}")
    print(f"  section count: {section_count} (<h2> occurrences)")
    print(f"  html bytes:    {len(html):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
