# Footnote

> *Wikipedia, with sources.* A small, containerised web app for chatting with a Wikipedia article. Paste a URL, get a locally-generated summary, then ask the article questions — answers are grounded against the article's own content and cite the sections they came from.

Everything in the running app stays local: the LLM (`llama3.2:3b` via Ollama), the embedder (`bge-small-en-v1.5` via sentence-transformers, in-process), and the vector store (Qdrant). No hosted inference at runtime.

*("Footnote" is the product name; `silver-spoon2` is the repo slug.)*

The full design rationale lives in [`REQUIREMENTS.md`](REQUIREMENTS.md) → [`DESIGN.md`](DESIGN.md) → [`TASKS.md`](TASKS.md), in that read order. [`NOTES.md`](NOTES.md) tracks the incidents and future-work items that surfaced during the build.

## Demo

Screenshots of the end-to-end flow (URL paste → summary → grounded chat → F7 out-of-corpus decline) live under [`docs/screenshots/`](docs/screenshots/).

## Prerequisites

- Docker (Desktop on macOS / Windows, or engine + compose plugin on Linux). Tested against Docker 24+.
- ~6 GB free RAM at peak (per DESIGN §6.3 — app + Qdrant + Ollama all loaded).
- ~4 GB free disk (multi-stage app image + Qdrant image + cached `llama3.2:3b` model).
- For **host-fallback mode** only: `ollama serve` running on the host plus the `llama3.2:3b` model pulled.

No host Python, Node, or `uv` install is required to run the app — those are only needed to run the test suite (see Testing).

## Quickstart

Two invocations, picked at compose-time via a profile flag. No editing of `docker-compose.yml` to switch between them.

### Containerised-Ollama (default)

```sh
docker compose --profile containerised-ollama up --build
```

First run downloads `llama3.2:3b` into the `ollama_models` named volume (~2 GB, one-time). The `app` service waits for the `ollama-pull` init service to complete, then runs its warmup ping inside the lifespan before reporting healthy. Open `http://localhost:8000`.

### Host-fallback (host-side Ollama)

If you already run `ollama serve` on the host and have `llama3.2:3b` pulled:

```sh
# In .env:
OLLAMA_BASE_URL=http://host.docker.internal:11434

docker compose up --build      # no --profile flag
```

Only `app` and `qdrant` come up; the Ollama services are gated out by the inactive `containerised-ollama` profile, and the `depends_on: ollama-pull` declaration carries `required: false` so the missing service doesn't block startup.

### Shutting down

`docker compose --profile containerised-ollama down` (or plain `down` for host-fallback). Qdrant data is **ephemeral** per A2 — no named volume, so stopping the stack wipes the active article. The pulled model survives in the `ollama_models` volume across restarts.

## Testing

Two invocations: unit suite (the coverage gate) and integration suite (assumes the stack is up).

### Unit suite + coverage (the canonical command)

```sh
uv run pytest -m "not integration" --cov=backend/app --cov-report=term --cov-report=html
```

- 238 tests, target ≥ 85% line coverage on application code (currently 95.66%).
- `[tool.coverage.report] fail_under = 85` in `pyproject.toml` causes a non-zero exit if it ever drops.
- HTML report regenerates into [`coverage_report/`](coverage_report/) (committed per A15). Open `coverage_report/index.html` in a browser.
- Coverage exclusions are documented in `pyproject.toml`'s `[tool.coverage.run] omit` list and in DESIGN §7.4: `main.py`, `config.py`, `domain/models.py`, and every package `__init__.py`. Two files sit notably below average and are intentional — `chunking/tokenizer.py` (27%, real SBERT tokenizer; unit tests use a fake per DESIGN §3) and `api/deps.py` (72%, FastAPI provider glue; lifespan-state branches are exercised by `main.py` and the integration suite, both excluded from coverage).

### Integration suite (run separately; not in coverage)

```sh
docker compose --profile containerised-ollama up -d
uv run pytest tests/integration -v
```

- **`test_embedder_smoke.py`** — loads the real `bge-small-en-v1.5` model and confirms vector size, distinctness, and unit magnitude. First run downloads ~133 MB into `~/.cache/huggingface/`.
- **`test_full_stack.py`** — over-the-network ingest → in-corpus chat → F7 out-of-corpus grounding decline against the live containerised app. Hits live Wikipedia for the fetch hop; rationale and the alternative (host stub + `WIKIPEDIA_API_BASE` override) are documented in the test docstring.

The integration suite **is not part of the coverage run** by design — it needs the compose stack up, would inflate the headline number with end-to-end paths that aren't really being asserted line-by-line, and is slow. Both purposes deserve their own command.

## Configuration

Every knob is an environment variable. [`.env.example`](.env.example) enumerates them with their defaults (matching DESIGN §6.4 verbatim). Copy to `.env` and edit. The compose stack mounts `.env` with `required: false`, so it's optional — pydantic-settings holds the same defaults in code.

The settings worth knowing about up front:

| Variable | Default | What it controls |
| --- | --- | --- |
| `LLM_MODEL` | `llama3.2:3b` | Ollama model tag pulled by the init service and used by all LLM calls. |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Override to `http://host.docker.internal:11434` for host-fallback. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | sentence-transformers model id; vector size is auto-detected. |
| `RETRIEVAL_TOP_K` | `4` | Top-K chunks the chat path retrieves. |
| `CHUNK_SECTION_CEILING_TOKENS` | `1000` | Sections at or below this stay whole; longer sections are sub-chunked. |
| `ARTICLE_CHAR_CAP` | `200000` | Long articles are truncated at the nearest section boundary; UI shows a flag (U13). |
| `SUMMARY_LONG_THRESHOLD_TOKENS` | `3500` | Above this, the summariser switches to the structured-outline path (A4). |
| `WARMUP_TIMEOUT_S` | `30` | Lifespan awaits the warmup ping with this bound; on timeout, app still serves with `warmup_ok=false`. |

See `.env.example` for the full list and DESIGN §6.4 for the per-knob rationale.

## Swapping the LLM, embedder, or vector DB

The three NF4 abstractions all sit behind Python `Protocol`s and are one-env-var swaps:

- **LLM model.** Change `LLM_MODEL` (e.g. `mistral:7b`, `qwen2.5:3b`). The `ollama-pull` init service picks up the new tag and pulls it on next compose-up. For a completely different LLM provider, implement `LLMClient` (see [`backend/app/llm/base.py`](backend/app/llm/base.py)) and wire it in [`backend/app/api/deps.py`](backend/app/api/deps.py).
- **Embedder.** Change `EMBEDDING_MODEL` (e.g. `sentence-transformers/all-MiniLM-L6-v2`). Vector size is detected from the model — Qdrant collection geometry follows automatically. For a non-SBERT embedder, implement `Embedder` (see [`backend/app/embedding/base.py`](backend/app/embedding/base.py)).
- **Vector DB.** Change `QDRANT_URL` to point at another Qdrant instance. For a different store entirely, implement `VectorStore` (see [`backend/app/vectorstore/base.py`](backend/app/vectorstore/base.py)) and wire it in `deps.py`.

DESIGN §12 Challenge 1 covers the `bge-small` vs `MiniLM` trade-off for the embedder; §12 Challenge 2 covers tuning `RETRIEVAL_TOP_K`.

## Cold-start behaviour

F10 commits to `<60s` summary and `<30s` chat — **after** Ollama has loaded the model into memory. The first request after compose-up always pays a cold-start tax dominated by Ollama loading `llama3.2:3b` weights.

The backend mitigates this with a startup **warmup ping** (A16, DESIGN §4.2): the lifespan issues a no-op `generate("ping", max_tokens=1)` and awaits it with a `WARMUP_TIMEOUT_S=30` bound before the app reports healthy. On timeout or error, the app still serves traffic with `warmup_ok=false` — degrade explicitly rather than fail silently.

`/api/health` surfaces `warmup_ok` to the frontend, which renders a "first request after startup is slower while the model loads" sub-label under the spinner when `warmup_ok` is false. Once the warmup has succeeded, the sub-label disappears.

## Maintaining fixtures

[`scripts/regenerate_fixture.py`](scripts/regenerate_fixture.py) fetches live Wikipedia and writes [`tests/fixtures/photosynthesis_rest_api.json`](tests/fixtures/photosynthesis_rest_api.json):

```sh
python scripts/regenerate_fixture.py
```

The script prints the article's display title, `<h2>` count, and HTML byte size so you can eyeball that the right thing came back. **Run by hand only** — it hits live Wikipedia and is non-idempotent across time (article content drifts).

When to regenerate: if the integration test's chat assertion breaks because Wikipedia renamed or restructured the *Photosynthesis* article (e.g. no section title contains "Light" or "Pigment" anymore), the fix is to regenerate the fixture and update the assertion to the new section names — *not* to weaken the assertion. The fixture is committed as the wire-format reference; the live integration test does not currently consume it (see `tests/integration/test_full_stack.py`'s docstring for the rationale).

## Known limitations

- **Single article at a time.** Pasting a new URL drops the prior article's vector store collection. There is no article history, no cross-article search, no "back to previous article" affordance.
- **Refresh wipes state.** Frontend state is in `App.tsx` component state only (A14). Reloading the page loses the active article and chat transcript; you have to re-ingest.
- **Qdrant is ephemeral.** No named volume per A2. `docker compose down` (or any container restart) wipes the index.
- **Single user, single concurrent request assumed.** No multi-tenant isolation, no request-level concurrency tuning beyond what FastAPI gives us for free.
- **English Wikipedia only.** Non-English wikis (`fr.wikipedia.org`, etc.) are rejected at URL validation with a clear user-facing message.
- **The chat backend is stateless.** Each question is answered independently against retrieved chunks. The transcript shown in the UI is purely client-side for user reference; the backend does not look at prior turns.
- **F7 grounding is best-effort.** The chat prompt instructs the LLM to answer only from retrieved chunks and to say "not found in the article" otherwise. The integration suite tests this with an out-of-corpus question that's canonical training data ("Who wrote the play Hamlet?") — we claim the model is instructed not to leak parametric knowledge and that the instruction is tested, not that the model never leaks.
- **Wikipedia REST endpoint deprecation.** The fetcher uses `/page/summary/` + `/page/html/`. Wikimedia retired `/page/mobile-sections/` mid-build; both replacements are stable for now, but `/page/html/` will eventually move to the MediaWiki Core REST API. See [`NOTES.md`](NOTES.md).

## Troubleshooting

- **`docker compose up` hangs at "ollama-pull"** — first run of the containerised-Ollama profile pulls ~2 GB. Give it ~3–5 minutes on a typical home connection. Subsequent ups skip the download via the `ollama_models` named volume.
- **`/api/health` returns `warmup_ok: false`** — the lifespan warmup didn't complete in `WARMUP_TIMEOUT_S=30` seconds. The app still serves; the next request will pay the cold-start tax. Common on a cold-CPU laptop where `llama3.2:3b` takes 30+ s to load.
- **Chat returns `503 llm_unavailable`** — Ollama isn't reachable. In containerised mode, check `docker compose ps` (the `ollama` service should be `healthy`). In host-fallback mode, check `ollama serve` is running and `OLLAMA_BASE_URL` is `http://host.docker.internal:11434`.
- **Chat returns `409 no_active_article`** — ingest a URL first. The chat endpoint guards on the Qdrant collection's existence (A12).
- **Chat returns `422 retrieval_empty`** — the question vectorised to a region of space where no chunks live. Rephrase, or check that the article was actually ingested (chunk count in the ingest response should be > 15).
- **First ingest after compose-up times out** — likely the warmup didn't complete and the first user request is paying the full cold-start tax inside the chat/ingest httpx timeout (`OLLAMA_TIMEOUT_S=120`). Wait for `/api/health` to show `warmup_ok: true` before submitting.
- **Port 8000 / 6333 / 11434 already in use** — adjust the host-side port in `docker-compose.yml` (only `8000:8000` is published by default; Qdrant and Ollama bind to the compose network only).

## Coverage report

[`coverage_report/index.html`](coverage_report/index.html) — last generated against `uv run pytest -m "not integration" --cov=backend/app --cov-report=html`. Current total: **95.66%** (715 statements, 31 missed, 238 tests passed). Exclusions are documented in DESIGN §7.4 and `pyproject.toml`.

## Project layout

```
backend/app/           FastAPI + RAG pipeline (DESIGN §1 components)
frontend/              React/Vite SPA (DESIGN §4.8)
scripts/               Fixture regenerator
tests/unit/            Hermetic unit suite (no model loads, no network)
tests/integration/     Real-stack smoke + real-SBERT smoke (marked `integration`)
tests/fixtures/        Cached REST API payloads
coverage_report/       Committed HTML coverage report (A15)
docs/screenshots/      Demo artefacts
REQUIREMENTS.md        The contract.
DESIGN.md              The interpretation.
TASKS.md               The build trail.
NOTES.md               Incidents + future work + AI/me delta.
```
