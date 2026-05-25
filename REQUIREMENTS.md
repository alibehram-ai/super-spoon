# REQUIREMENTS.md

My interpretation of the take-home brief. This document is the contract I'm building against. Anything not stated here is out of scope; anything ambiguous is captured under **Open Questions** with a recommendation that I'll proceed with unless overridden.

---

## 1. Purpose

Build a small, containerised web app that lets a user paste a Wikipedia URL, get a locally-generated summary, and then chat with the article using RAG against a local LLM. The product is small on purpose — the work being evaluated is the planning, the AI-collaboration trail, the test discipline, and the cleanliness of the stack, not feature breadth.

## 2. User Flow (the only flow)

1. User opens the app, sees a URL input.
2. User pastes a Wikipedia article URL and submits.
3. App fetches and parses the article (body text, section structure).
4. App generates a concise summary via the local LLM and displays it.
5. App chunks the article, embeds the chunks, stores them in a vector DB (replacing any prior article's data).
6. A chat box appears below the summary. User asks a question, app retrieves top-k chunks, passes them to the local LLM with a grounded prompt, and renders the answer.
7. Repeat step 6. Pasting a new URL restarts at step 3 and wipes the previous article's index.

## 3. In Scope

- Single-page web UI: URL input, summary panel, chat panel.
- Wikipedia article ingestion (English Wikipedia, standard article pages).
- Article chunking + embedding + vector store insertion.
- LLM-generated summary on ingest.
- RAG-grounded Q&A: retrieve from vector store, answer with local LLM only.
- Single-article session: one article active at a time; new URL replaces it.
- Single-turn chat: each question answered independently (no conversation memory).
- Local LLM via Ollama, containerised by default with a host-side fallback documented.
- Vector DB as a service in `docker compose`.
- One-command bring-up: `docker compose up`.
- Test suite with ≥85% line coverage and one integration test against the wired-up stack.
- Three planning artefacts at repo root: `REQUIREMENTS.md`, `DESIGN.md`, `TASKS.md`.
- `README.md` with prerequisites and run instructions; `.env.example` for config.
- Short screen recording or screenshots of end-to-end use.

## 4. Out of Scope

These are explicitly **not** being built. If I find myself drifting into any of them, I'll stop and revise this section.

- Authentication, user accounts, sessions.
- Multi-article history, search across articles, or article library UI.
- Conversation memory across questions (each chat turn is independent).
- Streaming token output (unless trivial — see Open Question OQ-7).
- Non-Wikipedia URLs (generic web scraping).
- Non-English Wikipedia (`*.wikipedia.org` for other languages).
- Hosted inference at runtime — no OpenAI/Anthropic/Gemini calls in the running app.
- Analytics, telemetry, feedback widgets, rate limiting beyond what the framework gives for free.
- Visual polish beyond "functional and unembarrassing".
- Mobile-specific layout work.
- Production hardening (HTTPS, secrets manager, prod-grade observability).
- Multi-user concurrency tuning beyond what a default FastAPI worker handles.

## 5. Functional Requirements

| ID  | Requirement                                                                                                                                   |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| F1  | User can submit a Wikipedia URL via a single text input on the main page.                                                                     |
| F2  | The app validates that the URL points to a Wikipedia article (host check + path check) before fetching.                                       |
| F3  | The app fetches the article and extracts the main body text and section structure. References are best-effort (see OQ-4).                     |
| F4  | The app generates a summary of the article via the local LLM and renders it in the UI.                                                        |
| F5  | The app chunks the article text, generates embeddings for each chunk, and stores them in the vector DB with metadata (section title, offset). |
| F6  | Submitting a new URL clears the vector DB collection for the previous article before ingesting the new one.                                   |
| F7  | The chat box accepts a question and returns an answer that is grounded in retrieved chunks (RAG), not the model's parametric knowledge.       |
| F8  | The chat answer surfaces which chunks/sections it drew from (citation strategy — see OQ-5).                                                   |
| F9  | The app returns a clear error to the user for: malformed URL, non-Wikipedia URL, 404 from Wikipedia, empty article, disambiguation page.      |
| F10 | The summary and chat both respond within a reasonable bound on a 3B-class model (target: summary < 60s, chat answer < 30s on a dev laptop).   |

## 6. Non-Functional Requirements

| ID   | Requirement                                                                                                                                                            |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NF1  | **One-command bring-up.** `docker compose up` brings the full stack online: app, vector DB, LLM runtime (default), or app + vector DB with documented host-side LLM.   |
| NF2  | **No secrets in the repo.** `.env.example` enumerates configurable values; real `.env` is gitignored.                                                                  |
| NF3  | **No hosted LLM calls at runtime.** Hosted models may be used by AI agents in the IDE during development only.                                                         |
| NF4  | **Swappable LLM and vector store.** Both sit behind interfaces so a different provider could be wired in without touching call sites. Justification belongs in DESIGN. |
| NF5  | **Test coverage ≥ 85% line coverage** on application code. Coverage report committed to repo. Generated boilerplate and entrypoint excluded — exclusions documented.   |
| NF6  | **Meaningful tests.** Unit tests mock the LLM and vector DB. At least one integration test runs the real wired-up stack end-to-end.                                    |
| NF7  | **Clear separation of concerns.** Scraping, chunking, embedding, retrieval, generation, and HTTP layer are distinct modules.                                           |
| NF8  | **Reasonable image sizes and clean compose.** No unused services, no stale build context bloat.                                                                        |
| NF9  | **Honest documentation.** README states prerequisites, known limitations, and how to swap the LLM model. Trade-offs called out in DESIGN.                              |
| NF10 | **Reproducibility.** A reviewer who has Docker installed can clone and run; no hidden host dependencies beyond what's in README.                                       |

## 7. Constraints (from the brief, non-negotiable)

- **Stack:** Python + FastAPI backend, React + Vite frontend. (Confirmed.)
- **LLM runtime:** Ollama, containerised by default; host-side Ollama documented as fallback for slow machines.
- **Vector DB:** runs as a compose service, not in-process.
- **Article handling:** replace on new URL; only one article active at a time.
- **Chat:** single-turn; no conversation memory in the prompt.
- **Embeddings:** local model preferred; hosted small embedding model acceptable if justified in DESIGN.
- **Planning artefacts:** REQUIREMENTS.md, DESIGN.md, TASKS.md at repo root.
- **Submission:** public git repo, README, docker-compose, coverage report, recording/screenshots.

## 8. Assumptions (proceed unless overridden)

These are the calls I'll make unilaterally if you don't push back. Each one is reversible early, expensive late — flag any that are wrong.

| ID  | Assumption                                                                                                                          |
| --- | ----------------------------------------------------------------------------------------------------------------------------------- |
| A1  | "Wikipedia article" means English Wikipedia (`en.wikipedia.org`) standard article pages. Special pages, talk pages, etc. are rejected at the URL-validation step. |
| A2  | The vector DB is allowed to persist between container restarts (named volume), but each new URL ingest still wipes its own collection. |
| A3  | The chat answers in plain prose, with optional inline citations to section names (see OQ-5).                                        |
| A4  | "Concise summary" is a 4–8 sentence paragraph, not a multi-section structured summary. Single LLM call.                              |
| A5  | The summary is regenerated on every fresh ingest; it is not cached separately from the article.                                      |
| A6  | A single user is using the app at a time. No multi-tenancy, no per-session isolation in the vector store.                            |
| A7  | The vector DB collection is named per-article (e.g., URL hash) so "replace" means drop-and-recreate one collection, not the whole DB. |
| A8  | The integration test is allowed to use a real but tiny model (or a stub model server) to keep CI feasible. Full Ollama in CI is not required. |
| A9  | Article length is capped (e.g., truncate at ~200k chars / ~50k tokens of raw text) to bound ingest time and memory. Cap is documented in README. |
| A10 | Reasonable defaults for chunking: ~800-token chunks with ~100-token overlap, section-aware. Tunable via env. (Detail in DESIGN.)     |
| A11 | Retrieval defaults: top-k=5, cosine similarity. Tunable via env.                                                                     |
| A12 | The app gracefully degrades if the vector DB is empty on chat (returns "ingest an article first") rather than erroring.              |
| A13 | The Wikipedia fetch path uses a polite user-agent string and respects a single retry on transient failure; no aggressive scraping.   |
| A14 | The frontend is a single page; no client-side routing library is required.                                                           |
| A15 | The coverage report format is `coverage.xml` + an HTML report committed as a folder or zipped artefact. Screenshot is acceptable.    |

## 9. Open Questions (decisions for you)

For each I've named 2–3 options, called out the trade-off, and made a recommendation. **The recommendation is my default if you don't reply** — flag the ones you want to override.

### OQ-1 — Specific local LLM model

- **Option A — `llama3.2:3b`** *(recommended)*: well-balanced, instruction-tuned, runs on a laptop. ~2GB pull.
- Option B — `qwen2.5:3b`: stronger reasoning on some benchmarks, similar size, slightly slower TTFT in my experience.
- Option C — `phi3:mini`: smallest footprint (~2.3GB), occasionally worse at grounded extraction.
- **Trade-off:** size vs answer quality on grounded Q&A. All three fit in the brief's "3B class" guidance.
- **My call if you don't reply:** A.

### OQ-2 — Vector database

- **Option A — Qdrant** *(recommended)*: easy compose service, good filtering, persistent volumes, mature Python client.
- Option B — Chroma: simpler, fewer moving parts, but its server mode is less battle-tested than Qdrant's.
- Option C — pgvector: nice if we already had Postgres in the stack; we don't, so it adds an unneeded service.
- **Trade-off:** Qdrant adds a small operational surface in exchange for first-class server semantics that make the integration test honest.
- **My call if you don't reply:** A.

### OQ-3 — Embedding model

- **Option A — `nomic-embed-text` via Ollama** *(recommended)*: keeps everything in one runtime, no extra Python deps for model weights, decent quality, persists in the same Ollama volume.
- Option B — `sentence-transformers/all-MiniLM-L6-v2` in-process: faster cold start, no Ollama round-trip per chunk, but adds PyTorch/Transformers to the app image (bigger image).
- Option C — Hosted small embedding model: brief allows it, but adds a runtime hosted dependency I'd rather avoid.
- **Trade-off:** image size and dep weight vs latency. Option A keeps the runtime topology tight.
- **My call if you don't reply:** A.

### OQ-4 — Scraping strategy

- **Option A — Wikipedia REST API (`/api/rest_v1/page/...`)** *(recommended)*: returns clean HTML/plain text and section structure. Stable, polite, no scraping fragility.
- Option B — `wikipedia` Python package: convenient, but wraps the same API and adds a dependency.
- Option C — Raw HTML scrape with BeautifulSoup: most flexible, least robust to layout changes.
- **Trade-off:** robustness vs flexibility. We don't need flexibility for this scope.
- **My call if you don't reply:** A.

### OQ-5 — Citation style in chat answers

- **Option A — Inline section names in brackets, e.g. "... according to [History] ..."** *(recommended)*: lightweight, readable, machine-checkable.
- Option B — Footnote-style numbered citations rendered below the answer with section name + offset.
- Option C — No citations, just the prose answer.
- **Trade-off:** auditability of grounding vs UI complexity. A is the cheapest visible signal that RAG is actually happening.
- **My call if you don't reply:** A.

### OQ-6 — How to handle references / external links in the article

- **Option A — Drop references and external-link sections entirely before chunking** *(recommended)*: they're mostly noise for grounded Q&A.
- Option B — Keep "References" as searchable chunks: lets the user ask "what sources back claim X?", but blows up the index with low-signal text.
- **Trade-off:** index quality vs answering meta-questions about sourcing.
- **My call if you don't reply:** A.

### OQ-7 — Streaming vs batch chat responses

- **Option A — Batch (single response after generation completes)** *(recommended)*: simpler API, simpler frontend, simpler tests. Acceptable on a 3B model.
- Option B — Streaming via SSE: nicer UX, but adds wire-format and frontend complexity that doesn't earn coverage points.
- **Trade-off:** UX polish vs scope discipline. The brief explicitly says don't pad.
- **My call if you don't reply:** A.

### OQ-8 — Disambiguation pages and redirects

- **Option A — Follow redirects silently, reject disambiguation pages with a clear user error** *(recommended)*: redirects are part of normal Wikipedia behaviour; disambiguation pages are ambiguous by definition.
- Option B — Follow both, treat disambiguation as an article: produces garbage summaries; bad UX.
- Option C — Reject both: too strict; redirects are common.
- **My call if you don't reply:** A.

### OQ-9 — Empty / very short article handling

- **Option A — Refuse to ingest if extracted body < threshold (e.g. 500 chars), show error** *(recommended)*: avoids meaningless summaries and an empty chat.
- Option B — Ingest anyway: technically works but produces a degraded experience.
- **My call if you don't reply:** A.

### OQ-10 — Coverage exclusions

- **Option A — Exclude `main.py` entrypoint, generated client code if any, and framework glue (e.g. `app.startup` hooks)** *(recommended)*: matches what the brief explicitly permits.
- Option B — Exclude only the entrypoint.
- **Trade-off:** stricter exclusions look more rigorous but make the 85% threshold genuinely harder.
- **My call if you don't reply:** A. Exclusions called out in `pyproject.toml`/`coveragerc` and in DESIGN.

### OQ-11 — Integration test scope

- **Option A — One full-stack test: start compose, ingest a known-stable Wikipedia URL (or a fixture), assert chat returns grounded answer** *(recommended)*. Heavy but high-signal.
- Option B — A "wired-up" test that uses real vector DB but a stub LLM: faster, but doesn't exercise the model.
- Option C — Both: ideal but probably over scope for 8–12h.
- **Trade-off:** signal vs CI cost. A satisfies the brief literally.
- **My call if you don't reply:** A, with the Wikipedia article cached as a fixture so the test is offline-deterministic.

### OQ-12 — Where the summary lives

- **Option A — Summary is stored on the backend keyed by article URL hash, served on GET** *(recommended)*: page refresh works without re-ingesting.
- Option B — Summary lives only in frontend state: refresh loses it.
- **My call if you don't reply:** A.

## 10. Definition of Done

- All hard requirements in §7 are satisfied.
- Functional requirements F1–F10 demonstrably work in the recording/screenshots.
- Non-functional requirements NF1–NF10 hold.
- Coverage report shows ≥ 85% with exclusions documented.
- REQUIREMENTS.md, DESIGN.md, TASKS.md present, internally consistent, and reflect what was actually built.
- README runs cleanly from a fresh clone on a machine with Docker installed.
- No hosted LLM calls in the running app.
- NOTES.md (optional) flags anything the AI got wrong and what I'd change with another two days.

---

*This document is frozen once we agree on the Open Questions. Changes after that point go into a changelog at the bottom of DESIGN.md, not silently here.*
