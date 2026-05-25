# REQUIREMENTS.md

My interpretation of the take-home brief. This document is the contract I'm building against. Anything not stated here is out of scope; anything ambiguous is captured in §10–12 with its resolution path.

---

## 1. Purpose

Build a small, containerised web app that lets a user paste a Wikipedia URL, get a locally-generated summary, and then chat with the article using RAG against a local LLM. The product is small on purpose — the work being evaluated is the planning, the AI-collaboration trail, the test discipline, and the cleanliness of the stack, not feature breadth.

## 2. User Flow (the only flow)

1. User opens the app, sees a URL input.
2. User pastes a Wikipedia article URL and submits.
3. App fetches and parses the article (body text, section structure).
4. App generates a concise summary via the local LLM and displays it.
5. App chunks the article, embeds the chunks, stores them in a vector DB (replacing any prior article's data).
6. A chat box appears below the summary. User asks a question, app retrieves top-K chunks, passes them to the local LLM with a grounded prompt, and renders the answer plus a structured list of source sections.
7. Repeat step 6. Pasting a new URL restarts at step 3 and wipes the previous article's index. Page refresh discards the active article (no persistence).

## 3. In Scope

- Single-page web UI: URL input, summary panel, chat panel.
- Wikipedia article ingestion (English Wikipedia only).
- Section-aware chunking, in-process embedding, vector store insertion.
- LLM-generated summary on ingest.
- RAG-grounded Q&A with structured source citations in the response payload.
- Single-article session; new URL replaces it; refresh loses it.
- Single-turn chat: each question answered independently. The UI may render a transcript of prior Q&A purely for user reference; the backend is stateless.
- Local LLM via Ollama, containerised by default with a host-side fallback documented.
- Vector DB as a service in `docker compose`, ephemeral (no named volume).
- One-command bring-up: `docker compose up`.
- Test suite with ≥85% line coverage and one integration test against the wired-up stack.
- Three planning artefacts at repo root: `REQUIREMENTS.md`, `DESIGN.md`, `TASKS.md`.
- `README.md` with prerequisites and run instructions; `.env.example` for config.
- Short screen recording or screenshots of end-to-end use.

## 4. Out of Scope

- Authentication, user accounts, sessions, multi-user concurrency tuning.
- Multi-article history, search across articles, article library UI.
- Conversation memory in the prompt (each chat turn is independent on the backend).
- Streaming token output.
- Non-Wikipedia URLs.
- Non-English Wikipedia (`*.wikipedia.org` for other languages).
- URL caching, summary caching, persistent embeddings across sessions.
- Hosted inference at runtime.
- Analytics, telemetry, rate limiting beyond framework defaults.
- Visual polish beyond "functional and unembarrassing".
- Mobile-specific layout work.
- Production hardening (HTTPS, secrets manager, observability stack).

## 5. Functional Requirements

| ID  | Requirement |
| --- | --- |
| F1  | User can submit a Wikipedia URL via a single text input on the main page. |
| F2  | URL validation accepts `https?://(en\.)(m\.)?wikipedia\.org/wiki/<Article_Title>`. `en.m.wikipedia.org` is normalised to `en.wikipedia.org`. URL fragments (`#History`) are stripped. URLs in `Special:`, `Talk:`, `Category:`, `File:`, `Help:`, `User:`, `Portal:` namespaces, `?curid=` style URLs, and non-English Wikipedia hosts (`fr.wikipedia.org`, etc.) are rejected at validation with a clear user-facing message. |
| F3  | The app fetches the article via the Wikipedia REST API and extracts the main body text and section structure. The "References", "External links", "Further reading", "See also", and "Notes" sections are stripped before chunking. |
| F4  | The app generates a 4–8 sentence summary via the local LLM and renders it in the UI. For articles whose cleaned body exceeds the model's context budget, the summarisation strategy is a DESIGN.md concern (see A4 and §12 Challenge 3). |
| F5  | The app chunks the article (section-aware; see A10), embeds each chunk in-process with sentence-transformers, and stores them in Qdrant with metadata: `section_title`, `section_index`, `chunk_index_in_section`, `char_offset_start`, `char_offset_end`, `text`. |
| F6  | Submitting a new URL drops and recreates the active Qdrant collection before ingest. |
| F7  | The chat endpoint embeds the question, retrieves top-K chunks, and assembles a prompt that instructs the LLM to answer **only** from the provided chunks and to reply "not found in the article" when the chunks don't contain the answer. Grounding is enforced through prompt design and verified with a test that asks a known-out-of-corpus question and asserts the model declines to answer. We do not claim the model never leaks parametric knowledge; we claim it is instructed not to and that the instruction is tested. |
| F8  | The chat response payload has shape `{answer: string, sources: [{section_title, chunk_index, score, excerpt}]}`. The frontend renders the answer with the source list alongside (as chips, footnotes, or a collapsible — exact UI is a DESIGN concern). Citations come from the retriever output, not from the LLM, so they cannot be hallucinated. |
| F9  | Clear user-facing errors for: malformed URL, non-Wikipedia URL, non-English wiki, namespace URL (Special/Talk/etc.), 404 from Wikipedia, disambiguation page, empty or very short article (< 500 chars body), Ollama unreachable, vector DB unreachable, retrieval returning zero chunks. |
| F10 | Latency targets — summary < 60s, chat answer < 30s on a developer laptop — apply **once Ollama has loaded the model into memory**. The first request after `docker compose up` may exceed these budgets due to cold start. The backend issues a best-effort warmup ping to Ollama on application startup (see A16). Cold-start behaviour is documented in `README.md`. |

## 6. Non-Functional Requirements

| ID   | Requirement |
| ---- | --- |
| NF1  | **One-command bring-up.** `docker compose up` brings the full stack online: FastAPI app, Qdrant, Ollama (default), or app + Qdrant with documented host-side Ollama fallback. |
| NF2  | **No secrets in the repo.** `.env.example` enumerates configurable values; real `.env` is gitignored. |
| NF3  | **No hosted LLM calls at runtime.** Hosted models may be used by AI agents in the IDE during development only. |
| NF4  | **Three swappable abstractions: LLM client, vector store, embedder.** Each behind an interface so a different provider could be wired in without touching call sites. Justification in DESIGN.md. |
| NF5  | **Test coverage ≥ 85% line coverage** on application code. Coverage report committed. Generated boilerplate, entrypoint, and framework glue excluded — exclusions documented in `.coveragerc` / `pyproject.toml` and called out in DESIGN. |
| NF6  | **Meaningful tests.** Unit tests mock the LLM, vector DB, and embedder. At least one integration test runs the real wired-up stack end-to-end against a cached fixture article. |
| NF7  | **Clear separation of concerns.** Scraper, chunker, embedder, retriever, generator, RAG orchestrator, and HTTP layer are distinct modules. |
| NF8  | **Reasonable image sizes and clean compose.** No unused services, no stale build context bloat. |
| NF9  | **Honest documentation.** README states prerequisites, known limitations, cold-start behaviour, and how to swap the LLM, embedder, and vector DB. Trade-offs called out in DESIGN. |
| NF10 | **Reproducibility.** A reviewer who has Docker installed can clone and run; no hidden host dependencies beyond what's in README. |

## 7. Constraints (non-negotiable, from the brief)

- Planning artefacts REQUIREMENTS.md, DESIGN.md, TASKS.md at repo root.
- No hosted LLM calls in the running app.
- Vector DB runs as a compose service, not in-process.
- Single command brings up the full stack.
- No secrets committed.
- ≥85% line coverage with meaningful tests; coverage report committed.
- Public git repo submission with README, docker-compose, recording/screenshots.

## 8. Confirmed Decisions (locked)

These are the decisions you confirmed (or explicitly delegated to me). They are repeated here so the doc stands alone.

| Decision | Value |
| --- | --- |
| Backend | Python / FastAPI |
| Frontend | React / Vite |
| LLM runtime | Ollama, containerised by default; host-side documented as fallback |
| LLM model | `llama3.2:3b` |
| Embedding model | Local sentence-transformers, **`bge-small-en-v1.5`** *(delegated by user; see §12 Challenge 1 for the case for MiniLM)* |
| Embedding execution | In-process (batched), not via Ollama |
| Vector DB | Qdrant, containerised, **ephemeral** (no named volume) |
| Chunking | Section-aware; sections under ceiling stay whole; longer sections sub-chunked at 512 tokens with 64-token overlap, respecting paragraph boundaries |
| Section ceiling | 1000 tokens |
| Retrieval top-K | 4 |
| Similarity metric | Cosine |
| URL handling | Replace semantics, one article active at a time. No caching, no history. |
| Chat | Single-turn, stateless backend. UI may show prior Q&A as transcript for user reference. |
| Scraping | Wikipedia REST API, English Wikipedia only. |
| Session | Single-session, refresh loses the active article. |
| Bad input | Validate URL pattern; fail clearly on disambiguation pages and non-English wikis. |

## 9. Assumptions (proceed unless overridden)

| ID  | Assumption |
| --- | --- |
| A1  | "Wikipedia article" means English Wikipedia (`en.wikipedia.org`, with `en.m.wikipedia.org` normalised). Other languages and special namespaces are rejected at validation. |
| A2  | **Vector DB is ephemeral.** No named volume. On `docker compose down` or container restart, all data is lost. The UI shows the empty state on next start. Consistent with the single-article scope and the "refresh loses the article" decision. |
| A3  | Chat answers are prose. Source attribution is delivered as a structured `sources` array in the response payload (F8), not embedded in the prose. |
| A4  | A summary is a 4–8 sentence paragraph generated by a **single** LLM call in all cases. The input to that call depends on article length: **(a)** short articles (cleaned body ≤ ~5K tokens): full body text; **(b)** long articles (cleaned body > ~5K tokens): a **structured outline** consisting of the lede paragraph, every section heading, and the first ~30 tokens (or first sentence) of each section. This keeps summarisation latency bounded by F10 regardless of article length while preserving full-article coverage. Outline extraction reuses the section-aware parser already required by A10 — no duplicated logic. The ~5K threshold and the per-section snippet size are tunable via env. See §12 Challenge 3 (resolved) for the comparison that drove this. |
| A5  | The summary is regenerated on every fresh ingest. Not cached. |
| A6  | A single user uses the app at a time. No multi-tenancy, no per-session isolation in the vector store. |
| A7  | The vector DB uses a single collection named `active_article`. Ingest drops and recreates it. Given A2 (ephemeral) and the no-caching decision, no URL-hashing or per-article dedup is needed. |
| A8  | The integration test uses real Qdrant and real Ollama (the wired-up stack), but runs against a cached fixture HTML for the Wikipedia article so it is offline-deterministic. The fixture is `Photosynthesis` (long, stable, well-sectioned, public domain). |
| A9  | Article length is capped at ~200k chars of cleaned body text (~50k tokens). Articles longer than this are truncated, with a flag surfaced in the UI. We do not reject long articles outright (see U13). |
| A10 | **Chunking is section-aware.** Sections under the 1000-token ceiling stay whole. Sections above the ceiling are sub-chunked at ~512 tokens with ~64-token overlap, respecting paragraph boundaries. Each chunk carries section metadata for citation (F5). All thresholds (ceiling, sub-size, overlap) configurable via env. |
| A11 | Retrieval defaults: top-K = 4, cosine similarity. Configurable via env. |
| A12 | If the chat endpoint is called before any article is ingested, it returns a clear "ingest an article first" response rather than erroring. |
| A13 | The Wikipedia fetch path uses a polite identifying user-agent and a single retry with exponential backoff on transient failure. No aggressive scraping. |
| A14 | The frontend is a single page; no client-side router. State is in React component state and lost on refresh. |
| A15 | Coverage report is generated as `coverage.xml` plus an HTML report. The HTML report is committed as a folder (or a screenshot of the summary line, per the brief's allowance). Exclusions documented in `.coveragerc`. |
| A16 | **Ollama warmup ping** on app startup: the backend makes a best-effort no-op generation request to Ollama once at boot to load the model into memory before the user's first request. Failure does not block startup; it logs a warning. |
| A17 | The chat response includes a structured `sources` array even when the LLM says "not found in the article" — the sources reflect what was retrieved, regardless of whether the LLM used them. Useful for debugging and demo transparency. |

## 10. Open Questions — Resolution Log

Audit trail for the OQs in the original draft. Each is now resolved.

| ID | Question | Resolution |
| --- | --- | --- |
| OQ-1 | Local LLM model | `llama3.2:3b` (user-confirmed). |
| OQ-2 | Vector DB | Qdrant (user-confirmed). |
| OQ-3 | Embedding model | **Reversed during review.** Originally recommended `nomic-embed-text` via Ollama. Flipped to in-process sentence-transformers because (i) batched in-process embedding is ~10× faster for a single-article workload; (ii) decouples ingest from LLM runtime; (iii) gives a third swappable abstraction in DESIGN; (iv) image-size impact is negligible (~500 MB CPU-only). Within sentence-transformers, chose `bge-small-en-v1.5` over `all-MiniLM-L6-v2` for the higher MTEB retrieval score at minimal extra cost (see §12 Challenge 1). |
| OQ-4 | Scraping strategy | Wikipedia REST API (user-confirmed). |
| OQ-5 | Citation style | **Reversed during review.** Originally inline section tags in prose. Flipped to structured `sources` array in response payload because (i) local 3B models hallucinate inline tags; (ii) structured sources are deterministically testable; (iii) frontend can render them cleanly. The LLM is not asked to emit citation tags. |
| OQ-6 | References / external links | Dropped before chunking (low retrieval signal). |
| OQ-7 | Streaming vs batch | Batch responses. |
| OQ-8 | Disambiguation / redirects | Follow redirects silently; reject disambiguation pages with a clear error. |
| OQ-9 | Empty / very short article | Refuse to ingest below 500 chars cleaned body; show error. |
| OQ-10 | Coverage exclusions | Entrypoint, framework glue, generated code excluded; documented in `.coveragerc`. |
| OQ-11 | Integration test scope | One full-stack test against a cached `Photosynthesis` fixture, real Qdrant, real Ollama. Offline-deterministic. |
| OQ-12 | Where the summary lives | **Revised during review.** Originally persisted on backend keyed by URL hash. With the user-confirmed "refresh loses the article" decision plus A2 (ephemeral DB), the summary lives in frontend state during the session only. No backend persistence. |

## 11. Open Questions Resolved Unilaterally During Drafting

Decisions I made while drafting that you did not explicitly address. Each is a candidate for override.

| ID  | Decision | Reasoning |
| --- | --- | --- |
| U1  | Section ceiling: 1000 tokens | Middle ground: sections below stay whole and retrieve cleanly; sections above sub-chunk with overlap. Tunable via env. |
| U2  | Article length cap: ~200k chars / ~50k tokens | Bounds ingest time and memory. Surfaces a UI flag if truncated. |
| U3  | Integration test fixture: `Photosynthesis` | Long, stable, well-sectioned, public domain. Cached as HTML in the repo for offline determinism. |
| U4  | Warmup ping to Ollama on app startup | Best-effort, non-blocking. Cold-start documented in README. Prevents the demo recording from looking broken on first request. |
| U5  | URL validation accepts `en.wikipedia.org` and `en.m.wikipedia.org` (latter normalised) | Mobile URLs are common copy-paste artefacts; rejecting them is hostile UX. URL fragments are stripped. |
| U6  | Polite identifying user-agent on Wikipedia fetch with single retry / backoff | Wikipedia's usage policy expects an identifying UA; one retry handles transient hiccups. |
| U7  | Single collection named `active_article` (no URL hashing) | Simpler than per-URL hashed collections, given ephemeral DB + no-caching scope. |
| U8  | `sources` array returned even on "not found in article" answers | Reflects what was retrieved, regardless of LLM disposition. Useful for debugging and demo transparency. |
| U9  | Coverage report committed as HTML report folder (or screenshot of summary line) | Brief explicitly allows either. HTML is more reviewable; screenshot is lower-friction. |
| U10 | Frontend has no client-side router | Single-page scope makes a router unjustified complexity. |
| U11 | Configurable thresholds (chunking, retrieval, length cap) live in env, not in YAML or a config service | One mechanism, documented in `.env.example`. |
| U12 | Stripped sections: "References", "External links", "Further reading", "See also", "Notes" | All low-signal for grounded Q&A; including them adds noise to retrieval. |
| U13 | Truncate-on-cap rather than reject for >200k char articles | Producing something on long articles is better UX than a hard reject. Surfaces a flag in the UI. |
| U14 | Each chunk's stored metadata includes the chunk text itself | Required to populate the `sources[].excerpt` field in F8 without a second lookup. |
| U15 | Disambiguation pages are detected via the Wikipedia REST API's page metadata, not by text heuristics | More reliable than parsing the page body for telltale strings. |

## 12. Top 3 Decisions to Challenge Before Moving to DESIGN.md

The three forks in the road where a second opinion would change the outcome.

### Challenge 1 — Embedding model: `bge-small-en-v1.5` vs `all-MiniLM-L6-v2`

I picked `bge-small-en-v1.5` (33 M params) over `all-MiniLM-L6-v2` (22 M params) because its MTEB retrieval score is ~10 points higher. For RAG, retrieval quality is the bottleneck, and the embedder is the cheapest place to lift it.

**Counter-argument:** MiniLM is the canonical "small RAG" embedder — years of production telemetry, faster on CPU, smaller download for the demo image. For Wikipedia-style Q&A where questions often map cleanly to section text, MiniLM may close most of the MTEB gap in practice. The seconds saved on embedding may matter more in the demo than a marginal retrieval bump on benchmarks.

**My recommendation:** stay with `bge-small-en-v1.5`, but make the embedder a one-env-var swap in DESIGN.md so we can A/B test if the demo answers feel weak.

### Challenge 2 — Top-K = 4

You confirmed K = 4. On a 3 B model with an 8 K context window, K = 5 or 6 chunks of ~600 tokens each would fit comfortably and improve recall for multi-hop questions ("How did X influence Y?"). K = 4 prioritises prompt cleanliness over recall.

**Counter-argument for staying:** smaller K means each retrieved chunk gets more attention in the LLM's prompt, and on a 3 B model attention dilution is a real failure mode. Tighter retrieval can outperform broader retrieval on grounded QA.

**My recommendation:** keep K = 4, but make it env-tunable (per A11) and try K = 6 in dev before recording the demo to see if it qualitatively improves answers. The decision can flip without a code change.

### Challenge 3 — Long-article summarisation strategy *(Resolved during review)*

**Original recommendation:** hierarchical map-reduce for long articles, leading-N for short ones.

**Your pushback:** hierarchical's latency cost is unacceptable on `llama3.2:3b` given F10's 60s budget. You proposed a third option: a single structured-prompt call composed of lede + section headings + first sentence of each section.

**Three-way comparison** (for a representative long Wikipedia article: ~10 K tokens cleaned body, ~12 sections, on `llama3.2:3b` warm):

| Strategy | LLM calls | Input tokens / call | Total input | Coverage | Estimated warm latency | F10 fit |
| --- | --- | --- | --- | --- | --- | --- |
| **Leading-N (~6 K)** | 1 | ~6,000 | ~6,000 | First ~60% of article | ~25–45 s (dominated by prefill) | Borderline; risks the 60 s cap |
| **Hierarchical (map-reduce)** | ~13 (12 sections + 1 merge) | ~500–1,000 | ~16,000 | Full | ~80–145 s | Violates F10 |
| **Structured outline** | 1 | ~1,000–1,500 (lede + headings + topic sentences) | ~1,200 | Full (low resolution) | ~10–20 s | Comfortable; leaves headroom for cold-start |

**Quality analysis.** Hierarchical wins on raw quality but is eliminated by F10. Between leading-N and structured outline:

- Leading-N's coverage is biased toward the lede paragraph, which on Wikipedia is *already* a summary. The LLM ends up paraphrasing an existing summary while missing late-article sections like "Legacy" or "Reception". The output reads competent but obvious.
- Structured outline exploits two Wikipedia editorial conventions that are not accidental: `MOS:LEAD` requires the lede to summarise the article, and `MOS:SECTIONS` requires the first sentence of each section to be a topical opener. Feeding the model these two signals gives full-article coverage at one-fifth the input tokens of leading-N. On Wikipedia specifically, this is a structural fit that leading-N can't match.

**Winner: structured outline.** It wins on latency unconditionally and on coverage versus leading-N; it loses on raw quality to hierarchical but hierarchical is off the table on F10 grounds. The strategy degrades gracefully for stub articles (no section headings → behaves like leading-N over a small body).

**Resolution committed in A4.** F10's summary budget is now realistic — the per-call input is bounded by design at ~1.5 K tokens regardless of article length.

## 13. Definition of Done

- All hard constraints in §7 are satisfied.
- Functional requirements F1–F10 demonstrably work in the recording / screenshots.
- Non-functional requirements NF1–NF10 hold.
- Coverage report shows ≥ 85% with exclusions documented.
- REQUIREMENTS.md, DESIGN.md, TASKS.md present, internally consistent, and reflect what was actually built.
- README runs cleanly from a fresh clone on a machine with Docker installed.
- No hosted LLM calls in the running app.
- `NOTES.md` (optional) flags anything the AI got wrong that I had to correct, and what I would change with another two days.

---

*This document is frozen once Challenges 1 and 2 in §12 are resolved (Challenge 3 was resolved during review). Changes after that point go into a changelog at the bottom of DESIGN.md, not silently here.*
