# NOTES.md

Scratch pad for things that didn't make it into REQUIREMENTS/DESIGN but matter for next-version thinking. Per REQUIREMENTS §13: optional file flagging AI mistakes that needed correction, and "what I would change with another two days." Living document.

---

## Future work

### ~~Migrate off `/page/mobile-sections/` to Parsoid-backed endpoints~~ — DONE mid-Phase-5

`WikipediaFetcher` originally used `https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title}` for section-structured article content. That endpoint was part of Wikimedia's legacy REST API and we expected to migrate "eventually". In practice Wikimedia retired the whole `mobile-sections*` family (lead, remaining, and combined) sooner than the deprecation notice suggested — discovered during T13's bare-metal smoke when ingesting `Photosynthesis` returned `article_not_found`. Probed alternatives: `/page/html/{title}` and `/page/mobile-html/{title}` both still serve Parsoid HTML; only the legacy mobile-sections family is gone.

Migration applied:
- Swapped the second-call endpoint from `/page/mobile-sections/{title}` to `/page/html/{title}`.
- Replaced the per-section JSON-text → `<p>` extractor with a single-pass Parsoid HTML parser (`_ParsoidHtmlParser`, stdlib `html.parser`). Sections split on any heading tag (`<h2>`..`<h6>`), matching the flattened section list the legacy API returned. `<p>` content within each section is collected; everything else (figures, tables, lists, edit-section spans) is dropped — same posture as the legacy parser.
- Kept `/page/summary/{title}` unchanged for disambiguation / redirect / 404 gating (U15 / OQ-8).
- Replaced `tests/fixtures/sample_mobile_sections.json` with `tests/fixtures/sample_parsoid.html` carrying the same logical structure (lede with `[1]` citation, "Overview" / "Details" / "References" sections); behavioural assertions in `test_wikipedia_fetcher.py` are unchanged.
- No new dependency added — stdlib `html.parser` handles Parsoid HTML well enough for our extraction (`<p>` text within heading-delimited sections).

T16's Photosynthesis fixture and regenerate script will land *after* this migration, so they pick up the new endpoint and parse shape for free — no follow-up needed there.

### Real per-phase ingest progress via Server-Sent Events

DESIGN §6.2 settled on a single indeterminate spinner because the alternatives were dishonest (wall-clock fake phase labels) or absent (no real progress channel). The honest version is an SSE endpoint that emits real events as the ingest pipeline advances:

```
event: fetch_done       data: { ms: 1200 }
event: clean_done       data: { ms: 200, truncated: false }
event: chunk_done       data: { ms: 80, chunk_count: 47 }
event: embed_done       data: { ms: 4100 }
event: upsert_done      data: { ms: 90 }
event: summary_started  data: { strategy: "outline" }
event: summary_done     data: { ms: 12000 }
event: complete         data: { ...full IngestResponse... }
```

Frontend opens an `EventSource` on `/api/ingest/stream`, updates the loading state on each event. Real progress for the user; no fake clocks. Costs a new endpoint, an `EventSource` client, and event-shape design — non-trivial but a clear UX win for the longest-running request in the app.

---

## Things the AI got wrong that I had to correct

Per REQUIREMENTS §13. Filled in as the project progresses.

### Phase 2 (T03–T06)

- **T04 — lede chunking gap.** Shipped the chunker matching DESIGN §3 literally (sections only); you caught that the lede would be unreachable in RAG and made me fix it as a follow-up commit (`0990057`). The fix prepends a synthetic `Section` for `CleanedArticle.lede` with `section_index=0` so the lede flows through the same chunking / embedding pipeline as everything else.
- **T04 — `sys.modules`-based lazy-import guard.** Brittle against T05 legitimately importing `sentence-transformers`; broke as soon as `sbert.py` landed. Replaced with a structural namespace check.

### Phase 3 (T07–T09)

No significant mid-stream corrections. The Protocol / fake / orchestrator boundaries from DESIGN §3 were tight enough that the implementations followed mechanically.

### Phase 4 (T10a–T10b)

- **T10a — `fastapi` dep scheduled in the wrong task.** TASKS.md scheduled `uv add fastapi` for T10b, but T10a's central exception handler IS FastAPI code and its parametrized §5 test needs a FastAPI app. I added the dep during T10a and noted it in the commit; T10b just reused it. Not a correction you had to make, but a TASKS.md ordering issue worth flagging — would have been cleaner to schedule `uv add fastapi` in T10a explicitly.
- **T10a — `UrlNamespaceError` message threading.** DESIGN §5 specifies the message as `"That looks like a {namespace} page, not an article."` with a placeholder, but the spec also says `(code, message, status)` are class-level attributes. Resolved by making the class-level message a generic default ("non-article Wikipedia page") and accepting an optional `namespace=` constructor arg that overrides via instance attribute when the validator knows which prefix matched. The §5 test asserts the class default; the validator threads the matched prefix.
- **T10b — `ASGITransport` doesn't drive the ASGI lifespan protocol.** httpx's `ASGITransport` ignores `lifespan` messages, so wrapping `create_app()` in `AsyncClient(transport=ASGITransport(app=app))` does NOT trigger the lifespan body. The lifespan tests instead invoke `lifespan(app)` directly as an `async with` context manager, then use ASGITransport for the HTTP probe inside that context. Worth remembering for T11/T12 ingest/chat tests — if they need a populated `app.state`, the same pattern applies.
- **T11 — lifespan side-stepped via dependency overrides.** Acting on the T10b lifespan-vs-ASGITransport note: the ingest tests override every `Depends(...)` provider in `app.dependency_overrides`, so `app.state` is never read and the lifespan doesn't need to run. Cleaner than nesting `async with lifespan(app):` around every HTTP probe, and it matches the T11 spec gotcha ("Use FastAPI dependency overrides for the fakes in tests, NOT real construction"). Recording so T12's chat tests follow the same pattern.
- **T11 — F9 error mapping written as a class of methods, not `@pytest.mark.parametrize`.** TASKS.md T11 says "parametrized". I chose 13 named methods inside a `TestF9ErrorMapping` class instead because each row needs a different fake (different exception class, sometimes a different `_RaisingVectorStore` subclass) and the parametrize id-rendering for exception-class params is noisy. Same end shape — one assertion per F9 row — but failure messages name the row (`test_llm_timeout_returns_504`) rather than a parametrize tuple. Flagging as a deliberate divergence from the plan, not a missed instruction.
**T12 by-hand smoke:** CHAT_GROUNDED prompt didn't enforce grounding
  strongly enough — F7 question leaked Shakespeare on first pass.
  Tightened to require exact "not found in the article" phrasing and
  moved the instruction to the end of the prompt for recency. Verified
  by hand against curl before proceeding to T13.

### Phase 5 (T13–T14)

- **T13 bare-metal smoke — Wikimedia retired `/page/mobile-sections/`.** Not an AI mistake — an external surprise. First post-T13 ingest of `Photosynthesis` returned 404 from the legacy endpoint we'd been calling all along. Resolution: pulled the Parsoid migration that NOTES already had as "future work" forward into a mid-Phase-5 fire drill, retired the JSON fixture in favour of an HTML one, swapped the parser. Full unit suite (238 tests) still green after the swap. Captured here so the trail is visible: the legacy REST API timeline turned out to be shorter than Wikimedia's own deprecation notice suggested. Lesson for future projects: even "long-term deprecation track" endpoints want a smoke test against the live API before relying on them in a take-home, not just synthetic fixtures.
- **T13 pre-flight — `.env` defaulted to docker hostnames.** Bare-metal smoke needs `localhost`-pointed URLs but the `.env.example` (and the user-copied `.env`) carried the containerised-mode `http://ollama:11434` / `http://qdrant:6333` from DESIGN §6.4. Symptom: `/api/health` returned `warmup_ok=false` because the warmup couldn't resolve `ollama`. Fixed `.env` to localhost; surfaced a separate test bug (`test_defaults_match_design_6_4_table` was reading user-local `.env` into the assertion — patched to `Settings(_env_file=None)` so the test asserts class defaults regardless of local config).
