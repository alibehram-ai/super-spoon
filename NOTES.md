# NOTES.md

Scratch pad for things that didn't make it into REQUIREMENTS/DESIGN but matter for next-version thinking. Per REQUIREMENTS §13: optional file flagging AI mistakes that needed correction, and "what I would change with another two days." Living document.

---

## Future work

### Migrate off `/page/mobile-sections/` to Parsoid-backed endpoints

`WikipediaFetcher` uses `https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title}` for section-structured article content. That endpoint is part of Wikimedia's legacy REST API and is on the long-term deprecation track. The eventual replacement is the MediaWiki Core REST API's HTML endpoint (`/page/html/{title}`, Parsoid-served), which returns the full article as a structured HTML/Parsoid DOM rather than pre-sectioned JSON.

Migration shape (when Wikimedia announces a firm equivalent):
- Swap the second-call endpoint in `WikipediaFetcher.fetch`.
- Add a Parsoid HTML → section-list parser (split on `<section>` tags or `<h2>` headings depending on Parsoid version).
- Keep `/page/summary/{title}` for disambiguation/redirect detection (U15) — that one isn't deprecated.
- Integration fixture regenerates against the new endpoint shape; assertions verify section titles unchanged.

This isn't urgent — the legacy REST API will be supported for years — but a production-bound version would do it.

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
