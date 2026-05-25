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

(Will be filled in as the project progresses. Per REQUIREMENTS §13.)
