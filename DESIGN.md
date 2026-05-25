# DESIGN.md

The design for the system specified in `REQUIREMENTS.md`. This document is the contract between requirements and implementation. Every choice here either implements a locked decision from REQUIREMENTS §8/§9/§10/§12 or makes a design-level call that REQUIREMENTS explicitly delegated. Conflicts with REQUIREMENTS are flagged at §10 of this doc, not silently resolved.

Read order: §1 (architecture) → §2 (data flow) → §3 (contracts) → §4 (decisions) → §5 (errors) → §6 (non-functional) → §7 (tests) → §8 (layout) → §9 (top 3 to challenge) → §10 (REQUIREMENTS deltas) → §11 (changelog).

---

## 1. Architecture overview

Ten components. One paragraph each. Each is a single Python module (or a small package) inside `backend/app/`, except the frontend.

**URL Validator** (`wikipedia/validator.py`). Pure function. Takes the raw URL string from the request body, normalises it (strip fragment, normalise `en.m.` → `en.`, lowercase host), and applies the F2 accept/reject rules as a sequence of checks. Returns a `ValidatedUrl` (canonical URL + article title) on success or raises `UrlValidationError` with a reason code. No I/O. Cheap, fully deterministic, and exhaustively unit-testable. Lives upstream of everything else so the rest of the pipeline can assume a clean article title.

**Wikipedia Fetcher** (`wikipedia/fetcher.py`). Wraps the Wikipedia REST API (`/page/summary/{title}` for metadata + disambiguation/redirect detection per U15, then `/page/mobile-sections/{title}` for sectioned body content). Carries the polite identifying user-agent from A13/U6 and does a single retry with exponential backoff on transient 5xx/timeout. Returns a `WikipediaArticle` carrying the lede, list of `RawSection`s (title + paragraphs), and the canonical title after redirect resolution. Raises `WikipediaFetchError(reason)` where `reason ∈ {not_found, disambiguation, transient, timeout}`. Network-bound; called from async route handlers via `httpx.AsyncClient`.

**Article Cleaner** (`wikipedia/cleaner.py`). Takes a `WikipediaArticle`, drops the sections listed in U12 (`References`, `External links`, `Further reading`, `See also`, `Notes`), enforces the F9/OQ-9 minimum (< 500 chars cleaned body → `ArticleTooShortError`), and enforces the A9/U13 maximum (truncate at the configured char cap, set `truncated=True` on the result). Returns a `CleanedArticle` with a normalised list of `Section`s, each carrying its title, ordered paragraphs as `list[str]`, char-offset ranges, and a `section_index`. Pure function over its input. Reused later for outline extraction during summarisation — no duplicated parsing.

**Section-aware Chunker** (`chunking/chunker.py`). Implements A10. For each `Section`: if the section's token count is at or below the section ceiling (default 1000), it's emitted as a single chunk preserving the full section. If above, the section is sub-chunked at `sub_size_tokens` (default 512) with `sub_overlap_tokens` (default 64) overlap, with chunk boundaries snapped backward to the nearest paragraph break (so we never split mid-sentence). Each chunk carries the metadata in F5: `section_title`, `section_index`, `chunk_index_in_section`, `char_offset_start`, `char_offset_end`, and `text`. A global `chunk_index` is assigned in document order — that's what F8's `sources[].chunk_index` refers to. Token counting uses the same tokenizer as the embedder (`bge-small-en-v1.5`'s tokenizer) to keep budgets honest. Pure function.

**Embedder** (`embedding/sbert.py`, ABC at `embedding/base.py`). Loads `BAAI/bge-small-en-v1.5` via `sentence-transformers` once at FastAPI startup (in the lifespan hook, before the warmup ping). Exposes `embed_texts(texts: list[str]) -> list[list[float]]` and a `vector_size` property (384 for this model). Embedding is batched at `EMBEDDING_BATCH_SIZE` (default 64). The implementation is sync and CPU-bound; route handlers offload it via `anyio.to_thread.run_sync` so the event loop isn't blocked. The class lives behind the `Embedder` Protocol so a hosted embedder (or a different SBERT model) can be swapped via env per §12 Challenge 1.

**Qdrant client wrapper** (`vectorstore/qdrant_store.py`, ABC at `vectorstore/base.py`). Owns the lifecycle of the single `active_article` collection (A7). `reset()` drops and recreates the collection with the embedder's vector size and cosine distance (configurable per A11). `upsert()` writes points whose payload carries the full Chunk metadata (including `text` per U14, so retrieval doesn't need a second hop). `search()` returns hits with metadata and score. `collection_exists()` is what the chat route checks for the A12 "ingest first" case. Uses the official `qdrant-client` (sync). Lives behind the `VectorStore` Protocol.

**Ollama LLM client wrapper** (`llm/ollama.py`, ABC at `llm/base.py`). Wraps `httpx.AsyncClient` against `/api/generate` (non-streaming per OQ-7). Exposes `generate(prompt, *, num_ctx, max_tokens, temperature, stop)` and `warmup()`. Prompt templates (system + summary, system + chat-grounded) live in `llm/prompts.py` as plain strings with `{placeholders}` — explicit, not Jinja, because there are exactly two of them. Carries timeout/error handling for the F9 cases (`OllamaUnreachableError`, `OllamaTimeoutError`, `LLMResponseInvalidError`). Sits behind the `LLMClient` Protocol.

**Summariser** (`rag/summariser.py`). Holds the A4 / §12 Challenge 3 logic. Counts tokens of the cleaned body using the embedder's tokenizer. If ≤ `SUMMARY_LONG_THRESHOLD_TOKENS` (default 3500), passes the full cleaned body to the LLM with the summary system prompt. Otherwise builds the **structured outline**: lede paragraph (verbatim) + for each section, the heading and its first sentence capped at `SUMMARY_PER_SECTION_TOKENS` tokens. Single LLM call either way. Returns the summary string; raises `LLMResponseInvalidError` if the model returns empty/non-prose output.

**RAG Orchestrator** (`rag/orchestrator.py`). Composes Embedder + VectorStore + LLMClient for the chat path. Validates that the collection exists (A12), embeds the question, calls `VectorStore.search(qvec, k=top_k)`, assembles the grounded prompt from the retrieved chunks, calls `LLMClient.generate()`, and constructs the `ChatResponse` payload. Source citations are built from the retriever output, not the LLM output (F8 / OQ-5) — they're attached to the response regardless of whether the model answered or said "not found in the article" (A17 / U8). No state across calls; this object is constructed once per request via FastAPI dependency injection.

**FastAPI app** (`api/`, with `main.py` as entrypoint). Three POST routes (`/ingest`, `/chat`) plus `/health`. `main.py` defines the lifespan: load embedder, construct vector store and LLM clients, **await** the A16 warmup ping with a bounded timeout (§4.2) so traffic only opens after the warmup resolves, mount the frontend static build at `/`. Dependency injection wires the orchestrator, summariser, fetcher, cleaner, chunker as FastAPI `Depends(...)` callables — the same wiring used by tests for substituting fakes. Errors are caught by a single exception handler that maps domain exceptions to the F9 error codes and structured JSON shape.

**React/Vite frontend** (`frontend/`). Single page, no router (U10/A14). Three components down the page: `UrlInput` (with inline validation feedback), `SummaryPanel` (renders summary + truncation flag from U13), `ChatPanel` (textbox + `ChatTranscript` of prior Q&A held in component state, each transcript entry rendering its `SourceList`). Loading state during ingest is a single indeterminate spinner with one label and an optional cold-start sub-label gated on `/api/health` (see §6.2). State held in `App.tsx` component state, lost on refresh (A14, OQ-12). Production build is served as static files by FastAPI from `/static`; the frontend hits `/api/*` on the same origin so no CORS.

Pushback on the listed component set: none. The user's list is exactly the set I'd draw. URL Validator is split out as its own module rather than living inside the Fetcher because the F2 reject cases never touch the network — keeping validation pure makes it cheap to test exhaustively and keeps fetch latency budgets honest.

---

## 2. Data flow

### Ingest flow

```
+--------------------+
| POST /api/ingest   |
| { url: string }    |
+----------+---------+
           |
           v
+--------------------+       UrlValidationError(reason)
| validate_url(url)  |--------------------------> { error, message } 400
| - normalise host   |
| - strip fragment   |
| - reject namespaces|
| - reject non-EN    |
+----------+---------+
           |  ValidatedUrl(title)
           v
+--------------------+       WikipediaFetchError(reason)
| WikipediaFetcher   |--------------------------> { error, message } 404/409/502
| .fetch(title)      |
| (httpx async +     |
|  retry once)       |
+----------+---------+
           |  WikipediaArticle
           v
+--------------------+       ArticleTooShortError
| ArticleCleaner     |--------------------------> { error, message } 422
| .clean(article)    |
| - drop F3 sections |
| - enforce min/max  |
| - set truncated    |
+----------+---------+
           |  CleanedArticle (with .truncated flag)
           v
+--------------------+
| Chunker.chunk()    |
| - section-aware    |
| - 1000 ceiling     |
| - 512/64 sub-chunks|
+----------+---------+
           |  list[Chunk]
           v
+--------------------+       (CPU-bound; runs in thread pool)
| Embedder           |
| .embed_texts(...)  |
| batched at 64      |
+----------+---------+
           |  list[vector]
           v
+--------------------+       QdrantUnreachableError
| VectorStore        |--------------------------> { error, message } 503
| .reset(coll, dim)  |
| .upsert(chunks,    |
|         vectors)   |
+----------+---------+
           |
           v
+--------------------+       LLMResponseInvalidError
| Summariser         |--------------------------> { error, message } 502
| .summarise(article)|       OllamaUnreachableError, OllamaTimeoutError
| - short: full body |
| - long: outline    |
+----------+---------+
           |  summary: str
           v
+----------------------------------------------------+
| 200 { summary, section_count, chunk_count,         |
|       truncated, char_count }                      |
+----------------------------------------------------+
```

### Query (chat) flow

```
+--------------------+
| POST /api/chat     |
| { question: str }  |
+----------+---------+
           |
           v
+--------------------+       NoActiveArticleError (A12)
| VectorStore        |--------------------------> { error: "no_active_article",
| .collection_exists |                              message: "Ingest a Wikipedia
| (active_article)   |                              article first." } 409
+----------+---------+
           |  ok
           v
+--------------------+
| Embedder           |  (thread pool)
| .embed_texts([q])  |
+----------+---------+
           |  qvec
           v
+--------------------+       QdrantUnreachableError
| VectorStore        |--------------------------> { error, message } 503
| .search(qvec, k=4) |
+----------+---------+
           |  list[Hit] (may be empty)
           v
+--------------------+       RetrievalEmptyError
| len(hits) == 0?    |--------------------------> { error: "retrieval_empty",
+----------+---------+                              message: "..." } 422
           |  hits
           v
+--------------------+
| build grounded     |
| prompt (chunks +   |
| question + system) |
+----------+---------+
           |  prompt
           v
+--------------------+       Ollama errors as above
| LLMClient          |
| .generate(prompt,  |
|   num_ctx=4096)    |
+----------+---------+
           |  answer
           v
+----------------------------------------------------+
| 200 { answer, sources: [                           |
|        { section_title, chunk_index, score,        |
|          excerpt }, ... 4 items                    |
|       ] }                                          |
+----------------------------------------------------+
```

Sources are always populated from `hits`, even when `answer == "not found in the article"` (U8/A17).

---

## 3. Module contracts

These are interfaces, not implementations. They define the test surface. Python `Protocol`s for the three NF4 swappable abstractions; plain classes elsewhere.

### Domain models (`domain/models.py`)

```python
@dataclass(frozen=True)
class ValidatedUrl:
    canonical_url: str       # e.g. "https://en.wikipedia.org/wiki/Photosynthesis"
    article_title: str       # e.g. "Photosynthesis"

@dataclass(frozen=True)
class RawSection:
    title: str               # "" for the lede
    paragraphs: list[str]

@dataclass(frozen=True)
class WikipediaArticle:
    canonical_title: str     # post-redirect
    lede: str                # full lede paragraph(s) as one string
    sections: list[RawSection]
    is_disambiguation: bool

@dataclass(frozen=True)
class Section:
    section_index: int
    title: str
    paragraphs: list[str]
    char_offset_start: int
    char_offset_end: int

@dataclass(frozen=True)
class CleanedArticle:
    title: str
    lede: str
    sections: list[Section]
    total_chars: int
    truncated: bool          # True iff hit the 200K char cap

@dataclass(frozen=True)
class Chunk:
    chunk_index: int                   # global, document order — F8 source id
    section_index: int
    chunk_index_in_section: int
    section_title: str
    text: str
    char_offset_start: int
    char_offset_end: int

@dataclass(frozen=True)
class Hit:
    chunk_index: int
    section_title: str
    text: str
    score: float

@dataclass(frozen=True)
class Source:
    section_title: str
    chunk_index: int
    score: float
    excerpt: str             # first ~200 chars of chunk text

@dataclass(frozen=True)
class ChatResponse:
    answer: str
    sources: list[Source]
```

### `LLMClient` (`llm/base.py`)

```python
class LLMClient(Protocol):
    async def generate(
        self,
        prompt: str,
        *,
        num_ctx: int,                          # required; caller passes CHAT_NUM_CTX or SUMMARY_NUM_CTX
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> str:
        """Return the LLM's completion text. Strips leading/trailing whitespace.
        num_ctx has no default — call-type choice (chat vs summary) lives at the
        call site, not buried in this abstraction.
        Raises:
            OllamaUnreachableError - connection refused / DNS / 5xx after retry.
            OllamaTimeoutError     - request exceeded OLLAMA_TIMEOUT_S.
            LLMResponseInvalidError - empty string or no `response` field.
        """

    async def warmup(self) -> bool:
        """One-shot no-op generation to load the model into VRAM.
        Returns True on success, False on any failure (best-effort per A16).
        Never raises."""
```

### `VectorStore` (`vectorstore/base.py`)

```python
class VectorStore(Protocol):
    def reset(self, collection: str, vector_size: int, metric: str = "cosine") -> None:
        """Drop the collection if it exists, then create it with the given
        vector size and distance metric (F6 / A7).
        Raises: QdrantUnreachableError."""

    def upsert(self, collection: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Write chunks with their embeddings. len(chunks) must equal len(embeddings).
        Each point's id is chunk.chunk_index; payload carries all Chunk fields per U14.
        Raises: QdrantUnreachableError, ValueError on length mismatch."""

    def search(self, collection: str, embedding: list[float], k: int) -> list[Hit]:
        """Return up to k hits ordered by score desc. Returns [] if no points.
        Raises: QdrantUnreachableError, CollectionMissingError."""

    def collection_exists(self, collection: str) -> bool:
        """Used by the chat route to detect the A12 case.
        Raises: QdrantUnreachableError."""
```

### `Embedder` (`embedding/base.py`)

```python
class Embedder(Protocol):
    @property
    def vector_size(self) -> int:
        """e.g. 384 for bge-small-en-v1.5."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batched embedding. Synchronous, CPU-bound; callers offload to a
        thread pool. Empty input returns [].
        Raises: EmbeddingError on model failure (not expected in practice)."""
```

### `WikipediaFetcher` (`wikipedia/fetcher.py`)

```python
class WikipediaFetcher:
    def __init__(self, *, base_url: str, user_agent: str, timeout_s: float, max_retries: int = 1) -> None: ...

    async def fetch(self, article_title: str) -> WikipediaArticle:
        """
        Calls /page/summary/{title} to detect disambiguation/redirect/404,
        then /page/mobile-sections/{title} for sectioned content.

        NOTE: /page/mobile-sections/ is on Wikimedia's long-term deprecation
        track; see NOTES.md for the Parsoid migration future-work item.

        Raises:
            WikipediaFetchError("not_found")      - 404 from summary endpoint.
            WikipediaFetchError("disambiguation") - summary.type == "disambiguation" (U15).
            WikipediaFetchError("transient")      - 5xx after one retry.
            WikipediaFetchError("timeout")        - request exceeded timeout_s.

        Redirects are followed silently per OQ-8; the canonical title in the
        returned WikipediaArticle reflects the post-redirect target.
        """
```

### `ArticleCleaner` (`wikipedia/cleaner.py`)

```python
class ArticleCleaner:
    def __init__(
        self,
        *,
        dropped_section_titles: frozenset[str] = frozenset({
            "References", "External links", "Further reading", "See also", "Notes",
        }),
        min_chars: int = 500,
        char_cap: int = 200_000,
    ) -> None: ...

    def clean(self, article: WikipediaArticle) -> CleanedArticle:
        """
        - Drop sections whose title is in dropped_section_titles (case-insensitive).
        - Strip inline citation markers ([1], [2], etc.) per F3 spirit.
        - Compute total_chars on the cleaned text.
        - If total_chars < min_chars: raise ArticleTooShortError.
        - If total_chars > char_cap: truncate to char_cap at the nearest section
          boundary backwards, set truncated=True (A9/U13). Never raise on overflow.
        """
```

### `Chunker` (`chunking/chunker.py`)

```python
class Chunker:
    def __init__(
        self,
        *,
        section_ceiling_tokens: int = 1000,
        sub_size_tokens: int = 512,
        sub_overlap_tokens: int = 64,
        tokenizer: Callable[[str], list[int]],   # same as Embedder's tokenizer
    ) -> None: ...

    def chunk(self, article: CleanedArticle) -> list[Chunk]:
        """
        For each section in article.sections:
          - If token_count(section.text) <= section_ceiling_tokens:
              emit one Chunk for the whole section.
          - Else:
              walk paragraphs accumulating until token budget hits sub_size_tokens,
              emit a chunk, back up by sub_overlap_tokens (snapped to paragraph
              start), continue.
        Assigns global chunk_index in document order.
        Never returns []  (callers rely on CleanedArticle having ≥1 section after cleaner).
        """
```

### `Summariser` (`rag/summariser.py`)

```python
class Summariser:
    def __init__(
        self,
        llm: LLMClient,
        *,
        long_threshold_tokens: int = 3500,
        per_section_tokens: int = 30,
        tokenizer: Callable[[str], list[int]],
        prompt_template_short: str,
        prompt_template_long: str,
    ) -> None: ...

    async def summarise(self, article: CleanedArticle) -> str:
        """
        Single LLM call (A4).
        - If token_count(article body) <= long_threshold_tokens:
            input = full cleaned body
            prompt = prompt_template_short.format(body=...)
        - Else:
            outline = lede + "\n\n".join(
                f"## {s.title}\n{first_sentence_or_n_tokens(s, per_section_tokens)}"
                for s in article.sections
            )
            prompt = prompt_template_long.format(outline=outline)
        Calls LLMClient.generate(prompt, num_ctx=SUMMARY_NUM_CTX) and returns the result.
        Raises: LLMResponseInvalidError, OllamaUnreachableError, OllamaTimeoutError.
        """
```

### `RAGOrchestrator` (`rag/orchestrator.py`)

```python
class RAGOrchestrator:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        llm: LLMClient,
        collection: str,
        top_k: int,
        chat_prompt_template: str,
        chat_num_ctx: int,                     # required; wired from CHAT_NUM_CTX via api/deps.py
    ) -> None: ...

    async def answer(self, question: str) -> ChatResponse:
        """
        1. If not vector_store.collection_exists(collection): raise NoActiveArticleError (A12).
        2. qvec = embedder.embed_texts([question])[0]  (offloaded to thread).
        3. hits = vector_store.search(collection, qvec, k=top_k).
        4. If not hits: raise RetrievalEmptyError.
        5. prompt = chat_prompt_template.format(
              chunks="\n\n".join(f"[{i+1}] {h.section_title}\n{h.text}" for i,h in enumerate(hits)),
              question=question,
           )
        6. answer = await llm.generate(prompt, num_ctx=chat_num_ctx).
        7. sources = [Source(h.section_title, h.chunk_index, h.score, h.text[:200]) for h in hits].
        8. Return ChatResponse(answer=answer.strip(), sources=sources).

        Raises: NoActiveArticleError, RetrievalEmptyError, QdrantUnreachableError,
                OllamaUnreachableError, OllamaTimeoutError, LLMResponseInvalidError.
        """
```

### `validate_url` (`wikipedia/validator.py`)

```python
_HOST_ACCEPT = {"en.wikipedia.org", "en.m.wikipedia.org"}
_HOST_REJECT_LANG_PREFIX_PATTERN = re.compile(r"^[a-z]{2,3}(\.m)?\.wikipedia\.org$")
_NAMESPACE_PREFIXES = ("Special:", "Talk:", "Category:", "File:", "Help:", "User:", "Portal:")

def validate_url(raw_url: str) -> ValidatedUrl:
    """
    Pipeline (each step raises UrlValidationError(reason) on failure).
    Order matches the §4.9 table exactly:
      1. parse_url(raw) — reason="malformed" if urlparse fails or scheme not in {http, https}.
      2. host accept   — reason="non_english"   if host matches r"^[a-z]{2,3}(\\.m)?\\.wikipedia\\.org$" but not in _HOST_ACCEPT.
      3. host wiki     — reason="non_wikipedia" if host doesn't end in .wikipedia.org.
      4. curid query   — reason="curid"         if query contains 'curid' (F2 rejects ?curid= URLs).
      5. path shape    — reason="malformed"     if path doesn't match /wiki/<title>.
      6. namespace     — reason="namespace"     if URL-decoded title starts with any _NAMESPACE_PREFIXES.
      7. title nonempty — reason="malformed"    if URL-decoded title is empty.
      8. normalise     — host en.m.wikipedia.org -> en.wikipedia.org; strip fragment; URL-decode title.
      9. return ValidatedUrl(canonical_url=..., article_title=...).
    """
```

---

## 4. Key design decisions with trade-offs

### 4.1 LLM runtime topology and model pull (one-command bring-up)

**Choice.** Three compose services + one init service:
- `ollama` — official `ollama/ollama` image, persistent named volume `ollama_models` for model cache, healthcheck on `/api/tags`.
- `ollama-pull` — init service using the same image, runs `ollama pull llama3.2:3b` against the ollama service, exits 0 on success. Marked `restart: "no"`.
- `qdrant` — official `qdrant/qdrant`, **no named volume** (ephemeral per A2).
- `app` — backend image (multi-stage build that also produces the frontend `dist/`); `depends_on: { ollama-pull: { condition: service_completed_successfully }, qdrant: { condition: service_started } }`.

The init-service pattern means: first `docker compose up` blocks on the model download (~2 GB once); the `ollama_models` named volume keeps it for subsequent ups. The app process doesn't even start until the model is local, so the warmup ping (next subsection) has a real chance of succeeding before the first user request.

**Alternatives rejected.**
- *Custom Ollama entrypoint that pulls then starts the server.* Conflates server lifecycle with provisioning; complicates restart semantics; needs a custom Dockerfile for a stock image.
- *App-side retry-until-ready that runs the pull on first request.* User's first request hangs for the entire ~2 GB download. Terrible demo UX; defeats the spirit of F10's "first request may be slow".
- *Healthcheck-only with no init.* Healthcheck checking `ollama list` doesn't *cause* the pull, just observes it. Needs a pull mechanism anyway, so this is a complement not an answer.

**Host-side fallback (NF1 alternative).** README documents: comment out the `ollama` and `ollama-pull` services, run `ollama serve` on the host, set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env`, `docker compose up app qdrant`. Same code path; only the URL changes.

### 4.2 Warmup ping (A16) and how it interacts with the pull

**Choice.** The warmup runs in the FastAPI `lifespan` context manager as `await asyncio.wait_for(llm.warmup(), timeout=WARMUP_TIMEOUT_S)` with `WARMUP_TIMEOUT_S=30` by default, wrapped in a try/except that catches both `asyncio.TimeoutError` and any exception bubbling from `warmup()`. On success: `app.state.warmup_ok = True` and the app starts accepting traffic. On timeout or exception: log a `WARNING` with the cause, set `app.state.warmup_ok = False`, and **still** start accepting traffic. The `/health` endpoint surfaces `warmup_ok` so the demo recording can tell at a glance whether the first user request will be warm or cold.

**Interpretation of A16's "best-effort".** A16 says the warmup is "best-effort, non-blocking" and that "failure does not block startup." We read "non-blocking" as "failure doesn't crash startup," **not** "fire-and-forget that completes invisibly some time later." The earlier draft used `asyncio.create_task(...)` and was effectively useless — the app declared ready before the model was loaded, and the user's first request could overtake the warmup and pay the cold-start tax anyway. Awaiting with a bounded timeout means: if the warmup succeeds (the common path now that the init service has already pulled the model), the first real request *is* warm; if it doesn't, we degrade explicitly rather than silently.

The warmup body is a single `generate("ping", max_tokens=1, num_ctx=512)` call. The Ollama timeout inside `OllamaClient.warmup()` is set tighter than `WARMUP_TIMEOUT_S` so the `wait_for` doesn't race the inner httpx timeout.

**Race-free contract.** Init-service-completed → app-starts → lifespan-awaits-warmup → traffic-accepted. The warmup and the model pull never race because they happen sequentially across services (the `depends_on: condition: service_completed_successfully` gate ensures the pull is finished before `app` even boots). If the user manually skips the init service or runs in host-fallback mode without pre-pulling, the warmup degrades to "warning log + `warmup_ok=False`, app still serves" — consistent with A16's "best-effort" wording under the interpretation above.

### 4.3 Embedder choice — `bge-small-en-v1.5`, in-process, env-swappable

**Choice.** `BAAI/bge-small-en-v1.5` loaded via `sentence-transformers`, 384-dim, ~133 MB on disk, loaded once at startup. Embedding runs synchronously inside `anyio.to_thread.run_sync` to keep the event loop unblocked.

**Why in-process and not Ollama.** Per OQ-3: batched in-process embedding is ~10× faster for a single-article workload than serial Ollama calls; decouples the embedding lifecycle from the LLM runtime; gives us a real third abstraction for NF4; image-size cost is negligible (sentence-transformers is ~500 MB CPU-only, the model itself ~133 MB).

**Why bge-small over MiniLM.** Per §12 Challenge 1: ~10-point MTEB retrieval lift at ~50% more compute, which is irrelevant at single-article scale. To preserve the option to flip if MiniLM proves to feel snappier in the demo, `EMBEDDING_MODEL` is the only env var the user needs to change; the `Embedder` Protocol means call sites don't move. Tokenizer is loaded from whatever model is configured.

### 4.4 Vector DB — Qdrant, ephemeral, single collection

**Choice.** `qdrant/qdrant:v1.11` (pinned for reproducibility), HTTP API, **no named volume**, single collection `active_article`. On every `/api/ingest`, the orchestrator calls `vector_store.reset("active_article", vector_size=384, metric="cosine")` — drops then recreates — before upserting (F6/A7).

**Alternatives rejected.**
- *Per-article hashed collections.* Pointless given A2 ephemerality and no caching. Adds collection-name plumbing for zero benefit.
- *In-process FAISS/Chroma.* Violates Constraint §7 (vector DB must be a compose service).
- *Persistent volume.* Would conflict with A2's ephemerality and let stale data leak across sessions.

### 4.5 Section-aware chunking strategy (A10)

**Choice.** Whole-section first, paragraph-aware sub-chunking second. For a section with `token_count <= section_ceiling_tokens` (default 1000), emit one chunk preserving the section verbatim — this is the common case for Wikipedia and means F8's citations point to clean section-level answers. For longer sections, walk the paragraph list accumulating tokens until `sub_size_tokens` (default 512); emit; rewind by `sub_overlap_tokens` (default 64) snapped backward to the previous paragraph boundary; continue.

**Why paragraph-aware.** Sub-chunking on raw token windows splits mid-sentence and produces chunks that fail closed retrieval ("This is because [chunk break]"). Snapping boundaries to paragraph breaks costs nothing — paragraph breaks are unambiguous in the REST API output — and gives every chunk semantic closure.

**Why 1000-token section ceiling.** Median Wikipedia section is 200–400 tokens; 1000 catches >80% of sections whole. The handful above it (1000+ token sections on long technical articles) genuinely benefit from sub-chunking for retrieval precision.

**All thresholds env-tunable (U11):** `CHUNK_SECTION_CEILING_TOKENS`, `CHUNK_SUB_SIZE_TOKENS`, `CHUNK_SUB_OVERLAP_TOKENS`.

### 4.6 Retrieval parameters (A11, §12 Challenge 2)

**Choice.** `top_k=4`, cosine distance, no MMR or rerank. Both `RETRIEVAL_TOP_K` and `SIMILARITY_METRIC` are env-tunable.

**Token math against `chat_num_ctx`.** Worst case: 4 chunks at the sub-chunk size (~512 tokens each) = 2048 tokens of content + ~300 tokens of system/grounding prompt + ~80 tokens of question + ~400 tokens of room for the answer = ~2828 tokens. `chat_num_ctx=4096` leaves ~1.2K tokens of slack — comfortable. The headroom is what lets §12 Challenge 2's "try K=6 in dev" recommendation work without a config change beyond `RETRIEVAL_TOP_K=6` (K=6 worst case is ~3.8K tokens, still fits 4096).

### 4.7 Long-article summarisation strategy

**Choice.** Implements A4 / §12 Challenge 3 resolution. Single LLM call always. Branching on cleaned-body token count:
- `<= SUMMARY_LONG_THRESHOLD_TOKENS` (default **3500**) → full body input.
- `> SUMMARY_LONG_THRESHOLD_TOKENS` → **structured outline**: verbatim lede + per-section `## <heading>\n<first sentence, capped at SUMMARY_PER_SECTION_TOKENS tokens>`.

**Why 3500 not 5000.** The earlier draft put the threshold at 5000 tokens, which made the *short-article path* the long-pole latency case (~25–40 s prefill on `llama3.2:3b` warm — borderline against F10's 60 s budget on cold start). 3500 moves the boundary so the short path tops out around ~15–25 s and leaves real headroom. The trade-off: articles with cleaned bodies in 3500–5000 tokens (which used to get full-body summarisation) now route to the outline path. On Wikipedia specifically that's fine — those mid-length articles still have a section structure the outline can exploit, and the F10-headroom argument outweighs the marginal quality loss on a few articles in that band. As a happy side-effect, *Photosynthesis* (~4–5K tokens cleaned) now exercises the outline path in the integration test, which is the right testing posture: the path with more logic gets the smoke test, not the trivial branch.

**Prompt structure (long-article path).**
```
[system]
You are summarising an English Wikipedia article. Below is its structured
outline: the lede paragraph in full, then each section heading with its
opening sentence. Produce a 4–8 sentence summary that reflects the full
article (not just the lede). Do not invent facts. Write plain prose, no
headings or bullets.

[outline]
<lede paragraph verbatim>

## <Section 1 title>
<first sentence of Section 1>

## <Section 2 title>
<first sentence of Section 2>
... (up to ~25 sections in practice)

[instruction]
Write the 4–8 sentence summary now.
```

**Token math against llama3.2:3b's context window.** Ollama's `num_ctx` defaults to 2048, which would silently truncate longer prompts. We set `SUMMARY_NUM_CTX=8192` explicitly per request. For long-article path:
- Input: ~200 tok lede + ~25 sections × ~30 tok = ~950 tok content + ~300 tok system framing = **~1250 tok input**.
- Output: 4–8 sentences ≈ 150–250 tok.
- Total: ~1500 tok — well below 8192.

For short-article path (threshold 3500):
- Input: up to 3500 tok body + ~200 tok framing = ~3700 tok.
- Output: ~200 tok.
- Total: ~3900 tok — well inside 8192.

**Latency expectation (warm).** Long path: ~1250 input tok prefill + ~200 output tok generation on `llama3.2:3b` warm ≈ 10–18 s on a developer laptop. Short path: ~3700 tok prefill ≈ 15–25 s. Both comfortably inside F10's 60 s budget on warm requests; cold-start adds the documented Ollama load tax on top.

**Alternatives rejected.** Per §12 Challenge 3: hierarchical map-reduce (~13 LLM calls, blows F10); leading-N (biased toward an already-summarised lede, misses late-article material). Outline wins on coverage at one-fifth the input tokens.

### 4.8 Frontend topology

**Choice.** Single React/Vite SPA. No router. State lives in `App.tsx` component state, lost on refresh (A14, OQ-12). Production build (`npm run build`) is produced in a Node stage of the backend's multi-stage Dockerfile, copied into the Python stage as `/app/static/`, and mounted by FastAPI as `app.mount("/", StaticFiles(directory="static", html=True))`. API routes live under `/api/*`. Same origin, no CORS, single container, single port (`8000`).

**Why FastAPI-serves-the-static-build over a separate frontend container.** Per your answer in clarifying-questions: one container is easier to bring up and is consistent with "one command brings the full stack online". The trade-off is a slightly less clean dev experience — Vite's hot reload requires running `npm run dev` separately on the host pointing at the running FastAPI's `/api` — but that's a dev convenience, not a deployment property.

### 4.9 URL validation pipeline (F2 concretised)

See §3 `validate_url` for the function signature; the design intent is that validation is a **sequence of cheap pure checks** ordered by specificity, each producing a distinct reason code for distinct user-facing messages:

| Order | Check | Reason on fail | User message |
| --- | --- | --- | --- |
| 1 | `urlparse` parses, scheme ∈ {http, https} | `malformed` | "That doesn't look like a URL." |
| 2 | Host is exactly `en.wikipedia.org` or `en.m.wikipedia.org` | `non_english` if matches `<lang>.wikipedia.org` | "Only English Wikipedia is supported." |
| 3 | Host ends in `.wikipedia.org` | `non_wikipedia` | "Please paste a Wikipedia URL." |
| 4 | Path matches `/wiki/<title>` (not `?curid=...`) | `curid` | "Use the article URL, not a curid link." |
| 5 | Title prefix not in namespace blocklist | `namespace` | "That's a {namespace} page, not an article." |
| 6 | Title non-empty after URL-decode | `malformed` | (same as 1) |

Normalisation runs after all checks pass: `en.m.` → `en.`, drop fragment, URL-decode title.

### 4.10 Disambiguation detection via REST API metadata (U15)

**Choice.** Use the Wikipedia REST API's `/page/summary/{title}` endpoint, which returns a `type` field. We check `summary.type == "disambiguation"`. No body-text heuristics, no parsing for telltale strings like "may refer to". Same call surfaces the 404 case (HTTP 404) and the redirect target (`titles.canonical` ≠ requested title).

This means the fetcher's first network hop is the summary endpoint (cheap, ~1 KB response), and only on a clean non-disambiguation result does it issue the second call to `/page/mobile-sections/{title}` for the body. Two-call ingest, but the first call is small and gates the second.

**Deprecation note.** `/page/mobile-sections/` is on Wikimedia's long-term deprecation track — the endpoint is part of the older REST API and will eventually be replaced by Parsoid-backed endpoints (`/page/html/{title}`, MediaWiki Core REST API). For the lifetime of this take-home it is fine to use, but a production-bound version would migrate. Tracked in `NOTES.md` as future work.

---

## 5. Error handling and edge cases

All errors surface to the frontend as `HTTP <status> { "error": "<code>", "message": "<user-facing>" }`. The frontend's `ErrorBanner` component renders the message verbatim. Status codes follow REST conventions: 400 client input, 404 not found, 409 conflict (state-dependent), 422 unprocessable (content-shape failures), 502 upstream model/DB failure, 503 upstream unreachable.

| F9 case | Owner module | Error code | HTTP | User-facing message |
| --- | --- | --- | --- | --- |
| Malformed URL | `validate_url` | `url_malformed` | 400 | "That doesn't look like a valid URL." |
| Non-Wikipedia URL | `validate_url` | `url_non_wikipedia` | 400 | "Please paste a Wikipedia article URL." |
| Non-English wiki | `validate_url` | `url_non_english` | 400 | "Only English Wikipedia (en.wikipedia.org) is supported." |
| Namespace URL | `validate_url` | `url_namespace` | 400 | "That looks like a {namespace} page, not an article." |
| `?curid=` URL | `validate_url` | `url_curid` | 400 | "Use the regular article URL, not a curid link." |
| 404 from Wikipedia | `WikipediaFetcher` | `article_not_found` | 404 | "Wikipedia doesn't have an article at that URL." |
| Disambiguation page | `WikipediaFetcher` | `article_disambiguation` | 422 | "That URL points to a disambiguation page — pick one of the linked articles." |
| Wikipedia transient/timeout | `WikipediaFetcher` | `wikipedia_unavailable` | 502 | "Wikipedia didn't respond. Try again." |
| Article < 500 chars | `ArticleCleaner` | `article_too_short` | 422 | "This article is too short to be useful (under 500 characters)." |
| Article > 200K chars | `ArticleCleaner` | *(not an error — truncated flag in response)* | 200 | UI shows "Article was truncated to fit." inline above the summary. |
| Ollama unreachable | `OllamaClient` | `llm_unavailable` | 503 | "The local LLM isn't responding. Is Ollama running?" |
| Ollama timeout | `OllamaClient` | `llm_timeout` | 504 | "The local LLM took too long. Try again or check Ollama load." |
| **Malformed/empty LLM response** | `OllamaClient` / `Summariser` | `llm_response_invalid` | 502 | "The model returned an unparseable response. Try rephrasing or check Ollama logs." |
| Qdrant unreachable | `QdrantVectorStore` | `vector_store_unavailable` | 503 | "The vector database isn't responding." |
| Retrieval zero chunks | `RAGOrchestrator` | `retrieval_empty` | 422 | "No relevant content found in the article for that question." |
| Chat before ingest (A12) | `RAGOrchestrator` | `no_active_article` | 409 | "Ingest a Wikipedia article first, then ask questions about it." |

**Error mapping is centralised** in `api/errors.py` via a single FastAPI `@app.exception_handler(DomainError)` registered in `main.py`. The handler maps the exception class to the status code and message; domain code raises typed exceptions and never builds HTTP responses itself.

**The malformed-LLM-response case was missing from F9** (I flagged this in the pre-draft summary). REQUIREMENTS.md has been amended per your direction to enumerate `llm_response_invalid` alongside the other F9 cases. See §11 changelog.

**Note on Ollama timeout vs unreachable.** Both are caught from `httpx` — `ConnectError` / `ConnectTimeout` / `RemoteProtocolError` → unreachable; `ReadTimeout` after the connection is established → timeout. Different codes because they need different user advice.

---

## 6. Non-functional design

### 6.1 Concurrency model

| Endpoint | Async? | Why |
| --- | --- | --- |
| `POST /api/ingest` | `async def` | Wikipedia fetch (httpx async) and LLM summary call (httpx async) are I/O-bound. The embedding step is sync CPU-bound and gets offloaded via `anyio.to_thread.run_sync(embedder.embed_texts, chunks)` so it doesn't block the event loop. The Qdrant client is sync; its `upsert` is fast enough (single-article scale, < 200 ms typical) to call directly, but is also wrapped in `to_thread` for safety. |
| `POST /api/chat` | `async def` | Same logic: embed in thread pool, search in thread pool (fast), LLM call awaited as I/O. |
| `GET /api/health` | `async def` | Reports liveness + `warmup_ok`. Trivial. |

A single user is assumed (A6); concurrency between requests is not optimised for, but the above means a concurrent second request doesn't *starve* the event loop while the first is embedding.

### 6.2 Surfacing ingest latency to the user

Ingest is blocking per OQ-7 — for a long article (~10K tokens cleaned body, ~50 chunks, outline-path summary) cold-start it can be 30–60 s, and warm ~20–40 s. The frontend shows a single **indeterminate spinner** with one milestone-driven label set on submit and a cold-start sub-label:

```
Primary:  "Processing article…"
Sub:      "First request after startup is slower while the model loads."
```

The label is set when the request fires and stays put until the response arrives — no wall-clock fake progress, no client-driven phase transitions. The earlier draft staged four labels by elapsed-time heuristics; that was dishonest progress dressed up as feedback and is dropped. The sub-label is shown only when the frontend's `/api/health` poll on page load returned `warmup_ok=false` (otherwise it's hidden — no need to apologise for cold start when the warmup succeeded). No cancel button: the backend has no real cancellation path, so a "Cancel" affordance would either silently keep working or lie about what it does. Both are worse than no button.

A real progress channel via SSE (per-phase events `fetch_done`, `chunk_done`, `embed_done`, `summarising`) would replace this with honest progress but adds a streaming endpoint and a `EventSource` client. Parked in `NOTES.md` as future work.

### 6.3 Resource ceilings (rough)

Two numbers per component: resident memory at idle (after the workload settles) and peak during normal operation. The rollup sums the peak column.

| Component | Memory (idle / peak) | Disk | Notes |
| --- | --- | --- | --- |
| `app` (Python + sentence-transformers + bge-small) | 700 MB / 1.2 GB | ~1.5 GB image | Peak hit during the embedding step of a long-article ingest. |
| `qdrant` | 100 MB / 300 MB | ~150 MB image | Peak with a 50-chunk article indexed. Ephemeral, no volume. |
| `ollama` (with llama3.2:3b loaded) | 3.5 GB / 4.5 GB | ~200 MB image + ~2 GB model in `ollama_models` named volume | Idle = model loaded, no active gen. Peak during a summary or chat generate. |
| `ollama-pull` (init, runs once then exits) | 50 MB / 50 MB | — | Reuses ollama image. Not counted in steady-state rollup. |
| **Steady-state rollup (peak)** | **~6 GB RAM** | ~4 GB disk | Sum of app + qdrant + ollama peak columns. Within developer-laptop budget. |

This is what NF8 (reasonable image sizes) and the README's "prerequisites" section commit to.

### 6.4 Configuration surface

Every env var. Defaults shown. **Bold** entries are the U11-flagged tunables (chunking, retrieval, length cap, model names).

| Env var | Default | Controls |
| --- | --- | --- |
| **`LLM_MODEL`** | `llama3.2:3b` | Ollama model tag pulled by `ollama-pull` and used by all LLM calls. |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Override to `http://host.docker.internal:11434` for host-side fallback. |
| `OLLAMA_TIMEOUT_S` | `120` | httpx timeout on LLM generate calls. |
| `OLLAMA_WARMUP_ENABLED` | `true` | A16 warmup ping toggle. |
| `WARMUP_TIMEOUT_S` | `30` | Lifespan awaits the warmup with this timeout; on TimeoutError, logs a warning and starts anyway (§4.2). |
| `SUMMARY_NUM_CTX` | `8192` | Ollama `num_ctx` for summary calls. |
| `CHAT_NUM_CTX` | `4096` | Ollama `num_ctx` for chat calls. |
| **`EMBEDDING_MODEL`** | `BAAI/bge-small-en-v1.5` | sentence-transformers model id. The one-env-var swap from §12 Challenge 1. |
| `EMBEDDING_BATCH_SIZE` | `64` | Batch size in `Embedder.embed_texts`. |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant HTTP endpoint. |
| `QDRANT_COLLECTION` | `active_article` | Single collection name (A7). |
| **`RETRIEVAL_TOP_K`** | `4` | Top-K hits returned by `VectorStore.search`. The §12 Challenge 2 flip-point. |
| **`SIMILARITY_METRIC`** | `cosine` | Qdrant distance metric. |
| **`CHUNK_SECTION_CEILING_TOKENS`** | `1000` | A10 ceiling. |
| **`CHUNK_SUB_SIZE_TOKENS`** | `512` | A10 sub-chunk size. |
| **`CHUNK_SUB_OVERLAP_TOKENS`** | `64` | A10 sub-chunk overlap. |
| **`ARTICLE_CHAR_CAP`** | `200000` | A9/U13 truncation cap. |
| `ARTICLE_MIN_CHARS` | `500` | F9/OQ-9 rejection floor. |
| `SUMMARY_LONG_THRESHOLD_TOKENS` | `3500` | A4 short/long branch. Lowered from 5000 during design review — see §4.7 rationale. |
| `SUMMARY_PER_SECTION_TOKENS` | `30` | A4 per-section snippet cap (first sentence capped at this many tokens). |
| `WIKIPEDIA_API_BASE` | `https://en.wikipedia.org/api/rest_v1` | Override for testing only. |
| `WIKIPEDIA_USER_AGENT` | `silver-spoon/0.1 (https://github.com/<user>/silver-spoon2; contact: <email>)` | A13/U6 polite UA. |
| `WIKIPEDIA_FETCH_TIMEOUT_S` | `15` | httpx timeout on REST API calls. |
| `LOG_LEVEL` | `INFO` | Standard. |

All of these go into `.env.example` with the values above. `.env` is gitignored (NF2).

---

## 7. Testing strategy

Target ≥ 85% line coverage (NF5) with meaningful tests (NF6).

### 7.1 Unit tests

LLM, VectorStore, and Embedder substituted with in-memory fakes (not mocks-with-asserts; full fakes that implement the Protocol). The fakes live in `tests/fakes.py`:
- `FakeLLM` — records every call; `generate()` returns a configurable canned response; can be set to raise specific exceptions.
- `FakeVectorStore` — dict-backed; cosine implemented manually for `search()`; `collection_exists` tracks state.
- `FakeEmbedder` — deterministic hash-to-vector; `vector_size=384`.

Test suites and what they assert:

| Suite | Asserts |
| --- | --- |
| `test_url_validator.py` | Each F2 reject reason (malformed/non_wikipedia/non_english/namespace/curid) raises with the right reason; `en.m.` normalises to `en.`; fragments stripped; URL-decoded titles round-trip; valid URLs return the canonical form. |
| `test_wikipedia_fetcher.py` | 404 → `WikipediaFetchError("not_found")`; `summary.type == "disambiguation"` → `("disambiguation")` per U15; 503 then 200 retry succeeds; 503 twice raises `("transient")`; user-agent header present on every call; redirect target is reflected in returned `canonical_title`. Uses `httpx.MockTransport`. |
| `test_article_cleaner.py` | Dropped section titles removed (case-insensitive); inline citation markers stripped; total_chars accurate; < min_chars raises `ArticleTooShortError`; > char_cap truncates at section boundary and sets `truncated=True`; section_index reassigned post-drop. |
| `test_chunker.py` | A section under the ceiling emits one chunk with full text; a section over the ceiling produces multiple chunks of approx `sub_size_tokens` each; sub-chunks have ~`sub_overlap_tokens` of overlap; chunk boundaries align to paragraph starts (no mid-sentence splits); global `chunk_index` is monotonic; all F5 metadata fields populated. |
| `test_embedder.py` (unit) | `SentenceTransformersEmbedder`'s logic with `sentence_transformers.SentenceTransformer` patched to a stub: `embed_texts([])` short-circuits to `[]`; batching iterates `EMBEDDING_BATCH_SIZE` slices in order; `vector_size` proxies to the underlying model. No real model load; suite stays fast. |
| `test_chunker.py` uses a deterministic fake tokenizer | The chunker depends on a tokenizer callable, not on the SBERT model. Unit tests pass a fake `lambda s: list(range(len(s.split())))` (one-token-per-word) so chunker behaviour is exercised without loading the embedder. |
| `test_qdrant_store.py` | Against a `qdrant-client` in-memory mode (or a docker fixture if local-mode is unstable): `reset` drops then creates; `upsert` then `search` returns hits ordered by score; `collection_exists` reflects creates and drops; length mismatch raises `ValueError`. |
| `test_ollama_client.py` | Mocks httpx: 200 with `response` field returns text; empty `response` raises `LLMResponseInvalidError`; connection refused raises `OllamaUnreachableError`; ReadTimeout raises `OllamaTimeoutError`; `warmup()` returns False (never raises) on any failure. |
| `test_summariser.py` | Short article (≤ threshold) calls LLM once with the full body in the prompt; long article calls LLM once with outline structure (lede + headings + first sentences); empty LLM response surfaces as `LLMResponseInvalidError`. |
| `test_orchestrator.py` | A12 — no collection → `NoActiveArticleError`; empty hits → `RetrievalEmptyError`; 4 hits → answer + 4 sources; sources are built from hits regardless of LLM output (U8/A17 — assert sources present even when answer is "not found in the article"); chunk_index preserved end-to-end. |
| `test_api_ingest.py` | Each F9 error from the ingest pipeline maps to the right HTTP status + error code (driven by parametrized test with FakeLLM/FakeVectorStore configured to raise specific exceptions); happy path returns the expected JSON shape; truncated article surfaces `truncated: true`. |
| `test_api_chat.py` | Happy path returns `{answer, sources}` shape; A12 → 409 with `no_active_article`; retrieval_empty → 422; ollama errors → 503/504; sources array shape matches F8. |
| `test_errors.py` | The central exception handler in `api/errors.py` maps every domain exception class to the correct (status, code, message). One row per F9 case. |

### 7.2 Integration test

Two files under `tests/integration/`, both marked `@pytest.mark.integration` and excluded from the default `pytest` invocation. The default unit run stays fast (no model loads, no network).

**`tests/integration/test_embedder_smoke.py`.** Loads the real `BAAI/bge-small-en-v1.5` via `sentence-transformers`. Asserts `vector_size == 384`, that two distinct texts produce two distinct (cosine < 1.0) vectors, and that `embed_texts(["the","cat","sat"])` returns three length-384 vectors of approximately unit magnitude. ~5–10 s on first model load (cached on disk), ~1 s on subsequent runs. This is the smoke test that confirms the abstraction binds to the real model correctly; the unit suite verifies the wrapping logic separately with the SBERT class patched.

**`tests/integration/test_full_stack.py`.** Per your answer in clarifying-questions: the test **assumes the docker stack is already running** (no compose plumbing inside pytest). It reads `QDRANT_URL` and `OLLAMA_BASE_URL` from env (defaulting to `http://localhost:6333` and `http://localhost:11434` for the case where compose-mapped ports are in use). The README documents the two-step:

```
docker compose up -d
pytest tests/integration -v
```

The test mocks **only** the Wikipedia HTTP fetch (via `httpx.MockTransport` injected into the fetcher fixture). It serves the cached `tests/fixtures/photosynthesis_rest_api.json` (cached REST API JSON, not raw HTML — see §10 delta). Real Qdrant, real Ollama, real embedder, real chunker, real orchestrator. The test does:

1. `POST /api/ingest { url: "https://en.wikipedia.org/wiki/Photosynthesis" }`. Assert 200 with non-empty summary and `chunk_count > 30`.
2. `POST /api/chat { question: "What pigment captures light in chloroplasts?" }`. Assert 200 with a non-empty answer; assert one of the `sources[].section_title` is `Light-dependent reactions` or `Pigments`; assert `len(sources) == 4`.
3. **F7 grounding test.** `POST /api/chat { question: "Who wrote the play Hamlet?" }`. Assert the answer contains the literal phrase `not found in the article` (case-insensitive). Assert `sources` is still a 4-element array (U8/A17). The question is chosen so the model unambiguously **knows the answer parametrically** (Shakespeare is firmly in `llama3.2:3b`'s training set) but the answer is unambiguously **not in the Photosynthesis article** — so a passing test confirms the grounding *instruction* is doing the work, not the model's ignorance.

**Fixture coupling.** The assertions above are tied to the committed `photosynthesis_rest_api.json` fixture, not live Wikipedia. The fixture is a snapshot of the REST API response at a point in time; Wikipedia articles drift (sections renamed, content rewritten) and a live-fetched integration test would be non-deterministic. If the article is ever edited so that "Light-dependent reactions" or "Pigments" disappear, the test fails — and the right fix is to regenerate the fixture, not to relax the assertion. `README.md` will include a "Maintaining fixtures" section covering how to regenerate (`curl` the REST API endpoints, commit the JSON, re-run the integration test, update assertions if section titles changed). That README section is a Phase-4 deliverable; it's on the TASKS list, not in DESIGN's scope to write here.

**Runtime.** Cold-stack (model not loaded): ~90–150 s dominated by Ollama prefill. Warm-stack (model loaded, e.g. test run after a manual warmup): ~30–60 s. Documented in README. Not parallelisable with unit tests; runs under a dedicated pytest marker (`@pytest.mark.integration`) excluded from the default coverage run.

### 7.3 The F7 grounding test (named explicitly)

- **Article:** Photosynthesis (the integration fixture).
- **Out-of-corpus question:** "Who wrote the play Hamlet?"
- **Assertion:** `"not found in the article" in answer.lower()` and `len(sources) == 4` (sources are populated from the retriever regardless).

The question is chosen deliberately: Shakespeare-as-Hamlet's-author is canonical training data for any 3B+ LLM, so a passing test confirms the **grounding instruction** is what's making the model decline — not that it doesn't know. This is what F7 commits to: "we claim the model is instructed not to leak, and the instruction is tested." If the model leaks on this question, the test fails and we tighten the chat prompt template.

### 7.4 Coverage exclusions (OQ-10)

Configured in `pyproject.toml` under `[tool.coverage.run]` and `[tool.coverage.report]`:

```
omit =
    backend/app/main.py              # uvicorn entrypoint, lifespan glue
    backend/app/api/__init__.py      # package init
    backend/app/**/__init__.py       # package inits
    backend/app/config.py            # pydantic-settings boilerplate
    backend/app/domain/models.py     # pure dataclasses
    tests/*
exclude_lines =
    pragma: no cover
    if TYPE_CHECKING:
    raise NotImplementedError
    \.\.\.                           # Protocol method stubs
```

`main.py` is excluded because exercising the uvicorn entrypoint requires spinning the real server, which is what the integration test does (uncounted). `domain/models.py` is excluded because frozen dataclasses with no logic don't need behavioural tests. The Protocol stubs (`...`) are excluded because they're definitions, not code.

These exclusions are called out in the README per NF5/NF9.

**Coverage artefact committed (A15 disambiguation).** A15 allows either an HTML report folder *or* a screenshot of the summary line. We commit the **HTML folder** (`coverage_report/`) — it's more reviewable, it contains the per-file breakdown the reviewer needs to see the exclusion list applied, and it's what A15 lists first. The screenshot path is the cheap fallback if reviewers complain about repo size; `coverage_report/` is ~1–2 MB for a project this size, which is fine.

---

## 8. Module → file layout

```
silver-spoon2/
├── REQUIREMENTS.md
├── DESIGN.md
├── TASKS.md
├── README.md
├── .env.example
├── .gitignore
├── .gitattributes
├── docker-compose.yml
├── Dockerfile                       # multi-stage: node build → python runtime
├── pyproject.toml                   # poetry/uv config + coverage + ruff
├── coverage_report/                 # generated, committed per A15
│
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py                  # FastAPI app, lifespan, static mount, exception handler
│       ├── config.py                # pydantic-settings (every env var in §6.4)
│       ├── api/
│       │   ├── __init__.py
│       │   ├── ingest.py            # POST /api/ingest
│       │   ├── chat.py              # POST /api/chat
│       │   ├── health.py            # GET /api/health
│       │   ├── errors.py            # DomainError hierarchy, FastAPI handler, F9 mapping
│       │   └── deps.py              # Depends(...) providers for orchestrator/summariser/etc.
│       ├── domain/
│       │   ├── __init__.py
│       │   └── models.py            # frozen dataclasses from §3
│       ├── wikipedia/
│       │   ├── __init__.py
│       │   ├── validator.py         # validate_url (§3, §4.9)
│       │   ├── fetcher.py           # WikipediaFetcher
│       │   └── cleaner.py           # ArticleCleaner
│       ├── chunking/
│       │   ├── __init__.py
│       │   ├── chunker.py           # Chunker
│       │   └── tokenizer.py         # thin wrapper over SBERT's tokenizer
│       ├── embedding/
│       │   ├── __init__.py
│       │   ├── base.py              # Embedder Protocol
│       │   └── sbert.py             # SentenceTransformersEmbedder
│       ├── vectorstore/
│       │   ├── __init__.py
│       │   ├── base.py              # VectorStore Protocol
│       │   └── qdrant_store.py      # QdrantVectorStore
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── base.py              # LLMClient Protocol
│       │   ├── ollama.py            # OllamaClient
│       │   └── prompts.py           # SUMMARY_SHORT, SUMMARY_LONG, CHAT_GROUNDED
│       └── rag/
│           ├── __init__.py
│           ├── summariser.py        # Summariser
│           └── orchestrator.py      # RAGOrchestrator
│
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── components/
│       │   ├── UrlInput.tsx
│       │   ├── LoadingState.tsx     # staged labels per §6.2
│       │   ├── SummaryPanel.tsx     # renders summary + truncation flag
│       │   ├── ChatPanel.tsx
│       │   ├── ChatTranscript.tsx   # client-side transcript (A14)
│       │   ├── SourceList.tsx       # renders F8 sources as a collapsible list
│       │   └── ErrorBanner.tsx      # renders F9 messages verbatim
│       ├── api/
│       │   ├── client.ts            # fetch() wrappers calling /api/*
│       │   └── types.ts             # TS mirrors of backend response shapes
│       └── styles/
│           └── app.css
│
└── tests/
    ├── conftest.py                  # session-scoped embedder load, env helpers
    ├── fakes.py                     # FakeLLM, FakeVectorStore, FakeEmbedder
    ├── fixtures/
    │   ├── photosynthesis_rest_api.json     # cached REST API response
    │   └── short_article_rest_api.json      # for cleaner truncation/min-chars tests
    ├── unit/
    │   ├── test_url_validator.py
    │   ├── test_wikipedia_fetcher.py
    │   ├── test_article_cleaner.py
    │   ├── test_chunker.py
    │   ├── test_embedder.py
    │   ├── test_qdrant_store.py
    │   ├── test_ollama_client.py
    │   ├── test_summariser.py
    │   ├── test_orchestrator.py
    │   ├── test_api_ingest.py
    │   ├── test_api_chat.py
    │   └── test_errors.py
    └── integration/
        ├── conftest.py              # env probing for QDRANT_URL/OLLAMA_BASE_URL
        ├── test_embedder_smoke.py   # real SBERT model smoke (§7.2)
        └── test_full_stack.py       # the OQ-11 end-to-end test (§7.2)
```

Backend is a flat Python package. Each component is its own module; the package boundary is the import contract. The `Protocol` files are paired one-per-package so the abstraction lives next to its only production implementation but isn't entangled with it.

---

## 9. Remaining design decisions to challenge before TASKS.md

No remaining challenges. The three top-tier challenges raised in the initial DESIGN draft (summary threshold, warmup mechanism, embedder sync/async) and the two lower-tier challenges raised in review revision 1 (warmup timeout value, chunk_index scoping) are all resolved. See §11 for the full review trail.

---

## 10. Deltas / amendments to REQUIREMENTS.md

Two things I changed or clarified during design. The first is an amendment I made in REQUIREMENTS.md during this session (per your direction). The second is a clarification that doesn't change REQUIREMENTS — but I want it on the record so a future reader doesn't see a conflict.

1. **F9 amended (REQUIREMENTS.md, this session).** Added `malformed or empty LLM response` (error code `llm_response_invalid`), `Ollama timeout` (`llm_timeout`), `article exceeding length cap` (handled as a truncation flag, not an error), and `chat called before any ingest` (`no_active_article`) to F9's enumeration. The original F9 listed only a subset; the design needed the full list to write §5's error mapping table.

2. **A8 wording clarification (not a REQUIREMENTS change).** A8 describes the integration test as running against a "cached fixture HTML for the Wikipedia article". The design uses a cached **REST API response (JSON)**, not raw HTML, because F3 says we fetch via the REST API — so the fixture has to be at the REST-API layer, not the HTML layer, or the integration test wouldn't exercise the real `WikipediaFetcher` + `ArticleCleaner` code paths. Same end-to-end coverage, different file format. If you want A8 amended to match, say the word; otherwise this DESIGN entry is the record.

3. **A4 wording clarification.** A4 says "first ~30 tokens (or first sentence) of each section". The design picks "first sentence, capped at `SUMMARY_PER_SECTION_TOKENS=30`" — i.e., the *intersection* of the two, not a disjunction. This is what §4.7 implements. Mention it here so the disjunction-vs-conjunction reading doesn't surprise anyone.

---

## 11. Changelog

This is where post-freeze design changes go (per REQUIREMENTS.md's frozen-doc rule).

- **2026-05-25** — DESIGN.md initial draft. REQUIREMENTS.md F9 amended (see §10 delta 1).
- **2026-05-25 (review revision 1)** — Eight changes applied after the first review pass:
  1. **`SUMMARY_LONG_THRESHOLD_TOKENS` lowered from 5000 to 3500** (§4.7, §6.4, §3 Summariser). Resolves the original §9.1 challenge. Tightens the F10 latency budget; *Photosynthesis* now exercises the outline path in the integration test, which is the right testing posture.
  2. **Warmup switched from `asyncio.create_task` to `await asyncio.wait_for(..., timeout=30.0)`** (§4.2, §1 FastAPI app). Resolves the original §9.2 challenge. Reinterprets A16's "best-effort": failure doesn't crash startup, but failure is no longer invisible. Added `WARMUP_TIMEOUT_S` to §6.4.
  3. **Sync Embedder Protocol confirmed** (§3, §6.1). Resolves the original §9.3 challenge with no change. Decision rationale stands: making the CPU-bound nature visible at the call site is a feature, not a leak; promoting the abstraction to async would bake in an executor assumption.
  4. **Ingest UI: dropped wall-clock staged labels and cancel button** (§6.2, §1 frontend). Replaced with a single indeterminate spinner, one milestone-driven label set on submit, and a cold-start sub-label gated on `/api/health.warmup_ok`. Real per-phase progress via SSE parked in `NOTES.md` as future work.
  5. **Unit suite no longer loads the real SBERT model** (§7.1, §7.2, §8 file tree). `test_embedder.py` now patches `sentence_transformers.SentenceTransformer` to a stub; the real-model smoke moved to `tests/integration/test_embedder_smoke.py` under the `integration` marker. `test_chunker.py` uses a fake tokenizer so it doesn't pull the embedder in.
  6. **Integration test fixture coupling documented** (§7.2). Assertions are tied to the committed `photosynthesis_rest_api.json` snapshot, not live Wikipedia. Drift in the live article will fail the test; fix is to regenerate the fixture. README will carry a "Maintaining fixtures" section (Phase-4 deliverable, on TASKS).
  7. **`/page/mobile-sections/` deprecation note** added to §4.10 and to the `WikipediaFetcher` docstring in §3. Migration to Parsoid-backed endpoints parked in `NOTES.md`.
  8. **Cleanup pass**: (a) §3 `validate_url` docstring rewritten to match the §4.9 check-order table step-for-step; (b) §6.3 memory table now has idle/peak columns and the rollup explicitly sums the peak column; (c) §7.4 disambiguates A15 by committing the HTML report folder (`coverage_report/`); (d) `num_ctx` default removed from `LLMClient.generate` and `RAGOrchestrator.__init__` — both now require it, so the per-call-type config wiring (`CHAT_NUM_CTX` vs `SUMMARY_NUM_CTX`) is explicit at call sites and never silently hardcoded.
  9. **F7 grounding-test question changed** from "Who won the 2024 FIFA World Cup?" to "Who wrote the play Hamlet?" (§7.2, §7.3). My own pickup during the revision: the original question is one the model might decline because it doesn't know (no 2024 World Cup exists; tournament is 2022 / 2026), which would pass the test for the wrong reason. The new question is canonical parametric knowledge that is unambiguously absent from *Photosynthesis* — so the test actually exercises the grounding instruction.
- **2026-05-25 (review revision 2)** — Both residual challenges from revision 1 resolved:
  1. **`WARMUP_TIMEOUT_S` stays at 30 s** (§4.2). The rare tail case where a slow host trips this is correctly handled by the existing graceful degradation: log warning, set `warmup_ok=False`, app still serves. Bumping to 60 s would slow startup on every cold compose-up to chase a low-frequency failure mode that's already harmless.
  2. **`Chunk.chunk_index` stays global** (§3, §4.5). Single stable identifier beats per-section readability; `section_title` already provides the human-readable context in the F8 sources payload. Decisive secondary argument surfaced during resolution: `chunk_index` is also the Qdrant point id (§3 `VectorStore.upsert` notes "point's id is `chunk.chunk_index`"), giving a clean 1:1 mapping between the citation identifier and the vector-store record. Per-section `(section_index, chunk_index_in_section)` tuples would break that mapping or require synthesising a composite point id, both worse than what we have.

  §9 now empty.
