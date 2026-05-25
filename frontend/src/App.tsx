import { useEffect, useState } from "react";
import { ApiClientError, health, ingest } from "./api/client";
import { ErrorBanner } from "./components/ErrorBanner";
import { LoadingState } from "./components/LoadingState";
import { SummaryPanel } from "./components/SummaryPanel";
import { UrlInput } from "./components/UrlInput";

// State lives in component state only (A14 / U10). No router, no localStorage,
// no global store — refresh wipes everything, by design.
export default function App(): JSX.Element {
  const [summary, setSummary] = useState<string | null>(null);
  const [sectionCount, setSectionCount] = useState(0);
  const [chunkCount, setChunkCount] = useState(0);
  const [charCount, setCharCount] = useState(0);
  const [truncated, setTruncated] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // null = probe hasn't returned yet (don't render the cold-start sub-label
  // optimistically); true/false = backend answered. Polled ONCE at boot.
  const [warmupOk, setWarmupOk] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    health()
      .then((r) => {
        if (!cancelled) setWarmupOk(r.warmup_ok);
      })
      .catch(() => {
        // Backend down at boot — treat as cold, so the user gets the
        // sub-label and isn't surprised by a slow first response if the
        // backend comes online later.
        if (!cancelled) setWarmupOk(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleIngest = async (url: string): Promise<void> => {
    setLoading(true);
    setError(null);
    setSummary(null);
    try {
      const result = await ingest(url);
      setSummary(result.summary);
      setSectionCount(result.section_count);
      setChunkCount(result.chunk_count);
      setCharCount(result.char_count);
      setTruncated(result.truncated);
    } catch (e) {
      if (e instanceof ApiClientError) {
        // F9: render the backend's user-facing message verbatim.
        setError(e.userMessage);
      } else {
        // Should be unreachable — fetch failures get wrapped by the client.
        setError("Something went wrong. Try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">silver-spoon2</h1>
        <p className="app__subtitle">
          Paste a Wikipedia article URL to ingest and summarise it locally.
        </p>
      </header>

      <UrlInput disabled={loading} onSubmit={handleIngest} />

      {error && <ErrorBanner message={error} />}
      {loading && <LoadingState warmupOk={warmupOk} />}

      {summary !== null && !loading && (
        <SummaryPanel
          summary={summary}
          sectionCount={sectionCount}
          chunkCount={chunkCount}
          charCount={charCount}
          truncated={truncated}
        />
      )}
    </div>
  );
}
