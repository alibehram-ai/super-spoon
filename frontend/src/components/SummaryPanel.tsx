interface SummaryPanelProps {
  summary: string;
  chunkCount: number;
  sectionCount: number;
  charCount: number;
  truncated: boolean;
}

export function SummaryPanel({
  summary,
  chunkCount,
  sectionCount,
  charCount,
  truncated,
}: SummaryPanelProps): JSX.Element {
  return (
    <section className="summary" aria-label="Article summary">
      {truncated && (
        <div className="summary__truncated" role="note">
          Article was truncated to fit. The summary reflects the truncated body.
        </div>
      )}
      <h2 className="summary__heading">Summary</h2>
      <p className="summary__body">{summary}</p>
      <div className="summary__meta">
        {sectionCount} section{sectionCount === 1 ? "" : "s"} · {chunkCount}{" "}
        chunk{chunkCount === 1 ? "" : "s"} · {charCount.toLocaleString()}{" "}
        characters indexed
      </div>
    </section>
  );
}
