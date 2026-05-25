import type { Source } from "../api/types";

interface SourceListProps {
  sources: Source[];
}

// F8: each source carries section_title, chunk_index, score, excerpt. All
// three of {section_title, excerpt, score} must be reachable without dev
// tools (T14 smoke item 10). Collapsed-by-default keeps the transcript
// readable; <details>/<summary> is the cheapest a11y-correct disclosure.
export function SourceList({ sources }: SourceListProps): JSX.Element | null {
  if (sources.length === 0) return null;
  return (
    <details className="sources">
      <summary className="sources__summary">
        {sources.length} source{sources.length === 1 ? "" : "s"}
      </summary>
      <ol className="sources__list">
        {sources.map((s) => (
          <li key={s.chunk_index} className="sources__item">
            <div className="sources__item-head">
              <span className="sources__title">{s.section_title}</span>
              <span className="sources__score">score {s.score.toFixed(3)}</span>
            </div>
            <p className="sources__excerpt">{s.excerpt}</p>
          </li>
        ))}
      </ol>
    </details>
  );
}
