import type { Source } from "../api/types";
import { SourceList } from "./SourceList";

export interface ChatEntry {
  question: string;
  answer: string;
  sources: Source[];
}

interface ChatTranscriptProps {
  entries: ChatEntry[];
}

export function ChatTranscript({
  entries,
}: ChatTranscriptProps): JSX.Element | null {
  if (entries.length === 0) return null;
  return (
    <ol className="transcript" aria-label="Chat history">
      {entries.map((entry, i) => (
        <li key={i} className="transcript__entry">
          <div className="transcript__question">
            <span className="transcript__role">You</span>
            <p className="transcript__text">{entry.question}</p>
          </div>
          <div className="transcript__answer">
            <span className="transcript__role">Answer</span>
            <p className="transcript__text">{entry.answer}</p>
            <SourceList sources={entry.sources} />
          </div>
        </li>
      ))}
    </ol>
  );
}
