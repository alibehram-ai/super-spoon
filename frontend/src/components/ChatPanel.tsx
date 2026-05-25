import { useState } from "react";

interface ChatPanelProps {
  disabled: boolean;
  loading: boolean;
  onSubmit: (question: string) => Promise<void>;
}

export function ChatPanel({
  disabled,
  loading,
  onSubmit,
}: ChatPanelProps): JSX.Element {
  const [value, setValue] = useState("");

  const handleSubmit = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled || loading) return;
    setValue("");
    await onSubmit(trimmed);
  };

  // Cmd/Ctrl+Enter submits — friendlier for multi-line questions than
  // requiring the user to mouse over to the button.
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      void handleSubmit(e as unknown as React.FormEvent);
    }
  };

  const isBusy = disabled || loading;

  return (
    <form className="chat" onSubmit={handleSubmit} aria-label="Ask a question">
      <textarea
        className="chat__field"
        placeholder="Ask a question about the article…"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isBusy}
        rows={2}
      />
      <div className="chat__row">
        <span className="chat__hint">
          {loading ? "Thinking…" : "⌘/Ctrl+Enter to send"}
        </span>
        <button
          className="chat__submit"
          type="submit"
          disabled={isBusy || value.trim().length === 0}
        >
          Ask
        </button>
      </div>
    </form>
  );
}
