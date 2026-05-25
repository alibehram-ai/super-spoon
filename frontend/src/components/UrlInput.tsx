import { useState } from "react";

interface UrlInputProps {
  disabled: boolean;
  onSubmit: (url: string) => void;
}

// Cheap UX check so the user doesn't fire a network request for something
// that obviously isn't a URL (empty, whitespace, no http(s)). Anything past
// this gate goes to the backend — F2 validation lives there, and per F9 the
// backend's structured error message is what we render.
function looksObviouslyMalformed(raw: string): boolean {
  const trimmed = raw.trim();
  if (!trimmed) return true;
  return !/^https?:\/\/\S+/i.test(trimmed);
}

export function UrlInput({ disabled, onSubmit }: UrlInputProps): JSX.Element {
  const [value, setValue] = useState("");
  const [showHint, setShowHint] = useState(false);

  const handleSubmit = (e: React.FormEvent): void => {
    e.preventDefault();
    if (disabled) return;
    if (looksObviouslyMalformed(value)) {
      setShowHint(true);
      return;
    }
    setShowHint(false);
    onSubmit(value.trim());
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setValue(e.target.value);
    if (showHint) setShowHint(false);
  };

  return (
    <>
      <form className="url-input" onSubmit={handleSubmit}>
        <input
          className="url-input__field"
          type="text"
          placeholder="https://en.wikipedia.org/wiki/Photosynthesis"
          value={value}
          onChange={handleChange}
          disabled={disabled}
          autoFocus
          spellCheck={false}
          autoComplete="off"
        />
        <button
          className="url-input__submit"
          type="submit"
          disabled={disabled}
        >
          Ingest
        </button>
      </form>
      {showHint && (
        <div className="url-input__hint">
          Paste a full URL starting with https://.
        </div>
      )}
    </>
  );
}
