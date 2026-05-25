interface ErrorBannerProps {
  // F9 contract: the backend's structured `message` is rendered verbatim.
  // No transformation, no truncation, no translation.
  message: string;
}

export function ErrorBanner({ message }: ErrorBannerProps): JSX.Element {
  return (
    <div className="error-banner" role="alert">
      {message}
    </div>
  );
}
