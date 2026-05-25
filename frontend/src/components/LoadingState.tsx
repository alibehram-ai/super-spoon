interface LoadingStateProps {
  // DESIGN §6.2: cold-start sub-label is shown only when the boot-time
  // /api/health probe reported warmup_ok=false. Once we know the model is
  // warm, no need to apologise.
  warmupOk: boolean | null;
}

// Single indeterminate spinner, one milestone label, optional cold-start
// sub-label. No wall-clock fake phase progression (DESIGN §6.2 — the earlier
// draft did this and it was dishonest). No cancel button — the backend has
// no real cancellation path, so a button that lies is worse than no button.
export function LoadingState({ warmupOk }: LoadingStateProps): JSX.Element {
  const showColdStart = warmupOk === false;
  return (
    <div className="loading" role="status" aria-live="polite">
      <div className="loading__spinner" aria-hidden="true" />
      <div>
        <div className="loading__label">Processing article…</div>
        {showColdStart && (
          <div className="loading__sublabel">
            First request after startup is slower while the model loads.
          </div>
        )}
      </div>
    </div>
  );
}
