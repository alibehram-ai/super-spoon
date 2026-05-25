import type {
  ApiError,
  ChatResponse,
  HealthResponse,
  IngestResponse,
} from "./types";

// All requests go to the same origin under /api/* (DESIGN §4.8). Vite proxies
// to the bare-metal FastAPI in dev; in prod the FastAPI app serves the SPA
// itself, so there's no cross-origin concern.

export class ApiClientError extends Error {
  readonly status: number;
  readonly code: string;
  // F9 contract: the user-facing string from the backend's structured payload.
  // Rendered verbatim by ErrorBanner — no client-side transformation.
  readonly userMessage: string;

  constructor(status: number, code: string, userMessage: string) {
    super(`${code} (${status}): ${userMessage}`);
    this.status = status;
    this.code = code;
    this.userMessage = userMessage;
  }
}

async function postJson<TResponse>(
  path: string,
  body: unknown,
): Promise<TResponse> {
  let res: Response;
  try {
    res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    // Network / DNS / CORS failures land here. The backend never spoke, so
    // we can't quote its message — synthesise a clean user-facing one
    // instead of leaking "TypeError: Failed to fetch" into the UI.
    throw new ApiClientError(
      0,
      "network_error",
      "Couldn't reach the backend. Is it running?",
    );
  }

  if (res.ok) {
    return (await res.json()) as TResponse;
  }

  // Non-2xx: try to parse the F9 envelope. If the body isn't JSON or doesn't
  // match shape, fall back to a generic message that still includes the
  // status — better than a blank banner.
  let payload: Partial<ApiError> | null = null;
  try {
    payload = (await res.json()) as Partial<ApiError>;
  } catch {
    payload = null;
  }
  const code = payload?.error ?? "unknown_error";
  const message =
    payload?.message ?? `Request failed with status ${res.status}.`;
  throw new ApiClientError(res.status, code, message);
}

export async function ingest(url: string): Promise<IngestResponse> {
  return postJson<IngestResponse>("/api/ingest", { url });
}

export async function chat(question: string): Promise<ChatResponse> {
  return postJson<ChatResponse>("/api/chat", { question });
}

export async function health(): Promise<HealthResponse> {
  const res = await fetch("/api/health");
  if (!res.ok) {
    throw new ApiClientError(
      res.status,
      "health_failed",
      `Health check failed (${res.status}).`,
    );
  }
  return (await res.json()) as HealthResponse;
}
