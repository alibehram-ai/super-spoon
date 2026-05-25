// Response shapes mirrored from backend/app/api/ingest.py + chat.py
// (which in turn mirror backend/app/domain/models.py). Kept in sync by hand —
// the backend is the source of truth; this file is a typed view of it.

export interface IngestResponse {
  summary: string;
  section_count: number;
  chunk_count: number;
  truncated: boolean;
  char_count: number;
}

export interface Source {
  section_title: string;
  chunk_index: number;
  score: number;
  excerpt: string;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
}

// Error envelope from backend/app/api/errors.py: { error, message } with
// HTTP status carrying the F9 case. F9 requires the frontend to render
// `message` verbatim.
export interface ApiError {
  error: string;
  message: string;
}

export interface HealthResponse {
  status: string;
  warmup_ok: boolean;
}
