# syntax=docker/dockerfile:1.7

# ---------- Stage 1: frontend build ----------
FROM node:20-alpine AS frontend-build

WORKDIR /build/frontend

# Lockfile-aware install layer — invalidated only when package*.json changes.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy the rest of the frontend source and build the production bundle.
COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: Python runtime ----------
FROM python:3.11-slim AS runtime

# uv as a static binary from the official image — avoids pulling pip's
# transitive deps and caches into the runtime layer.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_PREFERENCE=only-system

# Dependency install layer — pyproject + lock first so source edits don't bust
# this cache. --no-dev skips the dev group; --frozen requires uv.lock to match.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application source.
COPY backend/ ./backend/

# Frontend build output, mounted by main.py at <repo>/static (== /app/static).
COPY --from=frontend-build /build/frontend/dist/ ./static/

EXPOSE 8000

# --host 0.0.0.0 is required — without it uvicorn binds to 127.0.0.1 inside the
# container and the host port mapping is unreachable.
CMD ["uv", "run", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
