"""``POST /api/chat`` — DESIGN §2 chat flow.

Thin HTTP layer over ``RAGOrchestrator.answer``. The orchestrator (T09) owns
the whole flow — A12 gate, embed, retrieve, build prompt, call LLM, derive
sources from hits. This route just unmarshals the request, delegates, and
serialises the dataclass result onto the response model.

Errors flow through the central ``DomainError`` handler registered in
``main.py`` (T10a) — ``NoActiveArticleError`` (409), ``RetrievalEmptyError``
(422), ``OllamaUnreachableError`` (503), ``OllamaTimeoutError`` (504),
``LLMResponseInvalidError`` (502), and ``QdrantUnreachableError`` (503) all
land on their §5 (status, code) pair without per-route plumbing.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.app.api.deps import get_orchestrator
from backend.app.rag.orchestrator import RAGOrchestrator

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class SourceModel(BaseModel):
    section_title: str
    chunk_index: int
    score: float
    excerpt: str


class ChatResponseModel(BaseModel):
    answer: str
    sources: list[SourceModel]


@router.post("/api/chat", response_model=ChatResponseModel)
async def chat(
    payload: ChatRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator),
) -> ChatResponseModel:
    result = await orchestrator.answer(payload.question)
    return ChatResponseModel(
        answer=result.answer,
        sources=[
            SourceModel(
                section_title=s.section_title,
                chunk_index=s.chunk_index,
                score=s.score,
                excerpt=s.excerpt,
            )
            for s in result.sources
        ],
    )
