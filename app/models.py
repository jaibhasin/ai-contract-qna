from __future__ import annotations

from typing import List, Optional, Union
from pydantic import BaseModel, Field, HttpUrl


class RunRequest(BaseModel):
    """Request schema for /api/v1/hackrx/run.

    documents can be a single URL or a list of URLs.
    questions is a list of user questions to answer using RAG over the docs.
    """

    documents: Union[str, List[str]] = Field(..., description="Blob URL or list of URLs to ingest")
    questions: List[str] = Field(..., min_items=1)


class Source(BaseModel):
    doc: str
    page: Optional[int] = None
    chunk_id: str
    score: float


class Answer(BaseModel):
    question: str
    answer: str
    conditions: List[str]
    sources: List[Source]
    reasoning: str
    confidence: float = Field(ge=0, le=1)


class RunMetadata(BaseModel):
    model: str
    time_ms: int
    token_notes: Optional[str] = None


class RunResponse(BaseModel):
    answers: List[Answer]
    metadata: RunMetadata
