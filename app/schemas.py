"""
All request/response shapes live here so the API contract is explicit
and the LLM's JSON output can be validated against the same models.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    document: str = Field(..., description="Source document filename, e.g. leave-policy.md")
    section: str = Field(..., description="Section heading path, e.g. '4. Carry-forward > 4.1 Casual leave carry-forward'")
    chunk_id: Optional[int] = Field(None, description="Internal chunk id, useful for debugging/eval, not shown to end users")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    sufficient: bool
    # Debug/eval aid: not required by the spec, but cheap and useful.
    retrieved_chunk_ids: List[int] = []


class DocumentInfo(BaseModel):
    id: int
    filename: str
    uploaded_at: str
    num_chunks: int


class LLMAnswer(BaseModel):
    """
    This is the exact JSON shape we force the LLM to produce.
    Kept separate from QueryResponse because the LLM only knows about
    chunk *labels* (e.g. "C3"), not internal DB ids -- we translate
    labels -> Citation objects in rag.py after validation.
    """
    answer: str
    used_chunk_labels: List[str] = []
    sufficient: bool
