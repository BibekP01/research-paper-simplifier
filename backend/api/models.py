"""
Pydantic models for API request/response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class QARequest(BaseModel):
    """Request model for question answering."""
    question: str = Field(..., min_length=1, description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of sources to retrieve")
    min_score: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity score")


class Source(BaseModel):
    """Source information model."""
    rank: int
    paper_id: str
    paper_title: str
    section: str
    similarity_score: float
    text_preview: str


class QAResponse(BaseModel):
    """Response model for question answering."""
    question: str
    answer: str
    sources: List[Source]
    num_sources: int
    model_used: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    index_loaded: bool
    model_loaded: bool
    message: Optional[str] = None
