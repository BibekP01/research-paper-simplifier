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
    
    
class UploadResponse(BaseModel):
    """Response model for file upload."""
    document_id: str
    filename: str
    message: str
    metadata: dict


# --- ArXiv API Models ---

class ArxivSearchRequest(BaseModel):
    """Request model for arXiv paper search."""
    query: str = Field(..., min_length=1, description="Search query for arXiv papers")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum number of results to return")
    sort_by: str = Field(default="relevance", description="Sort criterion: relevance, lastUpdatedDate, submittedDate")


class ArxivPaperResponse(BaseModel):
    """Response model for a single arXiv paper."""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published_date: str
    updated_date: str
    pdf_url: str
    categories: List[str]
    primary_category: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None


class ArxivSearchResponse(BaseModel):
    """Response model for arXiv search results."""
    papers: List[ArxivPaperResponse]
    total_results: int
    query: str


class ArxivFetchRequest(BaseModel):
    """Request model for fetching and processing an arXiv paper."""
    arxiv_id: str = Field(..., min_length=1, description="ArXiv ID of the paper to fetch")


class SuggestedQuestionsResponse(BaseModel):
    """Response model for suggested questions about a document."""
    questions: List[str] = Field(..., description="List of suggested questions")
    document_id: str = Field(..., description="ID of the document")

