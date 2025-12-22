"""
FastAPI routes for the Research Paper Q&A API.
"""
import logging
from fastapi import APIRouter, HTTPException
from .models import QARequest, QAResponse, HealthResponse, Source

logger = logging.getLogger(__name__)

# Router instance
router = APIRouter(prefix="/api", tags=["qa"])

# Global QA pipeline instance (will be initialized in main.py)
qa_pipeline = None


def set_qa_pipeline(pipeline):
    """Set the global QA pipeline instance."""
    global qa_pipeline
    qa_pipeline = pipeline


@router.post("/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    """
    Ask a question about research papers.
    
    Args:
        request: QARequest containing question, top_k, and min_score
        
    Returns:
        QAResponse with answer and sources
        
    Raises:
        HTTPException: If QA pipeline is not initialized or processing fails
    """
    if qa_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="QA pipeline not initialized. Please check server logs."
        )
    
    try:
        logger.info(f"Processing question: {request.question[:50]}...")
        
        # Call the existing QA pipeline
        result = qa_pipeline.ask(
            question=request.question,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        # Convert sources to Pydantic models
        sources = [Source(**source) for source in result.get("sources", [])]
        
        # Create response
        response = QAResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
            num_sources=result["num_sources"],
            model_used=result["model_used"],
            timestamp=result["timestamp"]
        )
        
        logger.info(f"Successfully processed question with {len(sources)} sources")
        return response
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with system status
    """
    try:
        if qa_pipeline is None:
            return HealthResponse(
                status="unhealthy",
                index_loaded=False,
                model_loaded=False,
                message="QA pipeline not initialized"
            )
        
        # Check if retriever and LLM are loaded
        index_loaded = hasattr(qa_pipeline, 'retriever') and qa_pipeline.retriever is not None
        model_loaded = hasattr(qa_pipeline, 'llm') and qa_pipeline.llm is not None
        
        status = "healthy" if (index_loaded and model_loaded) else "degraded"
        
        return HealthResponse(
            status=status,
            index_loaded=index_loaded,
            model_loaded=model_loaded,
            message="System operational" if status == "healthy" else "Some components not loaded"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            index_loaded=False,
            model_loaded=False,
            message=f"Health check error: {str(e)}"
        )
