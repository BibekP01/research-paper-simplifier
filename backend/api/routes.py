"""
FastAPI routes for the Research Paper Q&A API.
"""
import logging
from fastapi import APIRouter, HTTPException
from .models import QARequest, QAResponse, HealthResponse, Source, UploadResponse

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


# --- New Endpoints for Chat with Paper ---

import shutil
import os
from pathlib import Path
from fastapi import UploadFile, File, Form
from backend.src.document_processor import DocumentProcessor
from backend.src.chunker import TextChunker
from backend.src.retriever import VectorStoreRetriever
from backend.src.qa_pipeline import QAPipeline

# Directory for uploads and vector stores
UPLOAD_DIR = Path("data/uploads")
PROCESSED_DIR = Path("data/processed")
VECTORSTORE_DIR = Path("data/vectorstore")

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=UploadResponse)
def upload_file(file: UploadFile = File(...)):
    """
    Upload a file (PDF/DOCX), process it, and create a vector index.
    """
    try:
        logger.info(f"Starting upload for file: {file.filename}")
        
        # 1. Save the file
        doc_id = file.filename.split('.')[0] # Simple doc_id from filename
        # Make doc_id safe
        doc_id = "".join([c if c.isalnum() else "_" for c in doc_id])
        
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"File saved to {file_path}")
        
        # 2. Process the document
        logger.info("Starting document processing (text extraction)...")
        processor = DocumentProcessor(upload_dir=str(UPLOAD_DIR), processed_dir=str(PROCESSED_DIR))
        processed_paper = processor.process_file(file_path)
        logger.info(f"Document processed. Extracted {len(processed_paper.full_text)} characters.")
        
        # Save processed text
        json_path, _ = processor.pdf_processor.save_processed_text(processed_paper)
        
        # 3. Chunk the document
        logger.info("Starting text chunking...")
        chunker = TextChunker()
        chunks = chunker.chunk_paper(json_path)
        logger.info(f"Document chunked into {len(chunks)} chunks.")
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")
            
        # 4. Create Vector Index
        # Create a specific directory for this document's index
        doc_index_dir = VECTORSTORE_DIR / doc_id
        doc_index_dir.mkdir(exist_ok=True)
        
        retriever = VectorStoreRetriever(index_path=None) # Initialize empty
        
        # Prepare embeddings (this might take a while for large docs)
        # We need to extract text from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks (this may take a while)...")
        embeddings = retriever.embedding_model.encode(texts, show_progress_bar=False)
        logger.info("Embeddings generated.")
        
        # Prepare metadata
        metadata = [chunk for chunk in chunks]
        
        # Build and save index
        retriever.build_index(embeddings, metadata)
        retriever.save_index(str(doc_index_dir / "index"))
        logger.info("Index built and saved.")
        
        return UploadResponse(
            document_id=doc_id,
            filename=file.filename,
            message="File processed and indexed successfully",
            metadata=processed_paper.metadata
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/chat/{document_id}", response_model=QAResponse)
async def chat_with_document(document_id: str, request: QARequest):
    """
    Chat with a specific document.
    """
    try:
        doc_index_dir = VECTORSTORE_DIR / document_id
        index_path = doc_index_dir / "index"
        
        if not index_path.with_suffix('.faiss').exists():
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found or not indexed.")
            
        # Initialize a temporary pipeline for this document
        # We reuse the global pipeline's LLM if available to save resources, 
        # but we need a specific retriever.
        
        # Check if we can reuse the global LLM config
        api_key = os.getenv("GOOGLE_API_KEY")
        
        # Create a new pipeline instance
        # We pass the specific index path
        # Note: QAPipeline loads index in __init__. 
        # This might be inefficient to do per request if index is huge, 
        # but for individual papers it's likely fine (fast load).
        
        doc_pipeline = QAPipeline(
            index_path=str(index_path),
            api_key=api_key
        )
        
        # Ask the question
        result = doc_pipeline.ask(
            question=request.question,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        # Format response
        sources = [Source(**source) for source in result.get("sources", [])]
        
        return QAResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
            num_sources=result["num_sources"],
            model_used=result["model_used"],
            timestamp=result["timestamp"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

