"""
FastAPI routes for the Research Paper Q&A API.
"""
import logging
import json
from fastapi import APIRouter, HTTPException
from .models import QARequest, QAResponse, HealthResponse, Source, UploadResponse, SuggestedQuestionsResponse

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


# --- ArXiv API Endpoints ---

from .models import (
    ArxivSearchRequest, 
    ArxivSearchResponse, 
    ArxivPaperResponse, 
    ArxivFetchRequest
)
from backend.src.paper_fetcher import PaperFetcher
import arxiv as arxiv_lib


@router.post("/arxiv/search", response_model=ArxivSearchResponse)
async def search_arxiv_papers(request: ArxivSearchRequest):
    """
    Search for papers on arXiv.
    
    Args:
        request: ArxivSearchRequest with query, max_results, and sort_by
        
    Returns:
        ArxivSearchResponse with list of papers
    """
    try:
        logger.info(f"Searching arXiv for: '{request.query}' (max_results={request.max_results})")
        
        # Map sort_by string to arxiv.SortCriterion
        sort_map = {
            "relevance": arxiv_lib.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv_lib.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv_lib.SortCriterion.SubmittedDate
        }
        sort_criterion = sort_map.get(request.sort_by, arxiv_lib.SortCriterion.Relevance)
        
        # Initialize PaperFetcher with correct config path
        fetcher = PaperFetcher(config_path="backend/config/config.yaml")
        
        # Search papers
        papers_data = fetcher.search_papers(
            query=request.query,
            max_results=request.max_results,
            sort_by=sort_criterion
        )
        
        # Convert to response models
        papers = [ArxivPaperResponse(**paper) for paper in papers_data]
        
        logger.info(f"Found {len(papers)} papers for query: '{request.query}'")
        
        return ArxivSearchResponse(
            papers=papers,
            total_results=len(papers),
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"ArXiv search failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ArXiv search failed: {str(e)}")


@router.post("/arxiv/fetch", response_model=UploadResponse)
async def fetch_arxiv_paper(request: ArxivFetchRequest):
    """
    Fetch an arXiv paper by ID, download PDF, process it, and create vector index.
    
    Args:
        request: ArxivFetchRequest with arxiv_id
        
    Returns:
        UploadResponse with document_id and metadata
    """
    try:
        logger.info(f"Fetching arXiv paper: {request.arxiv_id}")
        
        # Initialize PaperFetcher with correct config path
        fetcher = PaperFetcher(config_path="backend/config/config.yaml")
        
        # Fetch paper metadata
        paper_data = fetcher.fetch_by_id(request.arxiv_id)
        logger.info(f"Fetched metadata for: {paper_data['title']}")
        
        # Download PDF
        pdf_path = fetcher.download_pdf(
            paper_data['pdf_url'],
            arxiv_id=paper_data['arxiv_id']
        )
        logger.info(f"Downloaded PDF to: {pdf_path}")
        
        # Copy PDF to uploads directory so it can be served
        uploaded_pdf_path = UPLOAD_DIR / pdf_path.name
        shutil.copy2(pdf_path, uploaded_pdf_path)
        logger.info(f"Copied PDF to uploads directory: {uploaded_pdf_path}")
        
        # Create document ID from arxiv_id
        doc_id = "".join([c if c.isalnum() else "_" for c in paper_data['arxiv_id']])
        
        # Process the document (same as upload flow)
        logger.info("Starting document processing...")
        processor = DocumentProcessor(upload_dir=str(UPLOAD_DIR), processed_dir=str(PROCESSED_DIR))
        processed_paper = processor.process_file(uploaded_pdf_path)
        logger.info(f"Document processed. Extracted {len(processed_paper.full_text)} characters.")
        
        # Save processed text
        json_path, _ = processor.pdf_processor.save_processed_text(processed_paper)
        
        # Chunk the document
        logger.info("Starting text chunking...")
        chunker = TextChunker()
        chunks = chunker.chunk_paper(json_path)
        logger.info(f"Document chunked into {len(chunks)} chunks.")
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from the arXiv paper.")
        
        # Create Vector Index
        doc_index_dir = VECTORSTORE_DIR / doc_id
        doc_index_dir.mkdir(exist_ok=True)
        
        retriever = VectorStoreRetriever(index_path=None)
        
        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
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
            filename=f"{paper_data['arxiv_id']}.pdf",
            message="ArXiv paper fetched, processed, and indexed successfully",
            metadata={
                "arxiv_id": paper_data['arxiv_id'],
                "title": paper_data['title'],
                "authors": paper_data['authors'],
                "published_date": paper_data['published_date'],
                "abstract": paper_data['abstract'][:200] + "..."  # Truncate for response
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ArXiv fetch failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ArXiv fetch failed: {str(e)}")


@router.get("/arxiv/categories")
async def get_arxiv_categories():
    """
    Get list of common arXiv categories.
    
    Returns:
        Dictionary with category codes and names
    """
    categories = {
        "cs.AI": "Artificial Intelligence",
        "cs.LG": "Machine Learning",
        "cs.CL": "Computation and Language",
        "cs.CV": "Computer Vision and Pattern Recognition",
        "cs.NE": "Neural and Evolutionary Computing",
        "stat.ML": "Machine Learning (Statistics)",
        "math.OC": "Optimization and Control",
        "cs.RO": "Robotics",
        "cs.CR": "Cryptography and Security",
        "cs.DB": "Databases",
        "cs.SE": "Software Engineering",
        "cs.DC": "Distributed, Parallel, and Cluster Computing"
    }
    
    return {"categories": categories}


@router.get("/document/{document_id}/suggestions", response_model=SuggestedQuestionsResponse)
async def get_suggested_questions(document_id: str):
    """
    Generate suggested questions for a specific document using Gemini.
    
    Args:
        document_id: ID of the document
        
    Returns:
        SuggestedQuestionsResponse with list of suggested questions
    """
    try:
        logger.info(f"Generating suggested questions for document: {document_id}")
        
        # Try to find the JSON file - the document_id might not match the JSON filename exactly
        # Try multiple patterns:
        # 1. Exact match: document_id.json
        # 2. First part before underscore (for arXiv papers like "2501_02842v1" -> "2501.json")
        # 3. Replace underscores with dots
        
        json_path = None
        possible_patterns = [
            document_id,  # Exact match
            document_id.split('_')[0],  # First part (e.g., "2501" from "2501_02842v1")
            document_id.replace('_', '.'),  # Replace underscores with dots
        ]
        
        for pattern in possible_patterns:
            test_path = PROCESSED_DIR / f"{pattern}.json"
            if test_path.exists():
                json_path = test_path
                logger.info(f"Found JSON file: {json_path}")
                break
        
        if json_path is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Read the processed JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        # Extract title and abstract (first ~1000 chars for better context)
        title = doc_data.get('title', 'Unknown')
        full_text = doc_data.get('full_text', '')
        # Use more text for better context
        abstract = full_text[:1000] if len(full_text) > 1000 else full_text
        
        # Use Gemini to generate questions
        from backend.src.gemini_client import GeminiClient
        
        gemini = GeminiClient()
        
        prompt = f"""Based on this research paper, generate 4 short, specific questions.

Title: {title}

Content: {abstract}

Requirements:
- Each question should be 10-20 words maximum
- Ask about SPECIFIC concepts mentioned in the text
- Do NOT use generic phrases
- Return exactly 4 questions, one per line
- No numbering or bullets

Questions:"""

        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        response = gemini.model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 1200,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # Log finish reason for debugging
        if hasattr(response, 'candidates') and response.candidates:
            finish_reason = response.candidates[0].finish_reason
            logger.info(f"Gemini finish reason: {finish_reason}")
        
        # Parse the response into individual questions
        response_text = response.text.strip()
        logger.info(f"Raw Gemini response: {response_text}")
        
        # Try multiple parsing strategies
        questions = []
        
        # First, try splitting by newlines
        lines = [q.strip() for q in response_text.split('\n') if q.strip()]
        
        # Filter out lines that are just numbers, bullets, or dashes
        for line in lines:
            # Remove leading numbers, bullets, dashes
            cleaned = line.lstrip('0123456789.-•* ').strip()
            if cleaned and len(cleaned) > 10:  # Must be at least 10 chars to be a real question
                questions.append(cleaned)
        
        logger.info(f"Parsed {len(questions)} questions from response")
        
        # Ensure we have exactly 4 questions
        if len(questions) < 4:
            # Fallback to generic questions if generation failed
            logger.warning(f"Only generated {len(questions)} questions, using fallback")
            questions = [
                f"What is the main contribution of this paper?",
                f"What methodology does this paper use?",
                f"What are the key findings or results?",
                f"What are the limitations mentioned in this paper?"
            ]
        else:
            questions = questions[:4]  # Take only first 4
        
        logger.info(f"Generated {len(questions)} questions for document {document_id}")
        logger.info(f"Questions: {questions}")
        
        return SuggestedQuestionsResponse(
            questions=questions,
            document_id=document_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate suggestions: {str(e)}", exc_info=True)
        # Return generic fallback questions instead of failing
        return SuggestedQuestionsResponse(
            questions=[
                "What is the main contribution of this paper?",
                "What methodology does this paper use?",
                "What are the key findings or results?",
                "What are the limitations mentioned in this paper?"
            ],
            document_id=document_id
        )


