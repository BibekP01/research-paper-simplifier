"""
FastAPI application for Research Paper Q&A API.

This API wraps the existing QA pipeline and provides REST endpoints
for the React frontend.
"""
import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Add project root to path if needed, but preferably use module imports
# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

from backend.src.qa_pipeline import QAPipeline
from .routes import router, set_qa_pipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup: Initialize QA pipeline
    logger.info("Starting up API server...")
    try:
        logger.info("Initializing QA Pipeline...")
        pipeline = QAPipeline()
        set_qa_pipeline(pipeline)
        logger.info("QA Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize QA Pipeline: {e}", exc_info=True)
        logger.warning("API will start but /api/ask endpoint will not work")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")


# Create FastAPI app
app = FastAPI(
    title="Research Paper Q&A API",
    description="API for asking questions about research papers using AI",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

# Mount static files for uploads
# Ensure directory exists
os.makedirs("data/uploads", exist_ok=True)
app.mount("/api/files", StaticFiles(directory="data/uploads"), name="uploads")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Research Paper Q&A API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "backend.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
