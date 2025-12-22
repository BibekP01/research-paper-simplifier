# Research Paper Simplifier - Backend API

## FastAPI Backend for React Frontend

This directory contains the FastAPI backend that wraps the existing QA pipeline and provides REST API endpoints for the React frontend.

## Installation

```bash
# Install API dependencies
pip install -r requirements-api.txt

# Make sure main dependencies are also installed
pip install -r ../requirements.txt
```

## Running the API

```bash
# From the backend/api directory
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/api/health

## API Endpoints

### POST /api/ask
Ask a question about research papers.

**Request:**
```json
{
  "question": "What is self-attention?",
  "top_k": 5,
  "min_score": 0.3
}
```

**Response:**
```json
{
  "question": "What is self-attention?",
  "answer": "Self-attention is...",
  "sources": [...],
  "num_sources": 5,
  "model_used": "gemini-2.5-flash",
  "timestamp": "2025-12-22T10:20:00Z"
}
```

### GET /api/health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "index_loaded": true,
  "model_loaded": true
}
```

## Environment Variables

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_api_key_here
API_HOST=0.0.0.0
API_PORT=8000
```

## CORS Configuration

The API is configured to allow requests from:
- http://localhost:5173 (Vite dev server)
- http://localhost:3000 (Alternative dev port)

## Architecture

```
backend/api/
├── __init__.py       # Package init
├── main.py           # FastAPI app + startup
├── routes.py         # API endpoints
├── models.py         # Pydantic models
└── README.md         # This file
```

The API wraps the existing QA pipeline (`src/qa_pipeline.py`) without modifying it.
