/**
 * API client for communicating with the FastAPI backend.
 */
import type { QARequest, QAResponse, HealthResponse, UploadResponse } from '../types';

const API_BASE_URL = '/api';

/**
 * Ask a question about research papers.
 */
export async function askQuestion(
    question: string,
    topK: number = 5,
    minScore: number = 0.3
): Promise<QAResponse> {
    const request: QARequest = {
        question,
        top_k: topK,
        min_score: minScore
    };

    const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
}

/**
 * Check API health status.
 */
export async function checkHealth(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
    return response.json();
}

/**
 * Upload a file for processing.
 */
export async function uploadFile(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
}

/**
 * Chat with a specific document.
 */
export async function chatWithDocument(
    documentId: string,
    question: string,
    topK: number = 5,
    minScore: number = 0.3
): Promise<QAResponse> {
    const request: QARequest = {
        question,
        top_k: topK,
        min_score: minScore
    };

    const response = await fetch(`${API_BASE_URL}/chat/${documentId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
}

// --- ArXiv API Functions ---

import type { ArxivSearchParams, ArxivSearchResponse, ArxivFetchRequest } from '../types/arxiv';

/**
 * Search for papers on arXiv.
 */
export async function searchArxivPapers(params: ArxivSearchParams): Promise<ArxivSearchResponse> {
    const response = await fetch(`${API_BASE_URL}/arxiv/search`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
}

/**
 * Fetch and process an arXiv paper by ID.
 */
export async function fetchArxivPaper(arxivId: string): Promise<UploadResponse> {
    const request: ArxivFetchRequest = {
        arxiv_id: arxivId
    };

    const response = await fetch(`${API_BASE_URL}/arxiv/fetch`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
}

export async function getArxivCategories(): Promise<{ categories: Record<string, string> }> {
    const response = await fetch(`${API_BASE_URL}/arxiv/categories`);

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
}

/**
 * Get suggested questions for a specific document.
 */
export async function getSuggestedQuestions(documentId: string): Promise<{ questions: string[]; document_id: string }> {
    const response = await fetch(`${API_BASE_URL}/document/${documentId}/suggestions`);

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
}
