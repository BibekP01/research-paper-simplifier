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
