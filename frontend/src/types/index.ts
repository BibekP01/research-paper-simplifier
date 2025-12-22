/**
 * TypeScript type definitions for the Research Paper Q&A application.
 */

export interface Source {
    rank: number;
    paper_id: string;
    paper_title: string;
    section: string;
    similarity_score: number;
    text_preview: string;
}

export interface QAResponse {
    question: string;
    answer: string;
    sources: Source[];
    num_sources: number;
    model_used: string;
    timestamp: string;
}

export interface QARequest {
    question: string;
    top_k: number;
    min_score?: number;
}

export interface HealthResponse {
    status: 'healthy' | 'unhealthy' | 'degraded';
    index_loaded: boolean;
    model_loaded: boolean;
    message?: string;
}
