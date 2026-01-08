// ArXiv-related TypeScript types

export interface ArxivPaper {
    arxiv_id: string;
    title: string;
    abstract: string;
    authors: string[];
    published_date: string;
    updated_date: string;
    pdf_url: string;
    categories: string[];
    primary_category: string;
    comment?: string;
    journal_ref?: string;
    doi?: string;
}

export interface ArxivSearchParams {
    query: string;
    max_results?: number;
    sort_by?: 'relevance' | 'lastUpdatedDate' | 'submittedDate';
}

export interface ArxivSearchResponse {
    papers: ArxivPaper[];
    total_results: number;
    query: string;
}

export interface ArxivFetchRequest {
    arxiv_id: string;
}
