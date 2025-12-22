/**
 * Custom hook for managing Q&A state and API interactions
 */
import { useState, useCallback } from 'react';
import { askQuestion } from '../services/api';
import type { QAResponse } from '../types';

interface QAState {
    isLoading: boolean;
    error: string | null;
    data: QAResponse | null;
}

export const useQA = () => {
    const [state, setState] = useState<QAState>({
        isLoading: false,
        error: null,
        data: null
    });

    const ask = useCallback(async (question: string, topK: number = 5) => {
        setState(prev => ({ ...prev, isLoading: true, error: null }));

        try {
            const response = await askQuestion(question, topK);
            setState({
                isLoading: false,
                error: null,
                data: response
            });
            return response;
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Failed to get answer';
            setState(prev => ({
                ...prev,
                isLoading: false,
                error: errorMessage
            }));
            throw err;
        }
    }, []);

    const reset = useCallback(() => {
        setState({
            isLoading: false,
            error: null,
            data: null
        });
    }, []);

    return {
        ...state,
        ask,
        reset
    };
};
