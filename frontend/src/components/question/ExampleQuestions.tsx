/**
 * Example questions component with clickable chips
 */
import React from 'react';
import { Card } from '../ui/Card';

interface ExampleQuestionsProps {
    onSelect: (question: string) => void;
    disabled?: boolean;
    questions?: string[];
    isLoading?: boolean;
}

const DEFAULT_EXAMPLES = [
    "What is self-attention and how does it work?",
    "What are the main contributions of the Swin Transformer paper?",
    "How does the model handle long-range dependencies?",
    "What limitations are mentioned in the results section?"
];

export const ExampleQuestions: React.FC<ExampleQuestionsProps> = ({
    onSelect,
    disabled = false,
    questions,
    isLoading = false
}) => {
    const displayQuestions = questions || DEFAULT_EXAMPLES;

    return (
        <div className="mt-8">
            <p className="text-sm text-slate-400 mb-4 font-medium uppercase tracking-wider">
                Try these examples
            </p>
            {isLoading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {[1, 2, 3, 4].map((i) => (
                        <Card key={i} className="p-4 border-slate-800/50 animate-pulse">
                            <div className="h-12 bg-slate-800/50 rounded"></div>
                        </Card>
                    ))}
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {displayQuestions.map((question, index) => (
                        <Card
                            key={index}
                            hover={!disabled}
                            className={`p-4 transition-colors border-slate-800/50 ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-blue-500/30'
                                }`}
                            onClick={() => !disabled && onSelect(question)}
                        >
                            <div className="flex items-start gap-3">
                                <span className="text-blue-500 mt-1">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <circle cx="12" cy="12" r="10" />
                                        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                                        <path d="M12 17h.01" />
                                    </svg>
                                </span>
                                <p className="text-sm text-slate-300 font-medium leading-relaxed">
                                    {question}
                                </p>
                            </div>
                        </Card>
                    ))}
                </div>
            )}
        </div>
    );
};
