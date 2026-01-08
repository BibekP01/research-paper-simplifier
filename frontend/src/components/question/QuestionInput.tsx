/**
 * Main question input form with slider and validation
 */
import React, { useState } from 'react';
import { Button } from '../ui/Button';
import { ExampleQuestions } from './ExampleQuestions';
import { Search, Settings2 } from 'lucide-react';

interface QuestionInputProps {
    onSubmit: (question: string, topK: number) => void;
    isLoading?: boolean;
    placeholder?: string;
    suggestedQuestions?: string[];
    isLoadingQuestions?: boolean;
}

export const QuestionInput: React.FC<QuestionInputProps> = ({
    onSubmit,
    isLoading = false,
    placeholder = "Ask a question about the research papers...",
    suggestedQuestions,
    isLoadingQuestions = false
}) => {
    const [question, setQuestion] = useState('');
    const [topK, setTopK] = useState(5);
    const [showSettings, setShowSettings] = useState(false);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (question.trim() && !isLoading) {
            onSubmit(question, topK);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e as any);
        }
    };

    return (
        <div className="w-full max-w-3xl mx-auto fade-in">
            <form onSubmit={handleSubmit} className="relative">
                {/* Input Container */}
                <div className="relative group">
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-2xl blur-xl transition-opacity opacity-50 group-hover:opacity-100" />

                    <div className="relative glass-card rounded-2xl p-2 transition-all border-white/10 focus-within:border-blue-500/50">
                        <div className="flex flex-col gap-2">
                            <textarea
                                value={question}
                                onChange={(e) => setQuestion(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder={placeholder}
                                className="w-full bg-transparent border-none text-lg text-white placeholder-slate-500 focus:ring-0 resize-none p-4 min-h-[120px] scrollbar-hide focus:outline-none"
                                disabled={isLoading}
                                aria-label="Question input"
                                aria-invalid={!question.trim() && showSettings}
                            />

                            <div className="flex items-center justify-between px-4 pb-2">
                                {/* Settings Toggle */}
                                <button
                                    type="button"
                                    onClick={() => setShowSettings(!showSettings)}
                                    className={`p-2 rounded-lg transition-colors flex items-center gap-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 ${showSettings ? 'bg-white/10 text-blue-400' : 'text-slate-400 hover:text-white'
                                        }`}
                                    aria-expanded={showSettings}
                                    aria-controls="settings-panel"
                                    aria-label="Toggle settings"
                                >
                                    <Settings2 size={18} />
                                    <span className="hidden sm:inline">Settings</span>
                                </button>

                                {/* Submit Button */}
                                <Button
                                    type="submit"
                                    loading={isLoading}
                                    disabled={!question.trim()}
                                    className="min-w-[140px]"
                                    aria-label={isLoading ? "Processing question" : "Ask question"}
                                >
                                    {!isLoading && <Search size={18} className="mr-2" />}
                                    Ask Question
                                </Button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Settings Panel (Collapsible) */}
                <div className={`overflow-hidden transition-all duration-300 ease-in-out ${showSettings ? 'max-h-40 opacity-100 mt-4' : 'max-h-0 opacity-0'
                    }`}>
                    <div className="glass-card rounded-xl p-6 border-white/5">
                        <div className="flex flex-col gap-4">
                            <div className="flex items-center justify-between">
                                <label className="text-sm font-medium text-slate-300">
                                    Number of sources to retrieve
                                </label>
                                <span className="text-sm font-bold text-blue-400 bg-blue-500/10 px-3 py-1 rounded-full">
                                    {topK} sources
                                </span>
                            </div>
                            <input
                                type="range"
                                min="1"
                                max="10"
                                value={topK}
                                onChange={(e) => setTopK(parseInt(e.target.value))}
                                className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                            />
                            <div className="flex justify-between text-xs text-slate-500 px-1">
                                <span>1 (Faster)</span>
                                <span>10 (More Context)</span>
                            </div>
                        </div>
                    </div>
                </div>
            </form>

            {/* Example Questions */}
            {!isLoading && !question && (
                <ExampleQuestions
                    onSelect={setQuestion}
                    questions={suggestedQuestions}
                    isLoading={isLoadingQuestions}
                />
            )}
        </div>
    );
};
