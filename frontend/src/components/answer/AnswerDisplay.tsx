/**
 * Main answer display component with sources
 */
import React from 'react';
import { SourceList } from './SourceList';
import { Sparkles, Bot } from 'lucide-react';
import type { QAResponse } from '../../types';

interface AnswerDisplayProps {
    data: QAResponse;
}

export const AnswerDisplay: React.FC<AnswerDisplayProps> = ({ data }) => {
    return (
        <div className="w-full max-w-7xl mx-auto">
            {/* Answer Section */}
            <div className="max-w-3xl mx-auto glass-card rounded-2xl p-8 mb-12 border-blue-500/20 shadow-glow-blue fade-in relative overflow-hidden group">
                {/* Ambient background glow */}
                <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/10 rounded-full blur-[80px] -translate-y-1/2 translate-x-1/2 pointer-events-none" />

                <div className="flex items-center gap-3 mb-6 relative z-10">
                    <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 text-white shadow-lg">
                        <Bot size={24} />
                    </div>
                    <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                        AI Answer
                        <Sparkles size={16} className="text-yellow-400 animate-pulse" />
                    </h2>
                </div>

                <div className="prose prose-invert max-w-none relative z-10">
                    <p className="text-slate-200 text-lg leading-relaxed whitespace-pre-wrap font-light">
                        {data.answer}
                    </p>
                </div>

                <div className="mt-6 pt-6 border-t border-white/5 flex items-center justify-between text-xs text-slate-500 relative z-10">
                    <span>Model: {data.model_used}</span>
                    <span>Generated at {new Date(data.timestamp).toLocaleTimeString()}</span>
                </div>
            </div>

            {/* Sources Section */}
            <SourceList sources={data.sources} />
        </div>
    );
};
