/**
 * Grid layout for source cards
 */
import React from 'react';
import { SourceCard } from './SourceCard';
import type { Source } from '../../types';

interface SourceListProps {
    sources: Source[];
}

export const SourceList: React.FC<SourceListProps> = ({ sources }) => {
    if (!sources.length) return null;

    return (
        <div className="mt-12 fade-in" style={{ animationDelay: '0.2s' }}>
            <div className="flex items-center gap-3 mb-6">
                <h2 className="text-xl font-bold text-white">Sources</h2>
                <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 text-xs font-medium">
                    {sources.length} Papers
                </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {sources.map((source, index) => (
                    <div
                        key={source.rank}
                        className="fade-in"
                        style={{ animationDelay: `${0.1 * (index + 1)}s` }}
                    >
                        <SourceCard source={source} />
                    </div>
                ))}
            </div>
        </div>
    );
};
