/**
 * Individual source card component
 */
import React from 'react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import type { Source } from '../../types';
import { FileText, ExternalLink } from 'lucide-react';

interface SourceCardProps {
    source: Source;
    onClick?: () => void;
}

export const SourceCard: React.FC<SourceCardProps> = ({ source, onClick }) => {
    const getScoreVariant = (score: number) => {
        if (score >= 0.7) return 'high';
        if (score >= 0.5) return 'medium';
        return 'low';
    };

    return (
        <Card
            hover
            className="h-full flex flex-col transition-all duration-300 group border-slate-800/50 hover:border-blue-500/30"
            onClick={onClick}
        >
            <div className="flex items-start justify-between mb-3 gap-2">
                <div className="flex items-center gap-2 min-w-0">
                    <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 group-hover:bg-blue-500/20 transition-colors">
                        <FileText size={18} />
                    </div>
                    <h3 className="font-bold text-slate-200 text-sm leading-tight line-clamp-2 group-hover:text-blue-400 transition-colors">
                        {source.paper_title}
                    </h3>
                </div>
                <Badge variant={getScoreVariant(source.similarity_score)}>
                    {(source.similarity_score * 100).toFixed(0)}%
                </Badge>
            </div>

            <div className="mb-3">
                <span className="text-xs font-medium text-slate-500 uppercase tracking-wider bg-slate-800/50 px-2 py-1 rounded">
                    {source.section || 'Unknown Section'}
                </span>
            </div>

            <p className="text-sm text-slate-400 leading-relaxed line-clamp-4 flex-grow font-light">
                "{source.text_preview}"
            </p>

            <div className="mt-4 pt-3 border-t border-white/5 flex items-center justify-between text-xs text-slate-500">
                <span>Rank #{source.rank}</span>
                <span className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity text-blue-400">
                    View details <ExternalLink size={12} />
                </span>
            </div>
        </Card>
    );
};
