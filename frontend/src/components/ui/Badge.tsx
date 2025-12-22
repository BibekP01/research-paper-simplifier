/**
 * Badge component for displaying scores and status
 */
import React from 'react';

interface BadgeProps {
    children: React.ReactNode;
    variant?: 'default' | 'high' | 'medium' | 'low' | 'outline';
    className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
    children,
    variant = 'default',
    className = ''
}) => {
    const baseClasses = 'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium transition-colors';

    const variants = {
        default: 'bg-slate-800 text-slate-300',
        high: 'score-high',     // Defined in App.css
        medium: 'score-medium', // Defined in App.css
        low: 'score-low',       // Defined in App.css
        outline: 'border border-slate-700 text-slate-400'
    };

    return (
        <span className={`${baseClasses} ${variants[variant]} ${className}`}>
            {children}
        </span>
    );
};
