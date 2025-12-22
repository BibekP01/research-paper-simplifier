/**
 * Glass card component with hover effects
 */
import React from 'react';

interface CardProps {
    children: React.ReactNode;
    className?: string;
    hover?: boolean;
    onClick?: () => void;
}

export const Card: React.FC<CardProps> = ({
    children,
    className = '',
    hover = false,
    onClick
}) => {
    const hoverClass = hover ? 'hover-lift cursor-pointer' : '';
    const clickable = onClick ? 'cursor-pointer' : '';

    return (
        <div
            className={`glass-card rounded-xl p-6 ${hoverClass} ${clickable} ${className}`}
            onClick={onClick}
        >
            {children}
        </div>
    );
};
