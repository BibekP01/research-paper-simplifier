/**
 * Button component with glowing border animation
 */
import React from 'react';

interface ButtonProps {
    children: React.ReactNode;
    onClick?: () => void;
    loading?: boolean;
    disabled?: boolean;
    variant?: 'primary' | 'secondary';
    className?: string;
    type?: 'button' | 'submit' | 'reset';
}

export const Button: React.FC<ButtonProps> = ({
    children,
    onClick,
    loading = false,
    disabled = false,
    variant = 'primary',
    className = '',
    type = 'button'
}) => {
    const baseClasses = 'relative px-8 py-3 rounded-full font-medium text-sm transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed';

    const variantClasses = variant === 'primary'
        ? 'glowing-border hover:scale-105'
        : 'glass hover:bg-white/10';

    return (
        <button
            type={type}
            onClick={onClick}
            disabled={disabled || loading}
            className={`${baseClasses} ${variantClasses} ${className}`}
            style={variant === 'primary' ? { '--glow-bg': '#000' } as React.CSSProperties : undefined}
        >
            <span className="relative z-10 text-white flex items-center gap-2">
                {loading && (
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                        <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                            fill="none"
                        />
                        <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                        />
                    </svg>
                )}
                {loading ? 'Processing...' : children}
            </span>
        </button>
    );
};
