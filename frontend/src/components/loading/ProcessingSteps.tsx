/**
 * Animated processing steps to show progress
 */
import React, { useState, useEffect } from 'react';
import { CheckCircle2, Circle, Loader2 } from 'lucide-react';

const STEPS = [
    "Analyzing your question...",
    "Searching research papers...",
    "Extracting relevant sections...",
    "Generating comprehensive answer..."
];

export const ProcessingSteps: React.FC = () => {
    const [currentStep, setCurrentStep] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentStep(prev => (prev < STEPS.length - 1 ? prev + 1 : prev));
        }, 1500);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="max-w-md mx-auto mt-8 space-y-4">
            {STEPS.map((step, index) => {
                const isCompleted = index < currentStep;
                const isCurrent = index === currentStep;
                const isPending = index > currentStep;

                return (
                    <div
                        key={index}
                        className={`flex items-center gap-3 transition-all duration-500 ${isPending ? 'opacity-30' : 'opacity-100'
                            }`}
                    >
                        <div className="flex-shrink-0">
                            {isCompleted ? (
                                <CheckCircle2 className="text-green-500 w-5 h-5" />
                            ) : isCurrent ? (
                                <Loader2 className="text-blue-500 w-5 h-5 animate-spin" />
                            ) : (
                                <Circle className="text-slate-600 w-5 h-5" />
                            )}
                        </div>
                        <span className={`text-sm font-medium ${isCurrent ? 'text-blue-400' : isCompleted ? 'text-slate-300' : 'text-slate-500'
                            }`}>
                            {step}
                        </span>
                    </div>
                );
            })}
        </div>
    );
};
