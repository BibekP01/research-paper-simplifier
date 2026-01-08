/**
 * Navbar component with glassmorphism and custom logo
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../../assets/logo.svg';

export const Navbar: React.FC = () => {
    const navigate = useNavigate();
    const [scrolled, setScrolled] = useState(false);

    // Handle scroll effect for glass background
    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 20);
        };

        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    return (
        <nav
            className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled
                ? 'bg-black/50 backdrop-blur-xl border-b border-white/5 py-3'
                : 'bg-transparent py-6'
                }`}
        >
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between">
                    {/* Logo Section */}
                    <div onClick={() => navigate('/')} className="flex items-center gap-3 group cursor-pointer">
                        <div className="relative">
                            <img
                                src={logo}
                                alt="Logo"
                                className="h-10 w-10 transition-transform duration-300 group-hover:scale-110"
                            />
                            {/* Glow effect behind logo */}
                            <div className="absolute inset-0 bg-blue-500/20 blur-xl rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                        </div>

                        <div className="flex flex-col">
                            <span className="text-lg font-bold tracking-tight text-white group-hover:text-blue-400 transition-colors">
                                Research Paper
                            </span>
                            <span className="text-xs font-medium text-slate-400 tracking-wider uppercase">
                                Simplifier
                            </span>
                        </div>
                    </div>

                    {/* Right Section (Optional links/buttons) */}
                    <div className="hidden md:flex items-center gap-6">
                        <a
                            href="#"
                            className="text-sm font-medium text-slate-400 hover:text-white transition-colors"
                        >
                            About
                        </a>
                        <a
                            href="https://github.com"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sm font-medium text-slate-400 hover:text-white transition-colors"
                        >
                            GitHub
                        </a>
                    </div>
                </div>
            </div>
        </nav>
    );
};
