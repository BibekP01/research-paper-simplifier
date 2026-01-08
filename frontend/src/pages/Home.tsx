import { useNavigate } from 'react-router-dom';
import { Search, Upload, ArrowRight } from 'lucide-react';
import { Navbar } from '../components/layout/Navbar';

export function Home() {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen bg-black text-white relative overflow-x-hidden selection:bg-blue-500/30">
            {/* Background Gradients */}
            <div className="fixed inset-0 pointer-events-none z-0">
                <div className="absolute top-[-10%] left-[20%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]" />
                <div className="absolute bottom-[-10%] right-[20%] w-[40%] h-[40%] bg-purple-600/10 rounded-full blur-[120px]" />
            </div>

            <Navbar />

            <main className="relative z-10 pt-32 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto flex flex-col items-center justify-center min-h-[80vh]">
                <div className="text-center max-w-3xl mx-auto mb-16 fade-in">
                    <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-6">
                        Chat with your <span className="gradient-text">Research Paper</span>
                    </h1>
                    <p className="text-lg sm:text-xl text-slate-400 leading-relaxed mb-8">
                        Explore research papers from arXiv or upload your own to start an AI-powered conversation.
                    </p>
                </div>

                {/* Navigation Cards */}
                <div className="grid md:grid-cols-2 gap-8 w-full max-w-4xl">
                    {/* Explore Card */}
                    <button
                        onClick={() => navigate('/explore')}
                        className="group relative p-8 bg-slate-900/30 border-2 border-slate-800 rounded-2xl hover:border-blue-500/50 hover:bg-slate-900/50 transition-all duration-300 text-left overflow-hidden"
                    >
                        {/* Gradient Overlay on Hover */}
                        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-purple-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

                        <div className="relative z-10">
                            <div className="w-16 h-16 rounded-full bg-blue-500/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                                <Search className="w-8 h-8 text-blue-400" />
                            </div>

                            <h2 className="text-2xl font-bold text-white mb-3 flex items-center gap-2">
                                Explore Papers
                                <ArrowRight className="w-5 h-5 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all duration-300" />
                            </h2>

                            <p className="text-slate-400 mb-4">
                                Browse and search thousands of research papers from arXiv. Discover the latest in AI, ML, and more.
                            </p>

                            <div className="flex flex-wrap gap-2">
                                <span className="px-3 py-1 bg-slate-800/50 text-slate-400 text-xs rounded-full">
                                    arXiv Integration
                                </span>
                                <span className="px-3 py-1 bg-slate-800/50 text-slate-400 text-xs rounded-full">
                                    20+ Categories
                                </span>
                            </div>
                        </div>
                    </button>

                    {/* Upload Card */}
                    <button
                        onClick={() => navigate('/upload')}
                        className="group relative p-8 bg-slate-900/30 border-2 border-slate-800 rounded-2xl hover:border-purple-500/50 hover:bg-slate-900/50 transition-all duration-300 text-left overflow-hidden"
                    >
                        {/* Gradient Overlay on Hover */}
                        <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-pink-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

                        <div className="relative z-10">
                            <div className="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                                <Upload className="w-8 h-8 text-purple-400" />
                            </div>

                            <h2 className="text-2xl font-bold text-white mb-3 flex items-center gap-2">
                                Upload Paper
                                <ArrowRight className="w-5 h-5 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all duration-300" />
                            </h2>

                            <p className="text-slate-400 mb-4">
                                Upload your own research papers in PDF or DOCX format and start asking questions immediately.
                            </p>

                            <div className="flex flex-wrap gap-2">
                                <span className="px-3 py-1 bg-slate-800/50 text-slate-400 text-xs rounded-full">
                                    PDF Support
                                </span>
                                <span className="px-3 py-1 bg-slate-800/50 text-slate-400 text-xs rounded-full">
                                    DOCX Support
                                </span>
                            </div>
                        </div>
                    </button>
                </div>
            </main>
        </div>
    );
}

