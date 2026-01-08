import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Loader2, AlertCircle, ArrowLeft, Calendar, Users, Tag } from 'lucide-react';
import { Sidebar } from '../components/layout/Sidebar';
import { searchArxivPapers, fetchArxivPaper } from '../services/api';
import type { ArxivPaper } from '../types/arxiv';

export function Explore() {
    const navigate = useNavigate();
    const [query, setQuery] = useState('');
    const [papers, setPapers] = useState<ArxivPaper[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [isFetching, setIsFetching] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [hasSearched, setHasSearched] = useState(false);

    const handleSearch = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!query.trim()) return;

        setIsSearching(true);
        setError(null);
        setHasSearched(true);

        try {
            const response = await searchArxivPapers({
                query: query.trim(),
                max_results: 20,
                sort_by: 'relevance'
            });
            setPapers(response.papers);
        } catch (err) {
            console.error('Search failed:', err);
            setError('Failed to search arXiv. Please try again.');
        } finally {
            setIsSearching(false);
        }
    };

    const handleSelectPaper = async (paper: ArxivPaper) => {
        setIsFetching(paper.arxiv_id);
        setError(null);

        try {
            const response = await fetchArxivPaper(paper.arxiv_id);
            // Navigate to chat with the document ID
            navigate(`/chat/${response.document_id}?filename=${encodeURIComponent(response.filename)}`);
        } catch (err) {
            console.error('Fetch failed:', err);
            setError(`Failed to fetch paper: ${paper.title.substring(0, 50)}...`);
            setIsFetching(null);
        }
    };

    return (
        <div className="min-h-screen bg-black text-white flex">
            <Sidebar />

            {/* Main Content */}
            <div className="flex-1 ml-64 relative overflow-x-hidden selection:bg-blue-500/30">
                {/* Background Gradients */}
                <div className="fixed inset-0 pointer-events-none z-0">
                    <div className="absolute top-[-10%] left-[20%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]" />
                    <div className="absolute bottom-[-10%] right-[20%] w-[40%] h-[40%] bg-purple-600/10 rounded-full blur-[120px]" />
                </div>

                <main className="relative z-10 pt-20 pb-20 px-4 sm:px-6 lg:px-8 max-w-6xl mx-auto">
                    {/* Back Button */}
                    <button
                        onClick={() => navigate('/')}
                        className="mb-8 flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        <span>Back to Home</span>
                    </button>

                    {/* Header */}
                    <div className="text-center mb-12 fade-in">
                        <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-6">
                            Explore <span className="gradient-text">Research Papers</span>
                        </h1>
                        <p className="text-lg sm:text-xl text-slate-400 leading-relaxed">
                            Search and discover research papers from arXiv
                        </p>
                    </div>

                    {/* Search Bar */}
                    <form onSubmit={handleSearch} className="mb-12">
                        <div className="relative max-w-3xl mx-auto">
                            <input
                                type="text"
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                placeholder="Search for papers... (e.g., 'large language models', 'transformer architecture')"
                                className="w-full px-6 py-4 pr-14 bg-slate-900/50 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 transition-all"
                                disabled={isSearching}
                            />
                            <button
                                type="submit"
                                disabled={isSearching || !query.trim()}
                                className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-blue-500 hover:bg-blue-600 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg transition-colors"
                            >
                                {isSearching ? (
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                ) : (
                                    <Search className="w-5 h-5" />
                                )}
                            </button>
                        </div>
                    </form>

                    {/* Error Message */}
                    {error && (
                        <div className="mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 flex items-center gap-2 max-w-3xl mx-auto">
                            <AlertCircle className="w-5 h-5 flex-shrink-0" />
                            <p>{error}</p>
                        </div>
                    )}

                    {/* Loading State */}
                    {isSearching && (
                        <div className="flex flex-col items-center justify-center py-20">
                            <Loader2 className="w-12 h-12 text-blue-500 animate-spin mb-4" />
                            <p className="text-slate-400">Searching arXiv...</p>
                        </div>
                    )}

                    {/* No Results */}
                    {!isSearching && hasSearched && papers.length === 0 && (
                        <div className="text-center py-20">
                            <p className="text-slate-400 text-lg">No papers found. Try a different search query.</p>
                        </div>
                    )}

                    {/* Results */}
                    {!isSearching && papers.length > 0 && (
                        <div className="space-y-6">
                            <p className="text-slate-400 mb-6">
                                Found <span className="text-white font-semibold">{papers.length}</span> papers
                            </p>

                            {papers.map((paper) => (
                                <div
                                    key={paper.arxiv_id}
                                    className="bg-slate-900/30 border border-slate-800 rounded-xl p-6 hover:border-slate-700 transition-all duration-300 group"
                                >
                                    <div className="flex justify-between items-start gap-4">
                                        <div className="flex-1">
                                            <h3 className="text-xl font-semibold text-white mb-3 group-hover:text-blue-400 transition-colors">
                                                {paper.title}
                                            </h3>

                                            <div className="flex flex-wrap gap-4 mb-3 text-sm text-slate-400">
                                                <span className="flex items-center gap-1">
                                                    <Users className="w-4 h-4" />
                                                    {paper.authors.slice(0, 3).join(', ')}
                                                    {paper.authors.length > 3 && ` +${paper.authors.length - 3} more`}
                                                </span>
                                                <span className="flex items-center gap-1">
                                                    <Calendar className="w-4 h-4" />
                                                    {new Date(paper.published_date).toLocaleDateString()}
                                                </span>
                                            </div>

                                            <p className="text-slate-400 mb-4 line-clamp-3">
                                                {paper.abstract}
                                            </p>

                                            <div className="flex flex-wrap gap-2 mb-4">
                                                {paper.categories.slice(0, 4).map((category) => (
                                                    <span
                                                        key={category}
                                                        className="inline-flex items-center gap-1 px-3 py-1 bg-slate-800/50 text-slate-400 text-xs rounded-full"
                                                    >
                                                        <Tag className="w-3 h-3" />
                                                        {category}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>

                                        <button
                                            onClick={() => handleSelectPaper(paper)}
                                            disabled={isFetching !== null}
                                            className="px-6 py-3 bg-blue-500 hover:bg-blue-600 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-xl transition-all duration-300 font-medium whitespace-nowrap flex items-center gap-2"
                                        >
                                            {isFetching === paper.arxiv_id ? (
                                                <>
                                                    <Loader2 className="w-4 h-4 animate-spin" />
                                                    Processing...
                                                </>
                                            ) : (
                                                'Select Paper'
                                            )}
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Initial State */}
                    {!hasSearched && (
                        <div className="text-center py-20">
                            <Search className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                            <p className="text-slate-400 text-lg">
                                Enter a search query to discover research papers
                            </p>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
}
