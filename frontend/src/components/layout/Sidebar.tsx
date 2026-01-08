import { Link, useLocation } from 'react-router-dom';
import { Search, Upload } from 'lucide-react';

export function Sidebar() {
    const location = useLocation();

    const isActive = (path: string) => {
        return location.pathname === path;
    };

    return (
        <aside className="fixed left-0 top-0 h-screen w-64 bg-slate-900/50 backdrop-blur-xl border-r border-slate-800 flex flex-col z-50">
            {/* Logo/Title */}
            <div className="p-6 border-b border-slate-800">
                <h1 className="text-xl font-bold text-white">
                    Research Paper <span className="gradient-text">Simplifier</span>
                </h1>
            </div>

            {/* Navigation */}
            <nav className="flex-1 p-4 space-y-2">
                <Link
                    to="/explore"
                    className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group
                        ${isActive('/explore')
                            ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                            : 'text-slate-400 hover:bg-slate-800/50 hover:text-white'
                        }`}
                >
                    <Search className={`w-5 h-5 ${isActive('/explore') ? 'text-blue-400' : 'group-hover:scale-110 transition-transform'}`} />
                    <span className="font-medium">Explore</span>
                </Link>

                <Link
                    to="/upload"
                    className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group
                        ${isActive('/upload')
                            ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                            : 'text-slate-400 hover:bg-slate-800/50 hover:text-white'
                        }`}
                >
                    <Upload className={`w-5 h-5 ${isActive('/upload') ? 'text-blue-400' : 'group-hover:scale-110 transition-transform'}`} />
                    <span className="font-medium">Upload</span>
                </Link>
            </nav>

            {/* Footer */}
            <div className="p-4 border-t border-slate-800">
                <p className="text-xs text-slate-600 text-center">
                    Powered by Gemini AI
                </p>
            </div>
        </aside>
    );
}
