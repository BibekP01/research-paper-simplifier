import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { Upload as UploadIcon, FileText, Loader2, AlertCircle, ArrowLeft } from 'lucide-react';
import { Sidebar } from '../components/layout/Sidebar';
import { uploadFile } from '../services/api';

export function Upload() {
    const navigate = useNavigate();
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const onDrop = async (acceptedFiles: File[]) => {
        if (acceptedFiles.length === 0) return;

        const file = acceptedFiles[0];
        setIsUploading(true);
        setError(null);

        try {
            const response = await uploadFile(file);
            // Navigate to chat with the document ID and filename
            navigate(`/chat/${response.document_id}?filename=${encodeURIComponent(response.filename)}`);
        } catch (err) {
            console.error('Upload failed:', err);
            setError('Failed to upload file. Please try again.');
        } finally {
            setIsUploading(false);
        }
    };

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'application/pdf': ['.pdf'],
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
        },
        maxFiles: 1,
        multiple: false
    });

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

                <main className="relative z-10 pt-20 pb-20 px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto flex flex-col items-center justify-center min-h-screen">
                    {/* Back Button */}
                    <button
                        onClick={() => navigate('/')}
                        className="absolute top-8 left-8 flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        <span>Back to Home</span>
                    </button>

                    <div className="text-center max-w-3xl mx-auto mb-12 fade-in">
                        <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-6">
                            Upload your <span className="gradient-text">Paper</span>
                        </h1>
                        <p className="text-lg sm:text-xl text-slate-400 leading-relaxed mb-8">
                            Upload a research paper (PDF or DOCX) to start an AI-powered conversation.
                        </p>
                    </div>

                    {/* Upload Area */}
                    <div
                        {...getRootProps()}
                        className={`w-full max-w-2xl p-12 border-2 border-dashed rounded-2xl transition-all duration-300 cursor-pointer flex flex-col items-center justify-center gap-4 group
                            ${isDragActive ? 'border-blue-500 bg-blue-500/10' : 'border-slate-700 hover:border-blue-500/50 hover:bg-slate-900/50'}
                            ${isUploading ? 'pointer-events-none opacity-50' : ''}
                        `}
                    >
                        <input {...getInputProps()} />

                        {isUploading ? (
                            <>
                                <Loader2 className="w-16 h-16 text-blue-500 animate-spin" />
                                <p className="text-xl font-medium text-slate-300">Processing document...</p>
                                <p className="text-sm text-slate-500">Extracting text, chunking, and indexing.</p>
                            </>
                        ) : (
                            <>
                                <div className="w-20 h-20 rounded-full bg-slate-800/50 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                                    <UploadIcon className="w-10 h-10 text-blue-400" />
                                </div>
                                <div className="text-center">
                                    <p className="text-xl font-medium text-slate-200 mb-2">
                                        {isDragActive ? "Drop the file here" : "Drag & drop your paper here"}
                                    </p>
                                    <p className="text-slate-500">
                                        or click to browse files
                                    </p>
                                </div>
                                <div className="flex gap-4 mt-4 text-sm text-slate-600">
                                    <span className="flex items-center gap-1"><FileText className="w-4 h-4" /> PDF</span>
                                    <span className="flex items-center gap-1"><FileText className="w-4 h-4" /> DOCX</span>
                                </div>
                            </>
                        )}
                    </div>

                    {error && (
                        <div className="mt-8 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 flex items-center gap-2 fade-in">
                            <AlertCircle className="w-5 h-5" />
                            <p>{error}</p>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
}
