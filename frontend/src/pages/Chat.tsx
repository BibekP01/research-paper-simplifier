import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Document, Page, pdfjs } from 'react-pdf';
import { Navbar } from '../components/layout/Navbar';
import { QuestionInput } from '../components/question/QuestionInput';
import { AnswerDisplay } from '../components/answer/AnswerDisplay';
import { LoadingSpinner } from '../components/loading/LoadingSpinner';
import { chatWithDocument, getSuggestedQuestions } from '../services/api';
import type { QAResponse } from '../types';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// Configure PDF worker
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url,
).toString();

export function Chat() {
    const { documentId } = useParams<{ documentId: string }>();
    const [numPages, setNumPages] = useState<number | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [data, setData] = useState<QAResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>();
    const [isLoadingQuestions, setIsLoadingQuestions] = useState(false);

    // We assume the filename is available or we fetch it. 
    // For now, we might need to fetch document details or just try to load by ID if we saved it with ID.
    // But we saved with original filename. 
    // Ideally, we should have an endpoint to get document details.
    // For this prototype, let's assume we can't easily guess the filename unless we pass it or fetch it.
    // Wait, the user uploads, gets doc_id. The doc_id is derived from filename.
    // But we don't know the extension or full filename just from doc_id if we stripped it.
    // Let's assume for now we only support one file or we need to fetch metadata.
    // I'll add a metadata endpoint or just pass filename in state/query param.
    // Using query param for filename is easiest for now.

    const searchParams = new URLSearchParams(window.location.search);
    const filename = searchParams.get('filename');
    const fileUrl = filename ? `/api/files/${filename}` : null;

    function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
        setNumPages(numPages);
    }

    const handleQuestionSubmit = async (question: string, topK: number) => {
        if (!documentId) return;

        setIsLoading(true);
        setError(null);

        try {
            const response = await chatWithDocument(documentId, question, topK);
            setData(response);
        } catch (err) {
            console.error('Failed to ask question:', err);
            setError('Failed to get answer. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    // Fetch suggested questions on mount
    useEffect(() => {
        if (!documentId) return;

        const fetchSuggestions = async () => {
            setIsLoadingQuestions(true);
            try {
                const response = await getSuggestedQuestions(documentId);
                setSuggestedQuestions(response.questions);
            } catch (err) {
                console.error('Failed to fetch suggestions:', err);
                // Silently fail - component will use default questions
            } finally {
                setIsLoadingQuestions(false);
            }
        };

        fetchSuggestions();
    }, [documentId]);

    return (
        <div className="h-screen bg-black text-white flex flex-col overflow-hidden">
            <Navbar />

            <div className="flex-1 flex pt-20 overflow-hidden">
                {/* Left Panel: PDF Viewer */}
                <div className="w-1/2 border-r border-slate-800 bg-slate-900/50 flex flex-col">
                    <div className="p-4 border-b border-slate-800 flex justify-between items-center">
                        <h2 className="font-semibold text-slate-200 truncate">{filename || 'Document'}</h2>
                        <div className="text-sm text-slate-400">
                            {numPages ? `${numPages} Pages` : 'Loading...'}
                        </div>
                    </div>

                    <div className="flex-1 overflow-auto p-4 bg-slate-950">
                        {fileUrl ? (
                            <div className="flex justify-center">
                                <Document
                                    file={fileUrl}
                                    onLoadSuccess={onDocumentLoadSuccess}
                                    className="flex flex-col gap-4 max-w-full"
                                    loading={<LoadingSpinner />}
                                >
                                    {Array.from(new Array(numPages), (el, index) => (
                                        <Page
                                            key={`page_${index + 1}`}
                                            pageNumber={index + 1}
                                            width={600}
                                            renderTextLayer={true}
                                            renderAnnotationLayer={true}
                                            className="shadow-2xl"
                                        />
                                    ))}
                                </Document>
                            </div>
                        ) : (
                            <div className="flex items-center justify-center h-full text-slate-500">
                                Document not found
                            </div>
                        )}
                    </div>
                </div>

                {/* Right Panel: Chat */}
                <div className="w-1/2 flex flex-col bg-black relative">
                    {/* Background Gradients for Chat */}
                    <div className="absolute inset-0 pointer-events-none z-0 overflow-hidden">
                        <div className="absolute top-[20%] right-[-10%] w-[60%] h-[60%] bg-blue-600/5 rounded-full blur-[100px]" />
                    </div>

                    <div className="flex-1 overflow-y-auto p-6 z-10">
                        {!data && !isLoading && (
                            <div className="h-full flex flex-col items-center justify-center text-center text-slate-500 space-y-4">
                                <div className="w-16 h-16 rounded-2xl bg-slate-900 flex items-center justify-center mb-4">
                                    <span className="text-3xl">👋</span>
                                </div>
                                <h3 className="text-xl font-medium text-slate-200">Ask questions about this paper</h3>
                                <p className="max-w-md">
                                    Select text in the PDF or type a question below to get AI-powered answers with citations.
                                </p>
                            </div>
                        )}

                        {isLoading && (
                            <div className="py-12">
                                <LoadingSpinner />
                                <p className="text-center text-slate-500 mt-4">Analyzing document...</p>
                            </div>
                        )}

                        {data && !isLoading && (
                            <div className="fade-in">
                                <AnswerDisplay data={data} />
                            </div>
                        )}

                        {error && (
                            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-center">
                                {error}
                            </div>
                        )}
                    </div>

                    <div className="p-6 border-t border-slate-800 z-20 bg-black/80 backdrop-blur-sm">
                        <QuestionInput
                            onSubmit={handleQuestionSubmit}
                            isLoading={isLoading}
                            placeholder="Ask about this paper..."
                            suggestedQuestions={suggestedQuestions}
                            isLoadingQuestions={isLoadingQuestions}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}
