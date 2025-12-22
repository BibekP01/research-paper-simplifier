import { Navbar } from './components/layout/Navbar';
import { QuestionInput } from './components/question/QuestionInput';
import { AnswerDisplay } from './components/answer/AnswerDisplay';
import { LoadingSpinner } from './components/loading/LoadingSpinner';
import { ProcessingSteps } from './components/loading/ProcessingSteps';
import { useQA } from './hooks/useQA';
import './App.css';

function App() {
  const { ask, isLoading, error, data } = useQA();

  const handleQuestionSubmit = async (question: string, topK: number) => {
    try {
      await ask(question, topK);
    } catch (err) {
      console.error('Failed to ask question:', err);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white relative overflow-x-hidden selection:bg-blue-500/30">
      {/* Background Gradients */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[20%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[20%] w-[40%] h-[40%] bg-purple-600/10 rounded-full blur-[120px]" />
      </div>

      <Navbar />

      <main className="relative z-10 pt-32 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        {/* Hero Section - Fades out when loading or showing results */}
        <div className={`text-center max-w-3xl mx-auto mb-12 fade-in transition-all duration-500 ${isLoading || data ? 'opacity-0 h-0 overflow-hidden mb-0 scale-95' : ''}`}>
          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-6">
            Research Paper <span className="gradient-text">Simplifier</span>
          </h1>
          <p className="text-lg sm:text-xl text-slate-400 leading-relaxed">
            Instantly understand complex research papers with AI-powered summaries and Q&A.
            Just ask a question and get answers backed by sources.
          </p>
        </div>

        {/* Question Input Section - Moves to top when showing results */}
        <div className={`transition-all duration-700 ease-in-out ${data ? 'mb-12' : 'mb-12'}`}>
          <QuestionInput
            onSubmit={handleQuestionSubmit}
            isLoading={isLoading}
          />
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="max-w-md mx-auto text-center fade-in py-12">
            <LoadingSpinner />
            <ProcessingSteps />
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="max-w-3xl mx-auto mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-center fade-in">
            <p>{error}</p>
          </div>
        )}

        {/* Results Section */}
        {data && !isLoading && (
          <div className="fade-in">
            <AnswerDisplay data={data} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
