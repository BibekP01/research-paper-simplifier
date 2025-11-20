"""
Gradio Web Interface for Research Paper Q&A System
"""
import os
import gradio as gr
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root and src directory to path
import sys
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import QAPipeline with error handling
try:
    from src.qa_pipeline import QAPipeline
    QAPIPELINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import QAPipeline: {e}")
    QAPIPELINE_AVAILABLE = False

# Constants
DEFAULT_TOP_K = 5
MAX_TOP_K = 10
DEFAULT_CONFIG_PATH = str(Path(__file__).parent.parent / "config" / "config.yaml")
DEFAULT_INDEX_PATH = str(Path(__file__).parent.parent / "data" / "vectorstore" / "paper_embeddings")

# Global variable for the pipeline
qa_pipeline = None

# CSS for styling
CSS = """
:root {
    --primary: #4f46e5;
    --primary-light: #818cf8;
    --primary-dark: #4338ca;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --bg-primary: #f9fafb;
    --bg-secondary: #f3f4f6;
    --text-primary: #111827;
    --text-secondary: #4b5563;
    --border-radius: 8px;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    line-height: 1.6;
    color: var(--text-primary);
    background-color: var(--bg-primary);
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
    padding: 20px;
}

/* Header */
.header {
    text-align: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e5e7eb;
}

.header h1 {
    margin: 0;
    font-size: 1.875rem;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

/* Chat interface */
.chat-container {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 200px);
    max-height: 800px;
    background: white;
    border-radius: var(--border-radius);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    overflow: hidden;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.message {
    max-width: 80%;
    padding: 0.75rem 1rem;
    border-radius: var(--border-radius);
    line-height: 1.5;
    position: relative;
    word-wrap: break-word;
}

.user-message {
    align-self: flex-end;
    background-color: var(--primary);
    color: white;
    border-bottom-right-radius: 4px;
}

.assistant-message {
    align-self: flex-start;
    background-color: var(--bg-secondary);
    color: var(--text-primary);
    border-bottom-left-radius: 4px;
}

.message-content {
    margin-bottom: 0.5rem;
}

.message-time {
    font-size: 0.75rem;
    opacity: 0.8;
    text-align: right;
}

/* Sources section */
.sources-section {
    margin-top: 1.5rem;
    border-top: 1px solid #e5e7eb;
    padding-top: 1rem;
}

.sources-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    cursor: pointer;
}

.sources-content {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.source-item {
    border: 1px solid #e5e7eb;
    border-radius: var(--border-radius);
    padding: 0.75rem;
    background-color: white;
    transition: all 0.2s;
}

.source-item:hover {
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.source-title {
    font-weight: 600;
    color: var(--primary);
}

.source-score {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.5rem;
    border-radius: 9999px;
}

.high-score {
    background-color: rgba(16, 185, 129, 0.1);
    color: var(--success);
}

.medium-score {
    background-color: rgba(245, 158, 11, 0.1);
    color: var(--warning);
}

.low-score {
    background-color: rgba(239, 68, 68, 0.1);
    color: var(--danger);
}

.source-section {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.source-preview {
    font-size: 0.875rem;
    color: var(--text-secondary);
    background-color: var(--bg-secondary);
    padding: 0.5rem;
    border-radius: 4px;
    max-height: 100px;
    overflow-y: auto;
}

/* Input area */
.input-area {
    border-top: 1px solid #e5e7eb;
    padding: 1rem 1.5rem;
    background-color: white;
}

.input-container {
    display: flex;
    gap: 0.5rem;
}

.text-input {
    flex: 1;
    padding: 0.75rem 1rem;
    border: 1px solid #e5e7eb;
    border-radius: var(--border-radius);
    font-size: 0.9375rem;
    transition: all 0.2s;
}

.text-input:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2);
}

.submit-btn {
    background-color: var(--primary);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    padding: 0 1.5rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
}

.submit-btn:hover {
    background-color: var(--primary-dark);
}

.submit-btn:disabled {
    opacity: 0.7;
    cursor: not-allowed;
}

/* Example questions */
.example-questions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}

.example-btn {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 9999px;
    padding: 0.375rem 0.75rem;
    font-size: 0.8125rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s;
}

.example-btn:hover {
    background-color: var(--bg-secondary);
    border-color: #d1d5db;
}

/* Sidebar */
.sidebar {
    background-color: white;
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.sidebar-section {
    margin-bottom: 1.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #e5e7eb;
}

.sidebar-section:last-child {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
}

.sidebar-title {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
}

.stat-item {
    background-color: var(--bg-secondary);
    border-radius: 6px;
    padding: 0.75rem;
    text-align: center;
}

.stat-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 0.25rem;
}

.stat-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.setting-item {
    margin-bottom: 1rem;
}

.setting-label {
    display: block;
    font-size: 0.875rem;
    margin-bottom: 0.5rem;
    color: var(--text-secondary);
}

.clear-btn {
    width: 100%;
    padding: 0.5rem;
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: var(--border-radius);
    color: var(--danger);
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
}

.clear-btn:hover {
    background-color: #fef2f2;
    border-color: #fca5a5;
}

/* Loading animation */
@keyframes pulse {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}

.typing-indicator {
    display: flex;
    gap: 0.25rem;
    padding: 0.5rem 0;
}

.typing-dot {
    width: 8px;
    height: 8px;
    background-color: var(--text-secondary);
    border-radius: 50%;
    animation: pulse 1.4s infinite ease-in-out;
}

.typing-dot:nth-child(1) { animation-delay: 0s; }
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

/* Responsive adjustments */
@media (max-width: 768px) {
    .gradio-container {
        padding: 10px;
    }
    
    .message {
        max-width: 90%;
    }
    
    .stats-grid {
        grid-template-columns: 1fr;
    }
}
"""

def initialize_pipeline() -> Optional[QAPipeline]:
    """Initialize the Q&A pipeline with default or custom paths."""
    if not QAPIPELINE_AVAILABLE:
        logger.error("QAPipeline is not available. Please check the imports.")
        return None
    
    try:
        # Check if the index file exists
        index_path = os.getenv("INDEX_PATH", DEFAULT_INDEX_PATH)
        config_path = os.getenv("CONFIG_PATH", DEFAULT_CONFIG_PATH)
        
        if not os.path.exists(f"{index_path}.faiss"):
            logger.error(f"Index file not found at {index_path}.faiss")
            return None
            
        logger.info("Initializing QAPipeline...")
        pipeline = QAPipeline(
            index_path=index_path,
            config_path=config_path
        )
        logger.info("QAPipeline initialized successfully")
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize QAPipeline: {e}", exc_info=True)
        return None

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format sources as HTML for display."""
    if not sources:
        return ""
    
    sources_html = []
    for source in sources:
        # Determine score class for styling
        score = source.get("similarity_score", 0)
        if score >= 0.7:
            score_class = "high-score"
        elif score >= 0.4:
            score_class = "medium-score"
        else:
            score_class = "low-score"
        
        source_html = f"""
        <div class="source-item">
            <div class="source-header">
                <div class="source-title">{source.get('paper_title', 'Unknown')}</div>
                <div class="source-score {score_class}">
                    Score: {source.get('similarity_score', 0):.2f}
                </div>
            </div>
            <div class="source-section">Section: {source.get('section', 'N/A')}</div>
            <details>
                <summary>View context</summary>
                <div class="source-preview">{source.get('text_preview', 'No preview available')}</div>
            </details>
        </div>
        """.format(**source, score_class=score_class)
        sources_html.append(source_html)
    
    return """
    <div class="sources-section">
        <div class="sources-header">
            <span>📚 Sources</span>
        </div>
        <div class="sources-content">
            {}
        </div>
    </div>
    """.format("\n".join(sources_html))

def format_message(role: str, message: str) -> str:
    """Format a chat message with role and timestamp."""
    timestamp = datetime.now().strftime("%H:%M")
    role_emoji = "👤" if role == "user" else "🤖"
    role_class = "user-message" if role == "user" else "assistant-message"
    
    return f"""
    <div class="message {role_class}">
        <div class="message-content">{message}</div>
        <div class="message-time">{timestamp}</div>
    </div>
    """

def get_typing_indicator() -> str:
    """Return HTML for a typing indicator."""
    return """
    <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    </div>
    """

class ChatUI:
    """Main chat interface for the Research Paper Q&A system."""
    
    def __init__(self):
        self.pipeline = initialize_pipeline()
        self.chat_history = []
        self.stats = {
            "total_questions": 0,
            "total_sources": 0,
            "model_used": "N/A"
        }
        
        # Example questions
        self.example_questions = [
            "What is self-attention?",
            "How do transformers work?",
            "What are the main contributions of Swin Transformer?",
            "What limitations are mentioned in the papers?",
            "Compare different attention mechanisms.",
            "What datasets were used for evaluation?",
            "How does the model handle long sequences?"
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        return self.stats
    
    def update_stats(self, response: Dict[str, Any]) -> None:
        """Update statistics based on the response."""
        self.stats["total_questions"] += 1
        self.stats["total_sources"] += response.get("num_sources", 0)
        self.stats["model_used"] = response.get("model_used", self.stats["model_used"])
    
    def clear_history(self) -> Tuple[str, List]:
        """Clear the chat history."""
        self.chat_history = []
        return "", []
    
    def on_example_click(self, example: str) -> Tuple[str, str]:
        """Handle click on example question."""
        return example, ""
    
    def format_chat_history(self) -> List[Tuple[str, str]]:
        """Format chat history for Gradio's ChatInterface."""
        formatted = []
        for msg in self.chat_history:
            if msg["role"] == "user":
                formatted.append((msg["content"], None))
            else:
                if len(formatted) > 0 and formatted[-1][1] is None:
                    formatted[-1] = (formatted[-1][0], msg["content"])
                else:
                    formatted.append((None, msg["content"]))
        return formatted
    
    def chat(self, message: str, history: List, top_k: int, stream: bool) -> Tuple[str, List]:
        """Handle a chat message and return the response."""
        if not message or not message.strip():
            return "Please enter a question.", history
        
        if not self.pipeline:
            error_msg = "Error: Q&A pipeline is not available. Please check the logs."
            logger.error(error_msg)
            return error_msg, history
        
        try:
            # Add user message to history
            self.chat_history.append({"role": "user", "content": message})
            
            # Get response from pipeline
            response = self.pipeline.ask(
                question=message,
                top_k=top_k,
                stream=False  # Streaming not implemented in current QAPipeline
            )
            
            # Update statistics
            self.update_stats(response)
            
            # Format the response with sources
            answer = response.get("answer", "No answer generated.")
            sources = response.get("sources", [])
            
            if sources:
                answer += "\n\n" + format_sources(sources)
            
            # Add assistant response to history
            self.chat_history.append({"role": "assistant", "content": answer})
            
            # Format the full chat history
            formatted_history = self.format_chat_history()
            
            return "", formatted_history
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, history

def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""
    chat_ui = ChatUI()
    
    with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
        # Header
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                <div class="header">
                    <h1>🎓 Research Paper Q&A Assistant</h1>
                </div>
                """)
        
        with gr.Row(equal_height=True):
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat History",
                    elem_classes=["chatbot"],
                    height=600
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your question",
                        placeholder="Ask a question about the research papers...",
                        container=False,
                        scale=5,
                        elem_classes=["text-input"]
                    )
                    submit_btn = gr.Button("Send", variant="primary", elem_classes=["submit-btn"])
                
                # Example questions
                with gr.Row():
                    gr.Markdown("<div class='example-questions'>Try asking: </div>")
                    
                    for i, question in enumerate(chat_ui.example_questions[:4]):
                        btn = gr.Button(
                            question,
                            size="sm",
                            min_width=50,
                            elem_classes=["example-btn"],
                            scale=0
                        )
                        btn.click(
                            fn=chat_ui.on_example_click,
                            inputs=[btn],
                            outputs=[msg, chatbot],
                            queue=False
                        )
            
            # Sidebar
            with gr.Column(scale=1):
                with gr.Box(elem_classes=["sidebar"]):
                    # Stats
                    with gr.Group(elem_classes=["sidebar-section"]):
                        gr.Markdown("<div class='sidebar-title'>📊 Statistics</div>")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown(
                                    f"<div class='stat-item'><div class='stat-value'>{chat_ui.stats['total_questions']}</div>"
                                    "<div class='stat-label'>Questions Asked</div></div>"
                                )
                            with gr.Column():
                                gr.Markdown(
                                    f"<div class='stat-item'><div class='stat-value'>{chat_ui.stats['total_sources']}</div>"
                                    "<div class='stat-label'>Sources Cited</div></div>"
                                )
                    
                    # Settings
                    with gr.Group(elem_classes=["sidebar-section"]):
                        gr.Markdown("<div class='sidebar-title'>⚙️ Settings</div>")
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=DEFAULT_TOP_K,
                            step=1,
                            label="Number of sources to retrieve",
                            info="Higher values may provide more context but could be slower",
                            elem_classes=["setting-item"]
                        )
                        
                        # Add a dummy checkbox for streaming (not implemented in current QAPipeline)
                        stream = gr.Checkbox(
                            label="Stream response (not implemented)",
                            value=False,
                            interactive=False,
                            elem_classes=["setting-item"]
                        )
                    
                    # Actions
                    with gr.Group(elem_classes=["sidebar-section"]):
                        clear_btn = gr.Button(
                            "🗑️ Clear Chat History",
                            variant="secondary",
                            elem_classes=["clear-btn"]
                        )
                    
                    # About
                    with gr.Group(elem_classes=["sidebar-section"]):
                        gr.Markdown("""
                        <div class='sidebar-title'>ℹ️ About</div>
                        <div style='font-size: 0.875rem; color: var(--text-secondary); line-height: 1.5;'>
                            <p>This assistant helps you explore research papers using AI-powered Q&A.</p>
                            <p>It retrieves relevant information from the loaded papers and generates answers based on the content.</p>
                        </div>
                        """)
        
        # Event handlers
        msg.submit(
            fn=chat_ui.chat,
            inputs=[msg, chatbot, top_k, stream],
            outputs=[msg, chatbot],
            queue=True
        )
        
        submit_btn.click(
            fn=chat_ui.chat,
            inputs=[msg, chatbot, top_k, stream],
            outputs=[msg, chatbot],
            queue=True
        )
        
        clear_btn.click(
            fn=chat_ui.clear_history,
            inputs=[],
            outputs=[msg, chatbot],
            queue=False
        )
        
        # Load initial message
        demo.load(
            fn=lambda: ("", []),
            inputs=[],
            outputs=[msg, chatbot],
            queue=False
        )
    
    return demo

def main():
    """Run the Gradio web interface."""
    # Check if QAPipeline is available
    if not QAPIPELINE_AVAILABLE:
        print("Error: Could not import QAPipeline. Please check the installation and dependencies.")
        return
    
    # Create and launch the UI
    demo = create_ui()
    
    # Print the local URL
    print("\nStarting Research Paper Q&A Assistant...")
    print("Open the app in your browser at: http://127.0.0.1:7860\n")
    
    # Configure and launch the app with queue
    demo.queue(concurrency_count=3)  # Adjust concurrency as needed
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        favicon_path=None,
        show_error=True,
        enable_queue=True
    )

if __name__ == "__main__":
    main()