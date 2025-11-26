#!/usr/bin/env python
"""
Gradio Interface for Research Paper Q&A System using gr.Interface
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import gradio as gr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import QAPipeline
try:
    from src.qa_pipeline import QAPipeline
    QAPIPELINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import QAPipeline: {e}")
    QAPIPELINE_AVAILABLE = False

# Constants
DEFAULT_TOP_K = 5
DEFAULT_CONFIG_PATH = str(Path(__file__).parent.parent / "config" / "config.yaml")
DEFAULT_INDEX_PATH = str(Path(__file__).parent.parent / "data" / "vectorstore" / "paper_embeddings")

# CSS
CSS = """
:root {
    --primary: #4A90E2;
    --bg-secondary: #F8F9FA;
    --text-primary: #1A202C;
    --text-secondary: #4A5568;
    --border-color: #E2E8F0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: var(--bg-secondary);
}

.gradio-container {
    max-width: 1200px !important;
    padding: 24px;
}
"""

def initialize_pipeline() -> Optional[QAPipeline]:
    """Initialize the Q&A pipeline."""
    if not QAPIPELINE_AVAILABLE:
        logger.error("QAPipeline is not available.")
        return None
    
    try:
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

# Global pipeline
qa_pipeline = initialize_pipeline()

def format_sources_html(sources):
    """Format sources as HTML."""
    if not sources:
        return ""
    
    html_parts = ['<div style="margin-top: 16px; border-top: 1px solid #E2E8F0; padding-top: 16px;">']
    html_parts.append('<div style="font-weight: 600; color: #1A202C; font-size: 14px; margin-bottom: 12px;">📚 Sources</div>')
    
    for source in sources:
        score = source.get("similarity_score", 0)
        if score >= 0.7:
            score_bg = "#C6F6D5"
            score_color = "#22543D"
        elif score >= 0.5:
            score_bg = "#FEFCBF"
            score_color = "#744210"
        else:
            score_bg = "#FED7D7"
            score_color = "#742A2A"
        
        html_parts.append(f"""
<div style="background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; margin: 12px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <div style="font-weight: 600; color: #4A90E2; font-size: 14px; margin-bottom: 8px;">{source.get('paper_title', 'Unknown')}</div>
    <div style="font-size: 12px; font-weight: 600; padding: 4px 12px; border-radius: 12px; background: {score_bg}; color: {score_color}; display: inline-block; margin-bottom: 8px;">
        Score: {source.get('similarity_score', 0):.2f}
    </div>
    <div style="font-size: 13px; color: #4A5568; margin-bottom: 8px;">Section: {source.get('section', 'N/A')}</div>
    <div style="font-size: 13px; color: #4A5568; background-color: #F7FAFC; padding: 12px; border-radius: 8px; border-left: 3px solid #4A90E2;">
        {source.get('text_preview', 'No preview available')}
    </div>
</div>
        """)
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)

def answer_question(question: str, top_k: int = 5) -> str:
    """Answer a question using the QA pipeline."""
    if not question or not question.strip():
        return "Please enter a question."
    
    if not qa_pipeline:
        return "Error: Q&A pipeline is not available. Please check the logs."
    
    try:
        logger.info(f"Processing: {question[:50]}...")
        response = qa_pipeline.ask(question=question, top_k=int(top_k), stream=False)
        
        answer = response.get("answer", "No answer generated.")
        sources = response.get("sources", [])
        
        if sources:
            answer += "\n\n" + format_sources_html(sources)
        
        return answer
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return f"Error: {str(e)}"

# Create the interface
demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(
            label="Question",
            placeholder="Ask a question about the research papers...",
            lines=3
        ),
        gr.Slider(
            minimum=1,
            maximum=10,
            value=DEFAULT_TOP_K,
            step=1,
            label="Number of sources to retrieve"
        )
    ],
    outputs=gr.Textbox(
        label="Answer",
        lines=10
    ),
    title="🎓 Research Paper Q&A Assistant",
    description="Explore and understand research papers with AI-powered question answering",
    examples=[
        ["What is self-attention?", 5],
        ["How do transformers work?", 5],
        ["What are the main contributions?", 5],
        ["What limitations are mentioned?", 5]
    ],
    css=CSS,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    logger.info("Starting app on http://127.0.0.1:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
