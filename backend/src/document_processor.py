"""
DocumentProcessor Module

This module provides a unified interface for processing different document types
(PDF, DOCX) and extracting text/metadata for the RAG pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import docx
from .pdf_processor import PDFProcessor, ProcessedPaper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    A unified processor for research papers and documents (PDF, DOCX).
    """
    
    def __init__(self, upload_dir: str = "data/uploads", processed_dir: str = "data/processed"):
        self.upload_dir = Path(upload_dir)
        self.processed_dir = Path(processed_dir)
        self.pdf_processor = PDFProcessor(papers_dir=upload_dir, processed_dir=processed_dir)
        self._ensure_directories()
        
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_file(self, file_path: Union[str, Path]) -> ProcessedPaper:
        """
        Process a file (PDF or DOCX) and return a ProcessedPaper object.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            ProcessedPaper object
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return self.pdf_processor.process_paper(path)
        elif suffix == '.docx':
            return self._process_docx(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _process_docx(self, docx_path: Path) -> ProcessedPaper:
        """Process a DOCX file."""
        logger.info(f"Processing DOCX: {docx_path.name}")
        
        try:
            doc = docx.Document(docx_path)
            full_text = []
            sections = {}
            current_section = "Introduction" # Default
            
            # Simple extraction: paragraphs
            # We could try to detect headings for sections
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                
                # Simple heuristic for headings: short, bold, or specific style
                # For now, let's just treat everything as text, maybe improve later
                # or use style names if available 'Heading 1'
                
                if para.style.name.startswith('Heading'):
                    current_section = text
                    sections[current_section] = "" # Initialize
                else:
                    full_text.append(text)
                    if current_section not in sections:
                        sections[current_section] = ""
                    sections[current_section] += text + "\n"
            
            combined_text = "\n\n".join(full_text)
            
            # Metadata
            metadata = {
                'filename': docx_path.name,
                'file_size_mb': round(docx_path.stat().st_size / (1024 * 1024), 2),
                'num_sections': len(sections),
                'text_length': len(combined_text),
                'word_count': len(combined_text.split()),
                'processing_status': 'success',
                'file_type': 'docx'
            }
            
            return ProcessedPaper(
                pdf_path=str(docx_path), # Keeping field name for compatibility
                title=docx_path.stem.replace('_', ' '),
                sections=sections if sections else {"Full Text": combined_text},
                full_text=combined_text,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error processing DOCX {docx_path.name}: {e}")
            raise

    def save_processed_text(self, processed_paper: ProcessedPaper) -> None:
        """Delegate to PDFProcessor's save method or implement similar."""
        # We can reuse PDFProcessor's save logic or just implement simple save here
        # For now reusing PDFProcessor's logic by instantiating it or just copying
        # Since PDFProcessor.save_processed_text is an instance method, we can use self.pdf_processor
        self.pdf_processor.save_processed_text(processed_paper)
