"""
PDFProcessor Module for Research Paper Q&A Assistant

This module provides functionality to extract, clean, and process text from
research paper PDFs with support for complex layouts, section detection, and
robust error handling.
"""

import pdfplumber
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import unicodedata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedPaper:
    """Data class to store processed paper information."""
    pdf_path: str
    title: str
    sections: Dict[str, str]
    full_text: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PDFProcessor:
    """
    A class to process research paper PDFs with text extraction and cleaning.
    
    This class handles complex PDF layouts, cleans extracted text, detects
    paper sections, and provides robust error handling for various PDF types.
    """
    
    # Common section headers in research papers
    SECTION_PATTERNS = [
        r'^abstract\s*$',
        r'^introduction\s*$',
        r'^related\s+work\s*$',
        r'^background\s*$',
        r'^methodology\s*$',
        r'^methods\s*$',
        r'^approach\s*$',
        r'^experiments?\s*$',
        r'^results?\s*$',
        r'^evaluation\s*$',
        r'^discussion\s*$',
        r'^conclusion\s*$',
        r'^future\s+work\s*$',
        r'^references\s*$',
        r'^bibliography\s*$',
        r'^appendix\s*$',
    ]
    
    # Patterns for headers/footers to remove
    HEADER_FOOTER_PATTERNS = [
        r'^\d+\s*$',  # Page numbers
        r'^page\s+\d+\s*$',
        r'^\d+\s+of\s+\d+\s*$',
        r'^arXiv:\d+\.\d+v\d+.*$',  # arXiv identifiers
    ]
    
    def __init__(self, papers_dir: str = "data/papers", 
                 processed_dir: str = "data/processed"):
        """
        Initialize the PDFProcessor.
        
        Args:
            papers_dir: Directory containing PDF files
            processed_dir: Directory to save processed text files
        """
        self.papers_dir = Path(papers_dir)
        self.processed_dir = Path(processed_dir)
        self._ensure_directories()
        
        logger.info(f"PDFProcessor initialized")
        logger.info(f"Papers directory: {self.papers_dir}")
        logger.info(f"Processed directory: {self.processed_dir}")
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured data directories exist")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract raw text from a PDF file.
        
        This method handles both single-column and multi-column layouts,
        and provides robust error handling for corrupted or scanned PDFs.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted raw text from the PDF
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF is scanned/has no extractable text
            Exception: For other PDF processing errors
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting text from: {pdf_path.name}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) == 0:
                    raise ValueError("PDF has no pages")
                
                text_parts = []
                total_chars = 0
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text with layout preservation
                        page_text = page.extract_text(
                            layout=True,
                            x_tolerance=3,
                            y_tolerance=3
                        )
                        
                        if page_text:
                            text_parts.append(page_text)
                            total_chars += len(page_text)
                        
                        # Log progress for large documents
                        if page_num % 10 == 0:
                            logger.debug(f"Processed {page_num}/{len(pdf.pages)} pages")
                    
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        continue
                
                # Check if we extracted meaningful text
                if total_chars < 100:
                    raise ValueError(
                        "PDF appears to be scanned or has no extractable text. "
                        "OCR processing would be required."
                    )
                
                full_text = "\n\n".join(text_parts)
                logger.info(f"Successfully extracted {total_chars} characters from {len(pdf.pages)} pages")
                
                return full_text
        
        except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as e:
            raise ValueError(f"Corrupted PDF file: {e}")
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path.name}: {e}")
            raise
    
    def clean_text(self, raw_text: str) -> str:
        """
        Clean and format extracted text.
        
        This method:
        - Fixes hyphenation issues (words split across lines)
        - Removes headers/footers
        - Normalizes whitespace
        - Handles special characters
        - Preserves paragraph structure
        
        Args:
            raw_text: Raw text extracted from PDF
            
        Returns:
            Cleaned and formatted text
        """
        if not raw_text:
            return ""
        
        logger.debug("Cleaning extracted text")
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', raw_text)
        
        # Split into lines for processing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines (we'll add them back strategically)
            if not line:
                continue
            
            # Remove common headers/footers
            if self._is_header_or_footer(line):
                continue
            
            # Skip lines that are just numbers or single characters
            if len(line) <= 2 and not line.isalpha():
                continue
            
            cleaned_lines.append(line)
        
        # Join lines and fix hyphenation
        text = ' '.join(cleaned_lines)
        text = self._fix_hyphenation(text)
        
        # Normalize whitespace
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        
        # Add paragraph breaks at sentence boundaries when appropriate
        text = self._restore_paragraphs(text)
        
        # Clean up special characters while preserving important ones
        text = self._clean_special_characters(text)
        
        logger.debug(f"Text cleaned: {len(text)} characters")
        return text.strip()
    
    def _is_header_or_footer(self, line: str) -> bool:
        """
        Check if a line is likely a header or footer.
        
        Args:
            line: Text line to check
            
        Returns:
            True if line appears to be a header/footer
        """
        line_lower = line.lower().strip()
        
        for pattern in self.HEADER_FOOTER_PATTERNS:
            if re.match(pattern, line_lower, re.IGNORECASE):
                return True
        
        # Check for very short lines that are likely page numbers
        if len(line) < 5 and line.isdigit():
            return True
        
        return False
    
    def _fix_hyphenation(self, text: str) -> str:
        """
        Fix words split across lines with hyphens.
        
        Converts "trans-\nformer" to "transformer" and "trans- former" to "transformer"
        
        Args:
            text: Text with potential hyphenation issues
            
        Returns:
            Text with fixed hyphenation
        """
        # Fix hyphenation at line breaks: "word-\nword" -> "wordword"
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Fix hyphenation with spaces: "word- word" -> "wordword"
        text = re.sub(r'(\w+)-\s+(\w+)', lambda m: 
                     m.group(1) + m.group(2) if m.group(2)[0].islower() else m.group(0), 
                     text)
        
        return text
    
    def _restore_paragraphs(self, text: str) -> str:
        """
        Restore paragraph structure by adding breaks at appropriate points.
        
        Args:
            text: Text to process
            
        Returns:
            Text with restored paragraph breaks
        """
        # Add paragraph break after sentences followed by capital letter
        # (likely new paragraph)
        text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\n\2', text)
        
        # Add break before section headers (all caps or title case at start)
        text = re.sub(r'\n([A-Z][A-Z\s]{3,})\n', r'\n\n\1\n\n', text)
        
        return text
    
    def _clean_special_characters(self, text: str) -> str:
        """
        Clean special characters while preserving important ones.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with cleaned special characters
        """
        # Replace common ligatures
        replacements = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
            '–': '-',  # en-dash
            '—': '-',  # em-dash
            ''': "'",  # smart quotes
            ''': "'",
            '"': '"',
            '"': '"',
            '…': '...',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text 
                      if unicodedata.category(char)[0] != 'C' or char in '\n\t')
        
        return text
    
    def detect_sections(self, text: str) -> Dict[str, str]:
        """
        Detect and extract paper sections (Abstract, Introduction, etc.).
        
        Args:
            text: Cleaned text from the paper
            
        Returns:
            Dictionary mapping section names to their content
        """
        logger.debug("Detecting paper sections")
        
        sections = {}
        lines = text.split('\n')
        
        current_section = "Introduction"  # Default section
        current_content = []
        section_starts = []
        
        # First pass: identify section headers
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if line matches a section pattern
            for pattern in self.SECTION_PATTERNS:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    # Normalize section name
                    section_name = line_stripped.title()
                    section_starts.append((i, section_name))
                    break
        
        # If no sections found, return full text as one section
        if not section_starts:
            logger.warning("No sections detected, returning full text")
            return {"Full Text": text}
        
        # Second pass: extract content for each section
        for idx, (start_line, section_name) in enumerate(section_starts):
            # Determine end line (start of next section or end of document)
            end_line = section_starts[idx + 1][0] if idx + 1 < len(section_starts) else len(lines)
            
            # Extract section content
            section_lines = lines[start_line + 1:end_line]
            section_text = '\n'.join(line for line in section_lines if line.strip())
            
            if section_text.strip():
                sections[section_name] = section_text.strip()
        
        logger.info(f"Detected {len(sections)} sections: {', '.join(sections.keys())}")
        return sections
    
    def _extract_title(self, text: str, pdf_path: Path) -> str:
        """
        Extract paper title from text or filename.
        
        Args:
            text: Full text of the paper
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted or inferred title
        """
        # Try to extract title from first few lines
        lines = text.split('\n')[:10]
        
        for line in lines:
            line = line.strip()
            # Title is usually one of the first substantial lines
            if len(line) > 20 and len(line) < 200 and not line.isupper():
                # Check if it looks like a title (not a URL, not all numbers)
                if not re.match(r'^https?://', line) and not line.replace(' ', '').isdigit():
                    return line
        
        # Fallback: use filename without extension
        title = pdf_path.stem.replace('_', ' ').replace('-', ' ')
        logger.warning(f"Could not extract title, using filename: {title}")
        return title
    
    def process_paper(self, pdf_path: Path) -> ProcessedPaper:
        """
        Main pipeline to process a research paper PDF.
        
        This method orchestrates the entire processing workflow:
        1. Extract raw text from PDF
        2. Clean and format the text
        3. Detect and extract sections
        4. Extract metadata
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessedPaper object containing all extracted information
            
        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If PDF is corrupted or scanned
            Exception: For other processing errors
        """
        logger.info(f"Processing paper: {pdf_path.name}")
        
        try:
            # Step 1: Extract raw text
            raw_text = self.extract_text_from_pdf(pdf_path)
            
            # Step 2: Clean text
            cleaned_text = self.clean_text(raw_text)
            
            # Step 3: Detect sections
            sections = self.detect_sections(cleaned_text)
            
            # Step 4: Extract title
            title = self._extract_title(cleaned_text, pdf_path)
            
            # Step 5: Compile metadata
            metadata = {
                'filename': pdf_path.name,
                'file_size_mb': round(pdf_path.stat().st_size / (1024 * 1024), 2),
                'num_sections': len(sections),
                'text_length': len(cleaned_text),
                'word_count': len(cleaned_text.split()),
                'processing_status': 'success'
            }
            
            # Create ProcessedPaper object
            processed_paper = ProcessedPaper(
                pdf_path=str(pdf_path),
                title=title,
                sections=sections,
                full_text=cleaned_text,
                metadata=metadata
            )
            
            logger.info(f"Successfully processed: {title[:50]}...")
            return processed_paper
        
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except ValueError as e:
            logger.error(f"PDF processing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing {pdf_path.name}: {e}")
            raise
    
    def save_processed_text(self, processed_paper: ProcessedPaper, 
                           output_path: Optional[Path] = None) -> Tuple[Path, Path]:
        """
        Save processed paper to JSON and TXT files.
        
        Args:
            processed_paper: ProcessedPaper object to save
            output_path: Custom output path (optional, uses default if None)
            
        Returns:
            Tuple of (json_path, txt_path) where files were saved
            
        Raises:
            IOError: If file writing fails
        """
        if output_path is None:
            # Generate filename from PDF name
            pdf_name = Path(processed_paper.pdf_path).stem
            base_path = self.processed_dir / pdf_name
        else:
            base_path = output_path
        
        json_path = base_path.with_suffix('.json')
        txt_path = base_path.with_suffix('.txt')
        
        try:
            # Save as JSON (complete structured data)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(processed_paper.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON to: {json_path}")
            
            # Save as TXT (full text only, for easy reading)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Title: {processed_paper.title}\n")
                f.write("=" * 80 + "\n\n")
                f.write(processed_paper.full_text)
            logger.info(f"Saved TXT to: {txt_path}")
            
            return json_path, txt_path
        
        except IOError as e:
            logger.error(f"Error saving processed text: {e}")
            raise
    
    def process_directory(self, max_papers: Optional[int] = None) -> List[ProcessedPaper]:
        """
        Process all PDF files in the papers directory.
        
        Args:
            max_papers: Maximum number of papers to process (None for all)
            
        Returns:
            List of ProcessedPaper objects
        """
        pdf_files = list(self.papers_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.papers_dir}")
            return []
        
        if max_papers:
            pdf_files = pdf_files[:max_papers]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        processed_papers = []
        
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_path.name}")
            logger.info(f"{'='*70}")
            
            try:
                processed_paper = self.process_paper(pdf_path)
                self.save_processed_text(processed_paper)
                processed_papers.append(processed_paper)
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing complete: {len(processed_papers)}/{len(pdf_files)} papers successful")
        logger.info(f"{'='*70}\n")
        
        return processed_papers


def main():
    """Example usage of the PDFProcessor class."""
    
    print("\n" + "="*70)
    print("PDF Processor - Research Paper Text Extraction")
    print("="*70)
    
    # Initialize processor
    processor = PDFProcessor()
    
    # Get list of available PDFs
    pdf_files = list(processor.papers_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("\n❌ No PDF files found in data/papers/")
        print("Please run paper_fetcher.py first to download papers.")
        return
    
    print(f"\n📚 Found {len(pdf_files)} PDF files:")
    for i, pdf in enumerate(pdf_files, 1):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"  {i}. {pdf.name} ({size_mb:.2f} MB)")
    
    # Example 1: Process a single paper
    print(f"\n{'='*70}")
    print("[Example 1] Processing single paper")
    print(f"{'='*70}")
    
    try:
        test_pdf = pdf_files[0]
        print(f"\n📄 Processing: {test_pdf.name}")
        
        # Process the paper
        processed_paper = processor.process_paper(test_pdf)
        
        # Display results
        print(f"\n✅ Successfully processed!")
        print(f"\n📋 Paper Information:")
        print(f"  Title: {processed_paper.title}")
        print(f"  Sections detected: {len(processed_paper.sections)}")
        print(f"  Total words: {processed_paper.metadata['word_count']:,}")
        print(f"  Text length: {processed_paper.metadata['text_length']:,} characters")
        
        print(f"\n📑 Sections found:")
        for section_name in processed_paper.sections.keys():
            section_length = len(processed_paper.sections[section_name])
            print(f"  - {section_name}: {section_length:,} characters")
        
        # Show preview of abstract (if available)
        if 'Abstract' in processed_paper.sections:
            abstract = processed_paper.sections['Abstract']
            preview = abstract[:300] + "..." if len(abstract) > 300 else abstract
            print(f"\n📝 Abstract Preview:")
            print(f"  {preview}")
        
        # Save the processed paper
        json_path, txt_path = processor.save_processed_text(processed_paper)
        print(f"\n💾 Saved to:")
        print(f"  JSON: {json_path}")
        print(f"  TXT:  {txt_path}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except ValueError as e:
        print(f"\n❌ PDF Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Detailed error:")
    
    # Example 2: Process all papers in directory
    print(f"\n{'='*70}")
    print("[Example 2] Processing all papers in directory")
    print(f"{'='*70}\n")
    
    try:
        processed_papers = processor.process_directory(max_papers=None)
        
        print(f"\n✅ Batch processing complete!")
        print(f"  Successfully processed: {len(processed_papers)}/{len(pdf_files)} papers")
        
        # Summary statistics
        total_words = sum(p.metadata['word_count'] for p in processed_papers)
        total_sections = sum(p.metadata['num_sections'] for p in processed_papers)
        
        print(f"\n📊 Summary Statistics:")
        print(f"  Total words extracted: {total_words:,}")
        print(f"  Total sections detected: {total_sections}")
        print(f"  Average sections per paper: {total_sections/len(processed_papers):.1f}")
        
        print(f"\n📁 All processed files saved to: {processor.processed_dir}")
        
    except Exception as e:
        print(f"\n❌ Batch processing error: {e}")
        logger.exception("Detailed error:")
    
    print("\n" + "="*70)
    print("Processing complete! Check data/processed/ for output files.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
