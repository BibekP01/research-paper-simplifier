"""
TextChunker Module for Research Paper Q&A Assistant

This module provides intelligent text chunking functionality for splitting
research papers into meaningful chunks for RAG (Retrieval-Augmented Generation).
Features include sentence-based splitting, configurable overlap, section awareness,
and comprehensive metadata tracking.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab', quiet=True)


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    start_char: int
    end_char: int
    has_overlap: bool
    sentence_count: int


class TextChunker:
    """
    Intelligent text chunker for research papers.
    
    This class splits research paper text into meaningful chunks with:
    - Configurable chunk size and overlap
    - Sentence boundary preservation
    - Section awareness
    - Rich metadata tracking
    
    Attributes:
        chunk_size: Target chunk size in tokens (approximate)
        chunk_overlap: Overlap between chunks in tokens
        min_chunk_size: Minimum acceptable chunk size
    """
    
    def __init__(self, 
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None,
                 config_path: str = "config/config.yaml"):
        """
        Initialize the TextChunker.
        
        Args:
            chunk_size: Target chunk size in tokens (loads from config if None)
            chunk_overlap: Overlap between chunks in tokens (loads from config if None)
            config_path: Path to configuration file
        """
        # Load configuration
        config = self._load_config(config_path)
        
        # Set chunk parameters (use provided values or fall back to config)
        self.chunk_size = chunk_size or config.get('embeddings', {}).get('chunk_size', 512)
        self.chunk_overlap = chunk_overlap or config.get('embeddings', {}).get('chunk_overlap', 50)
        self.min_chunk_size = config.get('processing', {}).get('min_chunk_size', 100)
        
        # Token approximation factor (words to tokens)
        self.token_factor = 1.3
        
        logger.info(f"TextChunker initialized")
        logger.info(f"  Chunk size: {self.chunk_size} tokens (~{int(self.chunk_size/self.token_factor)} words)")
        logger.info(f"  Chunk overlap: {self.chunk_overlap} tokens (~{int(self.chunk_overlap/self.token_factor)} words)")
        logger.info(f"  Min chunk size: {self.min_chunk_size} tokens")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.debug(f"Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return {}
    
    def load_processed_paper(self, json_path: Path) -> Dict[str, Any]:
        """
        Load a processed paper from JSON file.
        
        Args:
            json_path: Path to processed paper JSON file
            
        Returns:
            Dictionary containing paper data
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        if not json_path.exists():
            raise FileNotFoundError(f"Processed paper not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
                logger.debug(f"Loaded paper: {json_path.name}")
                return paper_data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_path}: {e}")
            raise
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []
        
        # Use NLTK's sentence tokenizer
        sentences = sent_tokenize(text)
        
        # Clean up sentences (remove extra whitespace)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _count_words(self, text: str) -> int:
        """
        Count words in text.
        
        Args:
            text: Text to count words in
            
        Returns:
            Word count
        """
        return len(text.split())
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        
        Uses approximation: tokens ≈ words * 1.3
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        word_count = self._count_words(text)
        return int(word_count * self.token_factor)
    
    def create_chunks(self, 
                     sentences: List[str],
                     section_name: str,
                     paper_metadata: Dict[str, Any],
                     section_start_char: int = 0) -> List[Dict[str, Any]]:
        """
        Create chunks from sentences with overlap and metadata.
        
        This method intelligently groups sentences into chunks while:
        - Respecting the target chunk size
        - Adding overlap between chunks for context preservation
        - Never breaking sentence boundaries
        - Tracking comprehensive metadata
        
        Args:
            sentences: List of sentences to chunk
            section_name: Name of the section these sentences belong to
            paper_metadata: Metadata about the paper
            section_start_char: Starting character position in full text
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        chunk_index = 0
        char_position = section_start_char
        
        # Calculate target sizes in words (approximate)
        target_words = int(self.chunk_size / self.token_factor)
        overlap_words = int(self.chunk_overlap / self.token_factor)
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = self._count_words(sentence)
            sentence_tokens = int(sentence_words * self.token_factor)
            
            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk = self._create_chunk_dict(
                    sentences=current_chunk_sentences,
                    paper_metadata=paper_metadata,
                    section_name=section_name,
                    chunk_index=chunk_index,
                    start_char=char_position,
                    has_overlap=(chunk_index > 0)
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Calculate overlap: keep last few sentences for context
                overlap_tokens = 0
                overlap_sentences = []
                
                # Work backwards from end to build overlap
                for sent in reversed(current_chunk_sentences):
                    sent_tokens = self._estimate_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                # Update char position (move forward by non-overlapping content)
                non_overlap_text = ' '.join(current_chunk_sentences[:-len(overlap_sentences)] if overlap_sentences else current_chunk_sentences)
                char_position += len(non_overlap_text) + 1  # +1 for space
                
                # Start new chunk with overlap
                current_chunk_sentences = overlap_sentences.copy()
                current_tokens = overlap_tokens
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens
            i += 1
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk = self._create_chunk_dict(
                sentences=current_chunk_sentences,
                paper_metadata=paper_metadata,
                section_name=section_name,
                chunk_index=chunk_index,
                start_char=char_position,
                has_overlap=(chunk_index > 0)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_dict(self,
                          sentences: List[str],
                          paper_metadata: Dict[str, Any],
                          section_name: str,
                          chunk_index: int,
                          start_char: int,
                          has_overlap: bool) -> Dict[str, Any]:
        """
        Create a chunk dictionary with all metadata.
        
        Args:
            sentences: List of sentences in this chunk
            paper_metadata: Paper metadata
            section_name: Section name
            chunk_index: Index of this chunk
            start_char: Starting character position
            has_overlap: Whether this chunk has overlap with previous
            
        Returns:
            Complete chunk dictionary
        """
        # Join sentences into chunk text
        chunk_text = ' '.join(sentences)
        
        # Calculate metrics
        word_count = self._count_words(chunk_text)
        token_count = self._estimate_tokens(chunk_text)
        end_char = start_char + len(chunk_text)
        
        # Extract paper_id from metadata or path
        paper_id = paper_metadata.get('paper_id', 'unknown')
        if paper_id == 'unknown' and 'pdf_path' in paper_metadata:
            # Extract from filename (e.g., "2103.14030v2.pdf" -> "2103")
            pdf_path = Path(paper_metadata['pdf_path'])
            paper_id = pdf_path.stem.split('.')[0]
        
        # Create chunk dictionary
        chunk = {
            "chunk_id": f"{paper_id}_chunk_{chunk_index}",
            "paper_id": paper_id,
            "paper_title": paper_metadata.get('title', 'Unknown Title'),
            "section": section_name,
            "text": chunk_text,
            "chunk_index": chunk_index,
            "word_count": word_count,
            "token_count": token_count,
            "metadata": {
                "start_char": start_char,
                "end_char": end_char,
                "has_overlap": has_overlap,
                "sentence_count": len(sentences)
            }
        }
        
        return chunk
    
    def chunk_paper(self, json_path: Path) -> List[Dict[str, Any]]:
        """
        Main pipeline to chunk a single paper.
        
        This method:
        1. Loads the processed paper
        2. Iterates through sections
        3. Splits each section into sentences
        4. Creates overlapping chunks
        5. Returns all chunks with metadata
        
        Args:
            json_path: Path to processed paper JSON
            
        Returns:
            List of chunk dictionaries
            
        Raises:
            FileNotFoundError: If paper file doesn't exist
            ValueError: If paper data is invalid
        """
        logger.info(f"Chunking paper: {json_path.name}")
        
        # Load paper data
        paper_data = self.load_processed_paper(json_path)
        
        # Validate paper data
        if 'sections' not in paper_data:
            raise ValueError(f"Invalid paper data: missing 'sections' key")
        
        sections = paper_data.get('sections', {})
        if not sections:
            logger.warning(f"No sections found in {json_path.name}")
            return []
        
        # Prepare metadata
        paper_metadata = {
            'paper_id': json_path.stem.split('_')[0] if '_' not in json_path.stem else json_path.stem,
            'title': paper_data.get('title', 'Unknown Title'),
            'pdf_path': paper_data.get('pdf_path', str(json_path))
        }
        
        # Process each section
        all_chunks = []
        global_chunk_index = 0
        char_position = 0
        
        for section_name, section_text in sections.items():
            if not section_text or not section_text.strip():
                logger.debug(f"Skipping empty section: {section_name}")
                continue
            
            logger.debug(f"Processing section: {section_name} ({len(section_text)} chars)")
            
            # Split into sentences
            sentences = self.split_into_sentences(section_text)
            
            if not sentences:
                logger.warning(f"No sentences extracted from section: {section_name}")
                continue
            
            # Create chunks for this section
            section_chunks = self.create_chunks(
                sentences=sentences,
                section_name=section_name,
                paper_metadata=paper_metadata,
                section_start_char=char_position
            )
            
            # Update chunk indices to be global
            for chunk in section_chunks:
                chunk['chunk_index'] = global_chunk_index
                chunk['chunk_id'] = f"{paper_metadata['paper_id']}_chunk_{global_chunk_index}"
                global_chunk_index += 1
            
            all_chunks.extend(section_chunks)
            char_position += len(section_text) + 1
        
        # Validate chunks
        valid_chunks = self._validate_chunks(all_chunks)
        
        logger.info(f"Created {len(valid_chunks)} chunks from {len(sections)} sections")
        
        return valid_chunks
    
    def _validate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate chunks and filter out invalid ones.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            List of valid chunks
        """
        valid_chunks = []
        
        for chunk in chunks:
            # Check for empty text
            if not chunk.get('text', '').strip():
                logger.warning(f"Skipping empty chunk: {chunk.get('chunk_id')}")
                continue
            
            # Check minimum size
            if chunk.get('token_count', 0) < self.min_chunk_size:
                logger.debug(f"Chunk {chunk.get('chunk_id')} below minimum size ({chunk.get('token_count')} tokens)")
                # Still include it if it's the only chunk in a section
                # This handles short sections like abstracts
            
            # Check for reasonable size (not too large)
            if chunk.get('token_count', 0) > self.chunk_size * 2:
                logger.warning(f"Chunk {chunk.get('chunk_id')} unusually large ({chunk.get('token_count')} tokens)")
            
            valid_chunks.append(chunk)
        
        return valid_chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: Path) -> None:
        """
        Save chunks to JSON file.
        
        Args:
            chunks: List of chunk dictionaries
            output_path: Path to save chunks
            
        Raises:
            IOError: If file writing fails
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare output data
            output_data = {
                'paper_id': chunks[0]['paper_id'] if chunks else 'unknown',
                'paper_title': chunks[0]['paper_title'] if chunks else 'Unknown',
                'num_chunks': len(chunks),
                'total_tokens': sum(c['token_count'] for c in chunks),
                'chunks': chunks
            }
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(chunks)} chunks to {output_path}")
            
        except IOError as e:
            logger.error(f"Failed to save chunks to {output_path}: {e}")
            raise
    
    def chunk_directory(self, 
                       input_dir: str = "data/processed",
                       output_dir: str = "data/processed") -> Dict[str, Any]:
        """
        Batch process all papers in a directory.
        
        Args:
            input_dir: Directory containing processed paper JSON files
            output_dir: Directory to save chunked files
            
        Returns:
            Dictionary with processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Find all processed paper JSON files (exclude metadata and chunk files)
        json_files = [
            f for f in input_path.glob("*.json")
            if not f.name.endswith('_metadata.json') and not f.name.endswith('_chunks.json')
        ]
        
        if not json_files:
            logger.warning(f"No processed papers found in {input_dir}")
            return {
                'total_papers': 0,
                'successful': 0,
                'failed': 0,
                'total_chunks': 0
            }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Batch chunking {len(json_files)} papers")
        logger.info(f"{'='*70}\n")
        
        # Process each paper
        stats = {
            'total_papers': len(json_files),
            'successful': 0,
            'failed': 0,
            'total_chunks': 0,
            'papers': []
        }
        
        for json_file in json_files:
            try:
                # Chunk the paper
                chunks = self.chunk_paper(json_file)
                
                if not chunks:
                    logger.warning(f"No chunks created for {json_file.name}")
                    stats['failed'] += 1
                    continue
                
                # Save chunks
                output_filename = f"{json_file.stem}_chunks.json"
                output_file = output_path / output_filename
                self.save_chunks(chunks, output_file)
                
                # Update statistics
                stats['successful'] += 1
                stats['total_chunks'] += len(chunks)
                stats['papers'].append({
                    'paper_id': chunks[0]['paper_id'],
                    'title': chunks[0]['paper_title'],
                    'num_chunks': len(chunks),
                    'output_file': str(output_file)
                })
                
                logger.info(f"✅ {json_file.name}: {len(chunks)} chunks created")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {json_file.name}: {e}")
                stats['failed'] += 1
                continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Batch processing complete")
        logger.info(f"  Successful: {stats['successful']}/{stats['total_papers']}")
        logger.info(f"  Total chunks: {stats['total_chunks']}")
        logger.info(f"{'='*70}\n")
        
        return stats


def main():
    """Example usage of the TextChunker class."""
    
    print("\n" + "="*70)
    print("Text Chunker - Intelligent Paper Chunking for RAG")
    print("="*70)
    
    # Initialize chunker
    chunker = TextChunker()
    
    # Get list of processed papers
    processed_dir = Path("data/processed")
    json_files = [
        f for f in processed_dir.glob("*.json")
        if not f.name.endswith('_metadata.json') and not f.name.endswith('_chunks.json')
    ]
    
    if not json_files:
        print("\n❌ No processed papers found in data/processed/")
        print("Please run pdf_processor.py first to process papers.")
        return
    
    print(f"\n📚 Found {len(json_files)} processed papers:")
    for i, json_file in enumerate(json_files, 1):
        print(f"  {i}. {json_file.name}")
    
    # Example 1: Chunk a single paper
    print(f"\n{'='*70}")
    print("[Example 1] Chunking single paper")
    print(f"{'='*70}")
    
    try:
        test_paper = json_files[0]
        print(f"\n📄 Processing: {test_paper.name}")
        
        # Chunk the paper
        chunks = chunker.chunk_paper(test_paper)
        
        print(f"\n✅ Successfully chunked!")
        print(f"\n📊 Chunking Statistics:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total tokens: {sum(c['token_count'] for c in chunks):,}")
        print(f"  Average chunk size: {sum(c['token_count'] for c in chunks) / len(chunks):.0f} tokens")
        print(f"  Average words per chunk: {sum(c['word_count'] for c in chunks) / len(chunks):.0f}")
        
        # Section distribution
        sections = {}
        for chunk in chunks:
            section = chunk['section']
            sections[section] = sections.get(section, 0) + 1
        
        print(f"\n📑 Chunks by Section:")
        for section, count in sections.items():
            print(f"  - {section}: {count} chunks")
        
        # Show sample chunks
        print(f"\n📝 Sample Chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n  Chunk {i} (ID: {chunk['chunk_id']}):")
            print(f"    Section: {chunk['section']}")
            print(f"    Tokens: {chunk['token_count']}, Words: {chunk['word_count']}")
            print(f"    Has overlap: {chunk['metadata']['has_overlap']}")
            preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            print(f"    Text: {preview}")
        
        # Save chunks
        output_file = processed_dir / f"{test_paper.stem}_chunks.json"
        chunker.save_chunks(chunks, output_file)
        print(f"\n💾 Saved chunks to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Detailed error:")
    
    # Example 2: Batch process all papers
    print(f"\n{'='*70}")
    print("[Example 2] Batch processing all papers")
    print(f"{'='*70}\n")
    
    try:
        stats = chunker.chunk_directory(
            input_dir="data/processed",
            output_dir="data/processed"
        )
        
        print(f"\n✅ Batch processing complete!")
        print(f"\n📊 Summary Statistics:")
        print(f"  Papers processed: {stats['successful']}/{stats['total_papers']}")
        print(f"  Total chunks created: {stats['total_chunks']}")
        if stats['successful'] > 0:
            print(f"  Average chunks per paper: {stats['total_chunks']/stats['successful']:.1f}")
        
        if stats['papers']:
            print(f"\n📁 Processed Papers:")
            for paper in stats['papers']:
                print(f"  - {paper['paper_id']}: {paper['num_chunks']} chunks")
                print(f"    Title: {paper['title'][:60]}...")
        
        print(f"\n💾 All chunk files saved to: {processed_dir}")
        
    except Exception as e:
        print(f"\n❌ Batch processing error: {e}")
        logger.exception("Detailed error:")
    
    print("\n" + "="*70)
    print("Chunking complete! Ready for embedding and vector storage.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
