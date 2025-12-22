"""
Embedder Module for Research Paper Q&A Assistant

This module generates vector embeddings from text chunks using sentence-transformers.
Optimized for M1 Macs with MPS (Metal Performance Shaders) acceleration.

Features:
- Automatic device detection (MPS/CPU)
- Batch processing for efficiency
- Progress tracking with tqdm
- Comprehensive error handling
- JSON serialization support
"""

import json
import yaml
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Embedder:
    """
    Generate vector embeddings from text chunks for semantic search.
    
    This class handles:
    - Loading pre-trained sentence transformer models
    - Automatic M1 MPS acceleration detection
    - Batch processing for efficiency
    - Embedding generation with metadata preservation
    - JSON serialization for storage
    
    Attributes:
        model: SentenceTransformer model for generating embeddings
        model_name: Name of the model being used
        device: Device being used (mps, cuda, or cpu)
        embedding_dim: Dimension of the embedding vectors
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 device: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the Embedder with model and device configuration.
        
        Args:
            model_name: Name of sentence-transformer model (loads from config if None)
            device: Device to use ('mps', 'cuda', 'cpu', or None for auto-detect)
            config_path: Path to configuration file
        """
        # Set default config path if not provided
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config" / "config.yaml")

        # Load configuration
        config = self._load_config(config_path)
        
        # Set model name
        self.model_name = model_name or config.get('embeddings', {}).get(
            'model', 'sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Detect and set device
        self.device = self._detect_device(device)
        
        # Load the model
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"This may take a moment on first run (~80MB download)...")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            
            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"  Model: {self.model_name}")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
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
    
    def _detect_device(self, device: Optional[str] = None) -> str:
        """
        Detect the best available device for computation.
        
        Priority: MPS (M1/M2) > CUDA (NVIDIA) > CPU
        
        Args:
            device: Explicitly specified device (None for auto-detect)
            
        Returns:
            Device string ('mps', 'cuda', or 'cpu')
        """
        if device:
            logger.info(f"Using explicitly specified device: {device}")
            return device
        
        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            logger.info("🚀 MPS (Metal Performance Shaders) detected - M1/M2 acceleration enabled!")
            return "mps"
        
        # Check for CUDA (NVIDIA)
        elif torch.cuda.is_available():
            logger.info("🚀 CUDA detected - GPU acceleration enabled!")
            return "cuda"
        
        # Fallback to CPU
        else:
            logger.info("Using CPU (no GPU acceleration available)")
            return "cpu"
    
    def load_chunks(self, json_path: Path) -> List[Dict[str, Any]]:
        """
        Load chunked paper data from JSON file.
        
        Args:
            json_path: Path to chunks JSON file
            
        Returns:
            List of chunk dictionaries
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle nested structure with 'chunks' key
            if isinstance(data, dict) and 'chunks' in data:
                chunks = data['chunks']
            elif isinstance(data, list):
                chunks = data
            else:
                raise ValueError(f"Unexpected JSON structure in {json_path}")
            
            logger.debug(f"Loaded {len(chunks)} chunks from {json_path}")
            return chunks
            
        except FileNotFoundError:
            logger.error(f"Chunks file not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Convert to list for JSON serialization
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, 
                                  texts: List[str],
                                  batch_size: int = 32,
                                  show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently using batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        try:
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                device=self.device
            )
            
            # Convert to list of lists for JSON serialization
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def embed_chunks(self, 
                    chunks: List[Dict[str, Any]],
                    batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Add embeddings to chunk dictionaries.
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for processing
            
        Returns:
            List of chunks with embeddings added
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batch
        start_time = time.time()
        embeddings = self.generate_embeddings_batch(
            texts,
            batch_size=batch_size,
            show_progress=True
        )
        elapsed = time.time() - start_time
        
        # Add embeddings to chunks
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy['embedding'] = embedding
            chunk_copy['embedding_model'] = self.model_name
            chunk_copy['embedding_dim'] = self.embedding_dim
            chunks_with_embeddings.append(chunk_copy)
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
        logger.info(f"   Speed: {len(embeddings)/elapsed:.1f} chunks/second")
        
        return chunks_with_embeddings
    
    def save_embeddings(self, 
                       chunks_with_embeddings: List[Dict[str, Any]],
                       output_path: Path) -> None:
        """
        Save chunks with embeddings to JSON file.
        
        Args:
            chunks_with_embeddings: List of chunks with embeddings
            output_path: Path to save the file
            
        Raises:
            IOError: If file writing fails
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_with_embeddings, f, indent=2, ensure_ascii=False)
            
            # Calculate file size
            file_size = output_path.stat().st_size / 1024  # KB
            
            logger.info(f"💾 Saved embeddings to: {output_path}")
            logger.info(f"   File size: {file_size:.1f} KB")
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise
    
    def embed_paper(self, 
                   chunks_path: Path,
                   output_path: Optional[Path] = None,
                   batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Main pipeline to embed a single paper's chunks.
        
        This method:
        1. Loads chunks from JSON
        2. Generates embeddings
        3. Adds embeddings to chunk data
        4. Saves to output file
        
        Args:
            chunks_path: Path to chunks JSON file
            output_path: Path to save embeddings (auto-generated if None)
            batch_size: Batch size for processing
            
        Returns:
            List of chunks with embeddings
        """
        logger.info(f"Processing paper: {chunks_path.name}")
        
        # Load chunks
        chunks = self.load_chunks(chunks_path)
        
        # Generate embeddings
        chunks_with_embeddings = self.embed_chunks(chunks, batch_size=batch_size)
        
        # Determine output path
        if output_path is None:
            # Replace _chunks.json with _embeddings.json
            output_path = chunks_path.parent / chunks_path.name.replace(
                '_chunks.json', '_embeddings.json'
            )
        
        # Save embeddings
        self.save_embeddings(chunks_with_embeddings, output_path)
        
        return chunks_with_embeddings
    
    def embed_directory(self,
                       input_dir: str = "data/processed",
                       output_dir: Optional[str] = None,
                       batch_size: int = 32) -> Dict[str, Any]:
        """
        Batch process all chunk files in a directory.
        
        Args:
            input_dir: Directory containing chunk JSON files
            output_dir: Directory to save embeddings (same as input_dir if None)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else input_path
        
        # Find all chunk files
        chunk_files = list(input_path.glob("*_chunks.json"))
        
        if not chunk_files:
            logger.warning(f"No chunk files found in {input_dir}")
            return {
                'total_papers': 0,
                'successful': 0,
                'failed': 0,
                'total_chunks': 0,
                'total_time': 0,
                'papers': []
            }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Batch Embedding - Processing {len(chunk_files)} papers")
        logger.info(f"{'='*70}\n")
        
        stats = {
            'total_papers': len(chunk_files),
            'successful': 0,
            'failed': 0,
            'total_chunks': 0,
            'total_time': 0,
            'papers': []
        }
        
        start_time = time.time()
        
        # Process each file with progress bar
        for chunk_file in tqdm(chunk_files, desc="Embedding papers", unit="paper"):
            try:
                # Generate output filename
                output_file = output_path / chunk_file.name.replace(
                    '_chunks.json', '_embeddings.json'
                )
                
                # Process paper
                paper_start = time.time()
                chunks_with_embeddings = self.embed_paper(
                    chunk_file,
                    output_file,
                    batch_size=batch_size
                )
                paper_time = time.time() - paper_start
                
                # Update statistics
                stats['successful'] += 1
                stats['total_chunks'] += len(chunks_with_embeddings)
                
                # Extract paper info
                paper_id = chunks_with_embeddings[0].get('paper_id', 'unknown')
                paper_title = chunks_with_embeddings[0].get('paper_title', 'Unknown')
                
                stats['papers'].append({
                    'paper_id': paper_id,
                    'title': paper_title,
                    'num_chunks': len(chunks_with_embeddings),
                    'processing_time': paper_time,
                    'output_file': str(output_file)
                })
                
            except Exception as e:
                logger.error(f"Failed to process {chunk_file.name}: {e}")
                stats['failed'] += 1
        
        stats['total_time'] = time.time() - start_time
        
        # Log summary
        logger.info(f"\n{'='*70}")
        logger.info("Batch Embedding Complete!")
        logger.info(f"{'='*70}")
        logger.info(f"  Papers processed: {stats['successful']}/{stats['total_papers']}")
        logger.info(f"  Total chunks embedded: {stats['total_chunks']}")
        logger.info(f"  Total time: {stats['total_time']:.2f}s")
        if stats['successful'] > 0:
            logger.info(f"  Average time per paper: {stats['total_time']/stats['successful']:.2f}s")
            logger.info(f"  Overall speed: {stats['total_chunks']/stats['total_time']:.1f} chunks/second")
        
        return stats


def main():
    """Example usage of the Embedder class."""
    
    print("\n" + "="*70)
    print("Embedder - Vector Embedding Generation for RAG")
    print("="*70)
    
    # Initialize embedder (will auto-detect MPS on M1)
    print("\n[Step 1] Initializing Embedder...")
    embedder = Embedder()
    
    # Get list of chunk files
    processed_dir = Path("data/processed")
    chunk_files = list(processed_dir.glob("*_chunks.json"))
    
    if not chunk_files:
        print("\n❌ No chunk files found in data/processed/")
        print("Please run chunker.py first to create chunks.")
        return
    
    print(f"\n📚 Found {len(chunk_files)} chunked papers:")
    for i, chunk_file in enumerate(chunk_files, 1):
        print(f"  {i}. {chunk_file.name}")
    
    # Example 1: Embed a single paper
    print(f"\n{'='*70}")
    print("[Example 1] Embedding single paper")
    print(f"{'='*70}")
    
    try:
        test_file = chunk_files[0]
        print(f"\n📄 Processing: {test_file.name}")
        
        # Embed the paper
        start_time = time.time()
        chunks_with_embeddings = embedder.embed_paper(test_file, batch_size=32)
        elapsed = time.time() - start_time
        
        print(f"\n✅ Successfully embedded!")
        print(f"\n📊 Embedding Statistics:")
        print(f"  Total chunks: {len(chunks_with_embeddings)}")
        print(f"  Embedding dimension: {embedder.embedding_dim}")
        print(f"  Model: {embedder.model_name}")
        print(f"  Device: {embedder.device}")
        print(f"  Processing time: {elapsed:.2f}s")
        print(f"  Speed: {len(chunks_with_embeddings)/elapsed:.1f} chunks/second")
        
        # Show sample embedding
        sample_chunk = chunks_with_embeddings[0]
        print(f"\n📝 Sample Chunk:")
        print(f"  Chunk ID: {sample_chunk['chunk_id']}")
        print(f"  Paper: {sample_chunk['paper_title'][:60]}...")
        print(f"  Section: {sample_chunk['section']}")
        print(f"  Text length: {len(sample_chunk['text'])} chars")
        print(f"  Embedding dimension: {len(sample_chunk['embedding'])}")
        print(f"  First 10 embedding values:")
        print(f"    {sample_chunk['embedding'][:10]}")
        
        # Output file
        output_file = test_file.parent / test_file.name.replace('_chunks.json', '_embeddings.json')
        print(f"\n💾 Saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Detailed error:")
        return
    
    # Example 2: Batch process all papers
    print(f"\n{'='*70}")
    print("[Example 2] Batch embedding all papers")
    print(f"{'='*70}\n")
    
    try:
        stats = embedder.embed_directory(
            input_dir="data/processed",
            output_dir="data/processed",
            batch_size=32
        )
        
        print(f"\n✅ Batch embedding complete!")
        print(f"\n📊 Summary Statistics:")
        print(f"  Papers processed: {stats['successful']}/{stats['total_papers']}")
        print(f"  Total chunks embedded: {stats['total_chunks']}")
        print(f"  Total time: {stats['total_time']:.2f}s")
        if stats['successful'] > 0:
            print(f"  Average chunks per paper: {stats['total_chunks']/stats['successful']:.1f}")
            print(f"  Overall speed: {stats['total_chunks']/stats['total_time']:.1f} chunks/second")
        
        if stats['papers']:
            print(f"\n📁 Embedded Papers:")
            for paper in stats['papers']:
                print(f"  - {paper['paper_id']}: {paper['num_chunks']} chunks ({paper['processing_time']:.2f}s)")
                print(f"    Title: {paper['title'][:60]}...")
        
        print(f"\n💾 All embedding files saved to: {processed_dir}")
        
    except Exception as e:
        print(f"\n❌ Batch embedding error: {e}")
        logger.exception("Detailed error:")
        return
    
    print("\n" + "="*70)
    print("Embedding complete! Ready for vector store and semantic search.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
