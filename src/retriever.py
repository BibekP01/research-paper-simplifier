"""
FAISS-based vector store for semantic search over research paper chunks.
"""

import os
import json
import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path

import faiss
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorStoreRetriever:
    """
    A FAISS-based vector store for efficient semantic search over research paper chunks.
    
    This class handles loading embeddings, building and managing a FAISS index,
    and performing similarity searches with configurable parameters.
    """
    
    def __init__(self, model_name: str = None, index_path: str = None, config_path: str = None):
        """
        Initialize the VectorStoreRetriever.
        
        Args:
            model_name: Name of the sentence-transformers model to use for embeddings.
                       If None, loads from config.
            index_path: Path to a pre-built FAISS index. If provided, loads the index.
            config_path: Path to config file. Defaults to '../config/config.yaml'.
        """
        self.config = self._load_config(config_path)
        self.model_name = model_name or self.config['embeddings']['model']
        self.top_k = self.config['vectorstore'].get('top_k', 5)
        self.similarity_metric = self.config['vectorstore'].get('similarity_metric', 'cosine').lower()
        
        # Initialize FAISS index and metadata
        self.index = None
        self.metadata = []
        self.embedding_dim = None
        self.embedding_model = None
        
        # Load pre-built index if path is provided
        if index_path:
            self.load_index(index_path)
        
        # Initialize embedding model (lazy load)
        self._embedding_model = None
    
    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}, using defaults. Error: {e}")
            return {
                'embeddings': {'model': 'sentence-transformers/all-MiniLM-L6-v2'},
                'vectorstore': {'top_k': 5, 'similarity_metric': 'cosine'}
            }
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._embedding_model = SentenceTransformer(self.model_name)
            # Set embedding dimension based on the model
            self.embedding_dim = self._embedding_model.get_sentence_embedding_dimension()
        return self._embedding_model
    
    @embedding_model.setter
    def embedding_model(self, value):
        self._embedding_model = value
    
    def load_embeddings(self, embeddings_dir: str) -> Tuple[np.ndarray, List[dict]]:
        """
        Load embeddings and metadata from JSON files in the specified directory.
        
        Args:
            embeddings_dir: Directory containing *_embeddings.json files
            
        Returns:
            Tuple of (embeddings, metadata_list)
        """
        embeddings = []
        metadata_list = []
        
        # Find all embedding files
        embedding_files = list(Path(embeddings_dir).glob('*_embeddings.json'))
        
        if not embedding_files:
            raise FileNotFoundError(f"No embedding files found in {embeddings_dir}")
        
        logger.info(f"Loading embeddings from {len(embedding_files)} files in {embeddings_dir}")
        
        for file_path in tqdm(embedding_files, desc="Loading embedding files"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Handle both list and dict with 'chunks' key
                chunks = data.get('chunks', data) if isinstance(data, dict) else data
                
                for chunk in chunks:
                    if 'embedding' not in chunk:
                        logger.warning(f"Skipping chunk without embedding in {file_path}")
                        continue
                        
                    # Extract embedding and metadata
                    embedding = chunk.pop('embedding')
                    
                    # Ensure embedding is a list
                    if isinstance(embedding, str):
                        embedding = json.loads(embedding)
                    
                    # Update metadata with additional fields
                    chunk_metadata = {
                        'chunk_id': chunk.get('chunk_id', f"{file_path.stem}_{len(embeddings)}"),
                        'paper_id': chunk.get('paper_id', file_path.stem.split('_')[0]),
                        'text': chunk.get('text', ''),
                        'section': chunk.get('section', 'unknown'),
                        'metadata': chunk  # Store all other metadata
                    }
                    
                    embeddings.append(embedding)
                    metadata_list.append(chunk_metadata)
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if not embeddings:
            raise ValueError("No valid embeddings found in the specified directory")
        
        # Convert to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity if needed
        if self.similarity_metric == 'cosine':
            faiss.normalize_L2(embeddings_np)
        
        logger.info(f"Loaded {len(embeddings_np)} embeddings with dimension {embeddings_np.shape[1]}")
        return embeddings_np, metadata_list
    
    def build_index(self, embeddings: np.ndarray, metadata: List[dict] = None) -> None:
        """
        Build a FAISS index from the provided embeddings and metadata.
        
        Args:
            embeddings: Numpy array of embeddings (n_embeddings x embedding_dim)
            metadata: List of metadata dictionaries corresponding to each embedding
        """
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings provided to build index")
            
        n_embeddings, dim = embeddings.shape
        self.embedding_dim = dim
        
        # Initialize FAISS index based on similarity metric
        if self.similarity_metric == 'cosine':
            # For cosine similarity, we use Inner Product (IP) on normalized vectors
            self.index = faiss.IndexFlatIP(dim)
        else:  # Default to L2 distance
            self.index = faiss.IndexFlatL2(dim)
        
        # Add vectors to the index
        self.index.add(embeddings)
        
        # Store metadata if provided
        if metadata is not None:
            if len(metadata) != n_embeddings:
                logger.warning(f"Mismatch between number of embeddings ({n_embeddings}) "
                             f"and metadata items ({len(metadata)})")
            self.metadata = metadata
        else:
            self.metadata = [{} for _ in range(n_embeddings)]
        
        logger.info(f"Built FAISS index with {n_embeddings} vectors (dim={dim}, "
                   f"similarity={self.similarity_metric})")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[dict] = None) -> None:
        """
        Add new embeddings and metadata to the existing index.
        
        Args:
            embeddings: Numpy array of new embeddings
            metadata: List of metadata dictionaries for the new embeddings
        """
        if self.index is None:
            return self.build_index(embeddings, metadata)
            
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Dimension mismatch: expected {self.embedding_dim}, "
                           f"got {embeddings.shape[1]}")
        
        # Normalize new embeddings for cosine similarity
        if self.similarity_metric == 'cosine':
            faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Update metadata
        if metadata is not None:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in range(len(embeddings))])
        
        logger.info(f"Added {len(embeddings)} new vectors to the index")
    
    def save_index(self, index_path: str) -> None:
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            index_path: Base path for saving index files (.faiss and .json will be appended)
        """
        if self.index is None:
            raise ValueError("No index to save")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        
        # Save FAISS index
        faiss_file = f"{index_path}.faiss"
        faiss.write_index(self.index, faiss_file)
        
        # Save metadata
        metadata_file = f"{index_path}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'similarity_metric': self.similarity_metric,
                'num_vectors': len(self.metadata),
                'metadata': self.metadata
            }, f, indent=2)
        
        logger.info(f"Saved index to {faiss_file} and metadata to {metadata_file}")
    
    def load_index(self, index_path: str) -> None:
        """
        Load a pre-built FAISS index and metadata from disk.
        
        Args:
            index_path: Base path of the index files (.faiss and _metadata.json)
        """
        faiss_file = f"{index_path}.faiss"
        metadata_file = f"{index_path}_metadata.json"
        
        if not os.path.exists(faiss_file) or not os.path.exists(metadata_file):
            raise FileNotFoundError(
                f"Could not find index files. Expected {faiss_file} and {metadata_file}"
            )
        
        # Load FAISS index
        self.index = faiss.read_index(faiss_file)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            self.model_name = data.get('model_name', self.model_name)
            self.embedding_dim = data.get('embedding_dim', self.embedding_dim)
            self.similarity_metric = data.get('similarity_metric', self.similarity_metric)
            self.metadata = data.get('metadata', [])
        
        logger.info(f"Loaded index with {len(self.metadata)} vectors (dim={self.embedding_dim}, "
                   f"similarity={self.similarity_metric})")
    
    def generate_query_embedding(self, query_text: str) -> np.ndarray:
        """
        Generate an embedding for the given query text.
        
        Args:
            query_text: The query text to embed
            
        Returns:
            Numpy array containing the query embedding
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
            
        # Generate embedding
        query_embedding = self.embedding_model.encode(
            query_text,
            show_progress_bar=False,
            convert_to_numpy=True
        ).reshape(1, -1).astype('float32')
        
        # Normalize for cosine similarity if needed
        if self.similarity_metric == 'cosine':
            faiss.normalize_L2(query_embedding)
        
        return query_embedding
    
    def search(self, query: str, top_k: int = None) -> List[dict]:
        """
        Search for the most similar chunks to the query text.
        
        Args:
            query: The search query text
            top_k: Number of results to return (overrides config if provided)
            
        Returns:
            List of result dictionaries sorted by relevance
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        if self.index is None or len(self.metadata) == 0:
            raise ValueError("Index is empty. Load or build an index first.")
            
        top_k = top_k or self.top_k
        top_k = min(top_k, len(self.metadata))  # Ensure we don't ask for more than we have
        
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        # Search the index
        return self.search_by_embedding(query_embedding, top_k, query=query)
    
    def search_by_embedding(self, query_embedding: np.ndarray, 
                          top_k: int = None, query: str = None) -> List[dict]:
        """
        Search for the most similar chunks to the query embedding.
        
        Args:
            query_embedding: Pre-computed query embedding
            top_k: Number of results to return (overrides config if provided)
            query: Original query text (optional, for logging)
            
        Returns:
            List of result dictionaries sorted by relevance
        """
        if self.index is None or len(self.metadata) == 0:
            raise ValueError("Index is empty. Load or build an index first.")
            
        top_k = top_k or self.top_k
        top_k = min(top_k, len(self.metadata))
        
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure correct dimension
        if query_embedding.shape[1] != self.embedding_dim:
            raise ValueError(f"Query embedding dimension ({query_embedding.shape[1]}) "
                           f"does not match index dimension ({self.embedding_dim})")
        
        # Search the index
        start_time = time.time()
        distances, indices = self.index.search(query_embedding, top_k)
        search_time = time.time() - start_time
        
        # Prepare results
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < 0 or idx >= len(self.metadata):  # Skip invalid indices
                continue
                
            metadata = self.metadata[idx].copy()
            
            # For cosine similarity, convert IP to cosine similarity score [0, 1]
            if self.similarity_metric == 'cosine':
                # IP on normalized vectors is cosine similarity
                score = float((dist + 1) / 2)  # Convert from [-1, 1] to [0, 1]
            else:  # L2 distance - convert to similarity score
                # Simple conversion: 1 / (1 + distance)
                score = float(1 / (1 + dist))
            
            result = {
                'chunk_id': metadata.pop('chunk_id', f"chunk_{idx}"),
                'paper_id': metadata.pop('paper_id', 'unknown'),
                'text': metadata.pop('text', ''),
                'section': metadata.pop('section', 'unknown'),
                'similarity_score': score,
                'rank': i + 1,
                'metadata': metadata,
                'search_time_ms': search_time * 1000  # Convert to milliseconds
            }
            results.append(result)
        
        # Sort by score (descending) to ensure highest scores come first
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Log the search
        if query:
            logger.info(f"Search for '{query[:50]}{'...' if len(query) > 50 else ''}' "
                       f"returned {len(results)} results in {search_time*1000:.2f}ms")
        
        return results
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary containing index statistics
        """
        stats = {
            'num_vectors': len(self.metadata) if self.index else 0,
            'embedding_dim': self.embedding_dim,
            'similarity_metric': self.similarity_metric,
            'model_name': self.model_name,
            'index_type': str(type(self.index).__name__) if self.index else 'None',
            'is_trained': self.index.is_trained if self.index else False,
            'ntotal': self.index.ntotal if self.index else 0
        }
        
        # Add metadata statistics if available
        if self.metadata:
            paper_ids = set(m.get('paper_id') for m in self.metadata)
            sections = set(m.get('section', 'unknown') for m in self.metadata)
            
            stats.update({
                'unique_papers': len(paper_ids),
                'unique_sections': len(sections),
                'avg_chars_per_chunk': sum(
                    len(m.get('text', '')) for m in self.metadata
                ) / len(self.metadata) if self.metadata else 0
            })
        
        return stats


def build_and_save_index(embeddings_dir: str, output_dir: str, 
                        model_name: str = None, config_path: str = None) -> VectorStoreRetriever:
    """
    Helper function to build and save a FAISS index from embeddings.
    
    Args:
        embeddings_dir: Directory containing *_embeddings.json files
        output_dir: Directory to save the index files
        model_name: Name of the embedding model (optional)
        config_path: Path to config file (optional)
        
    Returns:
        Configured VectorStoreRetriever instance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize retriever
    retriever = VectorStoreRetriever(model_name=model_name, config_path=config_path)
    
    # Load embeddings and metadata
    embeddings, metadata = retriever.load_embeddings(embeddings_dir)
    
    # Build index
    retriever.build_index(embeddings, metadata)
    
    # Save index
    index_path = os.path.join(output_dir, 'paper_embeddings')
    retriever.save_index(index_path)
    
    # Print statistics
    stats = retriever.get_statistics()
    print("\nIndex Statistics:")
    print(f"- Total vectors: {stats['num_vectors']}")
    print(f"- Embedding dimension: {stats['embedding_dim']}")
    print(f"- Similarity metric: {stats['similarity_metric']}")
    print(f"- Unique papers: {stats.get('unique_papers', 'N/A')}")
    print(f"- Unique sections: {stats.get('unique_sections', 'N/A')}")
    print(f"- Average chunk length: {stats.get('avg_chars_per_chunk', 'N/A'):.0f} chars")
    print(f"\nIndex saved to: {index_path}.faiss")
    
    return retriever


def load_retriever(index_path: str, config_path: str = None) -> VectorStoreRetriever:
    """
    Helper function to load a pre-built FAISS index.
    
    Args:
        index_path: Base path of the index files (.faiss and _metadata.json)
        config_path: Path to config file (optional)
        
    Returns:
        Configured VectorStoreRetriever instance
    """
    retriever = VectorStoreRetriever(config_path=config_path)
    retriever.load_index(index_path)
    
    # Print statistics
    stats = retriever.get_statistics()
    print("\nLoaded Index Statistics:")
    print(f"- Total vectors: {stats['num_vectors']}")
    print(f"- Embedding dimension: {stats['embedding_dim']}")
    print(f"- Similarity metric: {stats['similarity_metric']}")
    print(f"- Model: {stats['model_name']}")
    
    return retriever


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FAISS Vector Store for Research Papers")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build a new FAISS index')
    build_parser.add_argument('--embeddings-dir', type=str, default='../data/processed',
                            help='Directory containing embedding JSON files')
    build_parser.add_argument('--output-dir', type=str, default='../data/vectorstore',
                            help='Directory to save the index files')
    build_parser.add_argument('--model', type=str, 
                            help='Name of the sentence-transformers model to use')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search the FAISS index')
    search_parser.add_argument('--index-path', type=str, default='data/vectorstore/paper_embeddings',
                             help='Base path of the index files (relative to project root)')
    search_parser.add_argument('--query', type=str, required=True,
                             help='Search query')
    search_parser.add_argument('--top-k', type=int, default=3,
                             help='Number of results to return')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show index statistics')
    stats_parser.add_argument('--index-path', type=str, default='data/vectorstore/paper_embeddings',
                            help='Base path of the index files (relative to project root)')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        # Build and save a new index
        retriever = build_and_save_index(
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            model_name=args.model
        )
        
        # Test with some sample queries
        test_queries = [
            "What is self-attention?",
            "How do transformers work?",
            "What are the limitations?"
        ]
        
        print("\nTesting with sample queries:")
        print("-" * 60)
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = retriever.search(query, top_k=3)  # Fixed: Use fixed top_k=3 for testing
            
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] {result['paper_id']} - {result['section']} (Score: {result['similarity_score']:.3f})")
                print("-" * 30)
                print(result['text'][:200] + ("..." if len(result['text']) > 200 else ""))
                print()
    
    elif args.command == 'search':
        # Load existing index and search
        retriever = load_retriever(args.index_path)
        
        print(f"\nSearching for: {args.query}")
        print("-" * 60)
        
        start_time = time.time()
        results = retriever.search(args.query, top_k=args.top_k)
        search_time = time.time() - start_time
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['paper_id']} - {result['section']} (Score: {result['similarity_score']:.3f})")
            print("-" * 30)
            print(result['text'])
            print()
        
        print(f"\nFound {len(results)} results in {search_time*1000:.2f}ms")
    
    elif args.command == 'stats':
        # Show index statistics
        retriever = load_retriever(args.index_path)
    
    else:
        parser.print_help()
