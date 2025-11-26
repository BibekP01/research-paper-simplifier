"""
Q&A Pipeline for research paper question answering using VectorStoreRetriever and GeminiClient.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from .retriever import VectorStoreRetriever
from .gemini_client import GeminiClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QAPipeline:
    """Pipeline for question answering over research papers using vector search and Gemini."""
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        config_path: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> None:
        """Initialize the Q&A pipeline with retriever and LLM client.
        
        Args:
            index_path: Path to the FAISS index file
            config_path: Path to the config file
            api_key: API key for Gemini (if not set in environment)
        """
        # Set default paths if not provided
        base_dir = Path(__file__).parent.parent
        self.index_path = index_path or str(base_dir / "data/vectorstore/paper_embeddings")
        self.config_path = config_path or str(base_dir / "config/config.yaml")
        
        # Set API key in environment if provided
        if api_key:
            import os
            os.environ["GOOGLE_API_KEY"] = api_key
        
        # Initialize components
        try:
            logger.info("Initializing VectorStoreRetriever...")
            self.retriever = VectorStoreRetriever(
                index_path=self.index_path,
                config_path=self.config_path
            )
            
            logger.info("Initializing GeminiClient...")
            self.llm = GeminiClient(config_path=self.config_path)
            
            logger.info("QAPipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize QAPipeline: {str(e)}")
            raise
    
    def ask(
        self,
        question: str,
        top_k: int = 5,
        stream: bool = False,
        min_score: float = 0.3
    ) -> Dict[str, Any]:
        """Process a question and return an answer with sources.
        
        Args:
            question: The question to answer
            top_k: Number of chunks to retrieve
            stream: Whether to stream the response (not implemented)
            min_score: Minimum similarity score for chunks to be considered relevant
            
        Returns:
            Dictionary containing the answer and metadata
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
            
        logger.info(f"Processing question: {question[:100]}...")
        
        # 1. Retrieve relevant chunks
        try:
            chunks = self.retriever.search(question, top_k=top_k)
            logger.info(f"Retrieved {len(chunks)} chunks")
            
            # 2. Filter chunks by similarity score
            valid_chunks = self._validate_chunks(chunks, min_score=min_score)
            
            if not valid_chunks:
                return {
                    "question": question,
                    "answer": "No relevant information found in the papers to answer this question.",
                    "sources": [],
                    "num_sources": 0,
                    "model_used": "N/A",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # 3. Generate answer using LLM
            response = self.llm.ask_question(question, valid_chunks)
            
            # 4. Format the response
            result = {
                "question": question,
                "answer": response.get("answer", "No answer generated."),
                "sources": self.format_sources(valid_chunks),
                "num_sources": len(valid_chunks),
                "model_used": response.get("model", "gemini-pro"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}", exc_info=True)
            return {
                "question": question,
                "answer": "An error occurred while processing your question. Please try again.",
                "sources": [],
                "num_sources": 0,
                "model_used": "N/A",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format chunk metadata for display in the response.
        
        Args:
            chunks: List of chunk dictionaries from the retriever
            
        Returns:
            List of formatted source dictionaries
        """
        formatted_sources = []
        
        for i, chunk in enumerate(chunks, 1):
            formatted_sources.append({
                "rank": i,
                "paper_id": chunk.get("paper_id", ""),
                "paper_title": chunk.get("paper_title", "Unknown"),
                "section": chunk.get("section", ""),
                "similarity_score": round(chunk.get("similarity_score", 0), 4),
                "text_preview": chunk.get("text", "")[:150] + "..."
            })
            
        return formatted_sources
    
    def _validate_chunks(
        self,
        chunks: List[Dict[str, Any]],
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Filter chunks by minimum similarity score.
        
        Args:
            chunks: List of chunk dictionaries
            min_score: Minimum similarity score (0-1)
            
        Returns:
            Filtered list of chunks
        """
        if not chunks:
            return []
            
        valid_chunks = [
            chunk for chunk in chunks 
            if chunk.get("similarity_score", 0) >= min_score
        ]
        
        logger.info(f"Filtered to {len(valid_chunks)} chunks with score >= {min_score}")
        return valid_chunks


def print_qa_result(result: Dict[str, Any]) -> None:
    """Pretty print the Q&A result."""
    print("\n" + "="*80)
    print(f"QUESTION: {result['question']}")
    print("-"*80)
    
    if result['sources']:
        print("\nSOURCES:")
        for src in result['sources']:
            print(f"[{src['rank']}] {src['paper_title']} (Score: {src['similarity_score']:.2f})"
                  f"\n   Section: {src['section']}"
                  f"\n   Preview: {src['text_preview']}")
    
    print("\n" + "-"*40)
    print("ANSWER:")
    print(result['answer'])
    print("\n" + "-"*40)
    print(f"Sources: {result['num_sources']} | "
          f"Model: {result['model_used']} | "
          f"{result['timestamp']}")
    print("="*80 + "\n")


def main():
    """Main function to test the QAPipeline."""
    # You can set your API key here or use environment variable GOOGLE_API_KEY
    api_key = os.getenv("GOOGLE_API_KEY")
    
    try:
        # Initialize the pipeline
        print("Initializing QAPipeline...")
        qa_pipeline = QAPipeline(api_key=api_key)
        
        # Test questions
        test_questions = [
            "What is self-attention and how does it work?",
            "What are the main contributions of the Swin Transformer paper?",
            "What limitations are mentioned in these papers?"
        ]
        
        # Process each question
        for question in test_questions:
            try:
                print(f"\n{'*' * 40}")
                print(f"PROCESSING QUESTION: {question}")
                print(f"{'*' * 40}")
                
                # Get and display the answer
                result = qa_pipeline.ask(question, top_k=5)
                print_qa_result(result)
                
            except Exception as e:
                print(f"Error processing question: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Failed to initialize QAPipeline: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())