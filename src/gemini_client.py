import os
import time
import logging
import json
from typing import Dict, List, Optional, Generator, Any, Union
from pathlib import Path
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv
import yaml
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeminiClient:
    """
    A client for interacting with Google's Gemini API for research paper Q&A.
    Handles API communication, prompt formatting, and response processing.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        config_path: str = "../config/config.yaml"
    ) -> None:
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Optional API key. If not provided, will try to load from .env
            model_name: Name of the Gemini model to use
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            config_path: Path to the config file
        """
        self.config = self._load_config(config_path)
        # Use the correct model name that we verified works
        self.model_name = 'gemini-2.5-flash'  # Override with the working model
        self.temperature = temperature or self.config.get('gemini', {}).get('temperature', 0.3)
        self.max_tokens = max_tokens or self.config.get('gemini', {}).get('max_output_tokens', 2048)
        
        # Load API key from .env.test for testing
        load_dotenv('.env.test')  # Try loading from test env file first
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            load_dotenv()  # Fall back to .env if .env.test doesn't exist
            self.api_key = os.getenv('GEMINI_API_KEY')
            
        if not self.api_key:
            raise ValueError(
                "API key not provided. Set GEMINI_API_KEY in .env file or pass as argument"
            )
            
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        logger.info(f"Initialized GeminiClient with model: {self.model_name}")
    
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error loading config file: {e}")
            return {}
    
    def validate_api_key(self) -> bool:
        """
        Validate the provided API key by making a test request.
        
        Returns:
            bool: True if the API key is valid, False otherwise
        """
        try:
            # Test if we can list models and find our target model
            models = list(genai.list_models())
            model_names = [model.name for model in models]
            if self.model_name not in model_names and f'models/{self.model_name}' not in model_names:
                logger.warning(f"Model {self.model_name} not found in available models")
                return False
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    def create_prompt(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Format the question and chunks into a prompt for the model.
        
        Args:
            question: The user's question
            chunks: List of chunk dictionaries with paper info and text
            
        Returns:
            str: Formatted prompt for the model
        """
        if not chunks:
            raise ValueError("No chunks provided for context")
            
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            paper_title = chunk.get('paper_title', 'Unknown Paper')
            section = chunk.get('section', 'N/A')
            text = chunk.get('text', '').strip()
            
            context_parts.append(
                f"[Chunk {i}] Paper: {paper_title}, Section: {section}\n"
                f"{text}\n"
            )
        
        # Build the prompt using string concatenation to avoid f-string backslash issues
        prompt_parts = [
            "You are a research paper assistant. Answer the question based on these paper excerpts.\n\n",
            "Context from papers:\n",
            "\n".join(context_parts),
            "\nQuestion: ", question, "\n\n",
            "Instructions:\n",
            "- Answer based only on the provided context\n",
            "- Cite which paper/section you're referencing\n",
            "- If context doesn't contain the answer, say so\n",
            "- Be concise but complete"
        ]
        prompt = "".join(part for part in prompt_parts if part)
        
        return prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((
            Exception,
            genai.types.StopCandidateException,
            ConnectionError,
            TimeoutError
        )),
        reraise=True
    )
    def ask_question(
        self, 
        question: str, 
        chunks: List[Dict[str, Any]],
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Ask a question based on the provided chunks of text.
        
        Args:
            question: The question to ask
            chunks: List of chunk dictionaries with paper info and text
            stream: Whether to stream the response
            
        Returns:
            Dict containing the question, answer, and metadata
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
            
        prompt = self.create_prompt(question, chunks)
        
        try:
            if stream:
                return self._ask_with_streaming(question, chunks)
                
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': self.temperature,
                    'max_output_tokens': self.max_tokens,
                }
            )
            
            # Extract chunk IDs used for citations
            chunk_ids = [str(chunk.get('chunk_id', '')) for chunk in chunks]
            
            return {
                'question': question,
                'answer': response.text.strip(),
                'chunks_used': chunk_ids,
                'model': self.model_name,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _ask_with_streaming(
        self, 
        question: str, 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ask a question with streaming response.
        
        Args:
            question: The question to ask
            chunks: List of chunk dictionaries with paper info and text
            
        Returns:
            Dict containing the question, answer, and metadata
        """
        prompt = self.create_prompt(question, chunks)
        chunk_ids = [str(chunk.get('chunk_id', '')) for chunk in chunks]
        
        try:
            response = self.model.generate_content(
                prompt,
                stream=True,
                generation_config={
                    'temperature': self.temperature,
                    'max_output_tokens': self.max_tokens,
                }
            )
            
            # Stream the response
            full_response = []
            print("\nAssistant: ", end="", flush=True)
            for chunk in response:
                if hasattr(chunk, 'text'):
                    print(chunk.text, end="", flush=True)
                    full_response.append(chunk.text)
            print("\n")
            
            return {
                'question': question,
                'answer': ''.join(full_response).strip(),
                'chunks_used': chunk_ids,
                'model': self.model_name,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during streaming response: {e}")
            raise

def load_sample_chunks() -> List[Dict[str, Any]]:
    """Load sample chunks for testing."""
    return [
        {
            'chunk_id': 'sample_1',
            'paper_id': '2103.00001',
            'paper_title': 'Attention Is All You Need',
            'section': '3.1',
            'text': 'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.'
        },
        {
            'chunk_id': 'sample_2',
            'paper_id': '1706.03762',
            'paper_title': 'Attention Is All You Need',
            'section': '3.2',
            'text': 'An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.'
        }
    ]

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize the client
        client = GeminiClient()
        
        # Validate API key
        if not client.validate_api_key():
            print("Error: Invalid API key. Please check your GEMINI_API_KEY in .env file.")
            exit(1)
            
        # Test with a sample question
        question = "What is self-attention?"
        print(f"\nQuestion: {question}")
        
        # Load sample chunks (in a real app, these would come from the retriever)
        chunks = load_sample_chunks()
        
        # Get answer with streaming
        print("\nGenerating answer (streaming)...")
        response = client.ask_question(question, chunks, stream=True)
        
        # Print formatted response
        print("\n\n=== Final Response ===")
        print(json.dumps(response, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise