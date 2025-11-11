import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from src.gemini_client import GeminiClient, load_sample_chunks

def test_gemini_client():
    """Test the GeminiClient with various scenarios."""
    print("=== Testing GeminiClient ===\n")
    
    # Initialize the client
    print("1. Initializing GeminiClient...")
    try:
        client = GeminiClient()
        print("   ✓ Client initialized successfully")
        print(f"   - Model: {client.model_name}")
        print(f"   - Temperature: {client.temperature}")
        print(f"   - Max tokens: {client.max_tokens}")
    except Exception as e:
        print(f"   ✗ Failed to initialize client: {e}")
        return
    
    # Test API key validation
    print("\n2. Validating API key...")
    if client.validate_api_key():
        print("   ✓ API key is valid")
    else:
        print("   ✗ API key validation failed")
        return
    
    # Load sample chunks for testing
    print("\n3. Loading sample chunks...")
    chunks = load_sample_chunks()
    print(f"   ✓ Loaded {len(chunks)} sample chunks")
    for i, chunk in enumerate(chunks, 1):
        print(f"      Chunk {i}: {chunk.get('paper_title', 'Unknown')} - {chunk.get('section', 'N/A')}")
    
    # Test non-streaming question
    print("\n4. Testing non-streaming question...")
    try:
        print("   Sending question: 'What is the main contribution of the Transformer model?'")
        response = client.ask_question(
            "What is the main contribution of the Transformer model?",
            chunks,
            stream=False
        )
        print("   ✓ Received response:")
        print("   " + "="*50)
        print(f"   Question: {response['question']}")
        print(f"   Answer: {response['answer']}")
        print(f"   Model: {response['model']}")
        print(f"   Timestamp: {response['timestamp']}")
        print(f"   Chunks used: {len(response['chunks_used'])} chunks")
        print("   " + "="*50)
    except Exception as e:
        print(f"   ✗ Error in non-streaming question: {e}")
    
    # Test streaming question
    print("\n5. Testing streaming question...")
    try:
        print("   Sending streaming question: 'How does the attention mechanism work?'")
        response = client.ask_question(
            "How does the attention mechanism work?",
            chunks,
            stream=True
        )
        print("\n   ✓ Streaming response completed")
        print("   " + "="*50)
        print(f"   Question: {response['question']}")
        print(f"   Answer length: {len(response['answer'])} characters")
        print(f"   Model: {response['model']}")
        print(f"   Timestamp: {response['timestamp']}")
        print(f"   Chunks used: {len(response['chunks_used'])} chunks")
        print("   " + "="*50)
    except Exception as e:
        print(f"   ✗ Error in streaming question: {e}")
    
    # Test error handling with empty chunks
    print("\n6. Testing error handling with empty chunks...")
    try:
        print("   Sending question with empty chunks...")
        response = client.ask_question("Will this fail?", [])
        print("   ✗ Expected error not raised")
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
    
    print("\n=== Testing complete ===")

if __name__ == "__main__":
    # Load environment variables from .env.test
    load_dotenv('.env.test')
    
    # Run the tests
    test_gemini_client()
