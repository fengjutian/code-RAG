import logging
from typing import List, Optional

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from coderag.config import (
    DEEPSEEK_API_BASE, 
    DEEPSEEK_API_KEY, 
    DEEPSEEK_EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)

# Initialize Embedding Client (supports DeepSeek and local models)
class EmbeddingClient:
    def __init__(self):
        self.client = None
        self.model = None
        self.local_model = None
        self.use_local = False
        
        # Try to initialize DeepSeek client first
        try:
            if DEEPSEEK_API_KEY and DEEPSEEK_EMBEDDING_MODEL and DEEPSEEK_API_BASE:
                self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
                self.model = DEEPSEEK_EMBEDDING_MODEL
                logger.info(f"DeepSeek client initialized with model: {self.model} (for embeddings)")
                logger.info(f"DeepSeek API Base: {DEEPSEEK_API_BASE}")
            else:
                logger.warning("DeepSeek configuration incomplete. Falling back to local embedding model.")
                self.use_local = True
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}. Falling back to local embedding model.")
            self.use_local = True
        
        # Initialize local model if needed
        if self.use_local:
            try:
                # Use a higher dimensional model that matches FAISS index dimension
                self.local_model = SentenceTransformer('all-mpnet-base-v2')  # 768维
                logger.info(f"Local embedding model initialized: all-mpnet-base-v2 (768 dimensions)")
                
                # Override embedding dimension to match local model
                global EMBEDDING_DIM
                EMBEDDING_DIM = 768
                logger.info(f"Embedding dimension set to: {EMBEDDING_DIM}")
                
            except Exception as e:
                logger.error(f"Failed to initialize local embedding model: {e}")
                # Fallback to smaller model
                try:
                    self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info(f"Fallback model initialized: all-MiniLM-L6-v2 (384 dimensions)")
                except Exception as e2:
                    logger.error(f"Failed to initialize fallback model: {e2}")

# Create embedding client instance
embedding_client = EmbeddingClient()


def _chunk_text(text: str, max_chars: int = 4000, overlap: int = 50) -> List[str]:
    """Improved chunking that respects sentence boundaries.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    import re
    
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    
    # Split by sentences (rudimentary but better than character-based)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed max_chars
        if current_length + sentence_length > max_chars:
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Calculate overlap from the end of current chunk
                if len(chunk_text) > overlap:
                    overlap_content = chunk_text[-overlap:]
                    current_chunk = [overlap_content, sentence]
                    current_length = len(overlap_content) + sentence_length + 1  # +1 for space
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                # Single sentence too long, split by words
                words = sentence.split()
                for i in range(0, len(words), max_chars):
                    chunks.append(' '.join(words[i:i+max_chars]))
                current_chunk = []
                current_length = 0
        else:
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=8),
    reraise=True,
)
def _embed_batch(inputs: List[str]) -> np.ndarray:
    """Generate embeddings using either DeepSeek API or local model. Returns shape (n, d)."""
    # Check if using local model
    if embedding_client.use_local:
        if embedding_client.local_model is None:
            raise RuntimeError("Local embedding model not initialized")
        
        logger.info(f"Using local embedding model: all-MiniLM-L6-v2")
        logger.info(f"Input texts count: {len(inputs)}")
        logger.info(f"First input sample (truncated): {inputs[0][:100]}...")
        
        # Use local model to generate embeddings
        embeddings = embedding_client.local_model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.array(embeddings, dtype="float32")
    
    # Fall back to DeepSeek API
    if embedding_client.client is None:
        raise RuntimeError("DeepSeek embedding client not initialized")
    
    # Log API call details for debugging
    logger.info(f"Calling DeepSeek embedding API with model: {embedding_client.model}")
    logger.info(f"DeepSeek API Base URL: {DEEPSEEK_API_BASE}")
    logger.info(f"Input texts count: {len(inputs)}")
    logger.info(f"First input sample (truncated): {inputs[0][:100]}...")
    
    response = embedding_client.client.embeddings.create(
        model=embedding_client.model,
        input=inputs,
        timeout=30,
    )
    arr = np.array([d.embedding for d in response.data], dtype="float32")
    return arr


def generate_embeddings(text: str) -> Optional[np.ndarray]:
    """Generate embeddings using either DeepSeek API or local model.

    Args:
        text: The input text to generate embeddings for

    Returns:
        numpy array of embeddings or None if generation fails
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding generation")
        return None

    # Ensure local model is initialized as fallback
    if not embedding_client.local_model:
        try:
            embedding_client.local_model = SentenceTransformer('all-mpnet-base-v2')
            logger.info("Local embedding model initialized as fallback: all-mpnet-base-v2")
            
            # Update embedding dimension to match the model
            global EMBEDDING_DIM
            EMBEDDING_DIM = 768
            logger.info(f"Embedding dimension set to: {EMBEDDING_DIM}")
            
        except Exception as e:
            logger.error(f"Failed to initialize local embedding model: {e}")
            return None

    try:
        logger.info(f"Generating embeddings for text of length: {len(text)}")

        chunks = _chunk_text(text, max_chars=4000)
        vecs = _embed_batch(chunks)  # shape (n, d)

        # Average chunk embeddings for a stable single vector
        avg = np.mean(vecs, axis=0, dtype=np.float32).reshape(1, -1)
        logger.info(f"Successfully generated embeddings with shape: {avg.shape}")
        return avg

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        # Detailed debugging for 404 errors
        if "404" in str(e) and not embedding_client.use_local:
            logger.error("DeepSeek API returned 404 error. Switching to local embedding model...")
            logger.error(f"1. API Base URL: {DEEPSEEK_API_BASE}")
            logger.error(f"2. Model name: {DEEPSEEK_EMBEDDING_MODEL}")
            logger.error(f"3. API Key: {'Valid format' if DEEPSEEK_API_KEY and len(DEEPSEEK_API_KEY) > 10 else 'Invalid or missing'}")
            logger.error(f"4. Network connectivity to {DEEPSEEK_API_BASE}")
            
            # Switch to local model for future requests
            embedding_client.use_local = True
            
            # Try again with local model
            try:
                logger.info("Retrying with local embedding model...")
                chunks = _chunk_text(text, max_chars=4000)
                vecs = _embed_batch(chunks)
                avg = np.mean(vecs, axis=0, dtype=np.float32).reshape(1, -1)
                logger.info(f"Successfully generated embeddings with local model, shape: {avg.shape}")
                return avg
            except Exception as local_e:
                logger.error(f"Local embedding generation also failed: {local_e}")
                return None
        
        # For other errors, try local model directly if not already using it
        if not embedding_client.use_local:
            logger.error("DeepSeek API failed. Switching to local embedding model...")
            embedding_client.use_local = True
            
            # Try again with local model
            try:
                logger.info("Retrying with local embedding model...")
                chunks = _chunk_text(text, max_chars=4000)
                vecs = _embed_batch(chunks)
                avg = np.mean(vecs, axis=0, dtype=np.float32).reshape(1, -1)
                logger.info(f"Successfully generated embeddings with local model, shape: {avg.shape}")
                return avg
            except Exception as local_e:
                logger.error(f"Local embedding generation also failed: {local_e}")
                return None
        
        return None
