import logging
import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from coderag.config import EMBEDDING_DIM, FAISS_INDEX_FILE, WATCHED_DIR

logger = logging.getLogger(__name__)

index = faiss.IndexFlatIP(EMBEDDING_DIM)
metadata: List[Dict[str, Any]] = []


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length in-place, returns the same array."""
    if mat is None or mat.size == 0:
        return mat
    faiss.normalize_L2(mat)
    return mat


def clear_index() -> None:
    """Delete the FAISS index and metadata files if they exist, and
    reinitialize the index."""
    global index, metadata

    try:
        # Delete the FAISS index file
        if os.path.exists(FAISS_INDEX_FILE):
            os.remove(FAISS_INDEX_FILE)
            logger.info(f"Deleted FAISS index file: {FAISS_INDEX_FILE}")

        # Delete the metadata file
        metadata_file = "metadata.npy"
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
            logger.info(f"Deleted metadata file: {metadata_file}")

        # Reinitialize the FAISS index and metadata
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        metadata = []
        logger.info("FAISS index and metadata cleared and reinitialized")

    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        raise


def add_to_index(
    embeddings: np.ndarray, full_content: str, filename: str, filepath: str
) -> None:
    """Add embeddings and metadata to the FAISS index.

    Args:
        embeddings: The embedding vectors to add
        full_content: The original file content
        filename: Name of the file
        filepath: Full path to the file
    """

    try:
        if embeddings is None or embeddings.size == 0:
            logger.warning(f"Empty embeddings provided for {filename}")
            return

        if embeddings.shape[1] != index.d:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"FAISS index dimension {index.d}"
            )

        # Convert absolute filepath to relative path
        try:
            relative_filepath = os.path.relpath(filepath, WATCHED_DIR)
        except ValueError:
            logger.warning(
                f"Could not create relative path for {filepath}, using "
                f"absolute path"
            )
            relative_filepath = filepath

        # Normalize for cosine similarity (IndexFlatIP)
        vecs = embeddings.astype("float32", copy=True)
        vecs = _l2_normalize(vecs)
        index.add(vecs)
        metadata.append(
            {
                # Store only a snippet to keep metadata small
                "content": (full_content[:3000] if full_content else ""),
                "filename": filename,
                "filepath": relative_filepath,
            }
        )

        logger.debug(f"Added {filename} to index (total entries: {index.ntotal})")

    except Exception as e:
        logger.error(f"Error adding {filename} to index: {str(e)}")
        raise


def save_index() -> None:
    """Save the FAISS index and metadata to disk."""
    try:
        faiss.write_index(index, FAISS_INDEX_FILE)
        with open("metadata.npy", "wb") as f:
            np.save(f, np.array(metadata, dtype=object))
        logger.debug(f"Index saved with {index.ntotal} entries")
    except Exception as e:
        logger.error(f"Error saving index: {str(e)}")
        raise


def load_index() -> Optional[faiss.Index]:
    """Load the FAISS index and metadata from disk.

    Returns:
        The loaded FAISS index or a new empty index if loading fails
    """
    global index, metadata

    try:
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists("metadata.npy"):
            index = faiss.read_index(FAISS_INDEX_FILE)
            with open("metadata.npy", "rb") as f:
                metadata = np.load(f, allow_pickle=True).tolist()
            logger.info(f"Loaded index with {index.ntotal} entries")
            return index
        else:
            if not os.path.exists(FAISS_INDEX_FILE):
                logger.warning(f"FAISS index file not found: {FAISS_INDEX_FILE}")
            if not os.path.exists("metadata.npy"):
                logger.warning("Metadata file not found: metadata.npy")
            
            # Create new empty index and metadata if files don't exist
            logger.info("Creating new empty FAISS index")
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            metadata = []
            
            # Save the new empty index to disk
            try:
                faiss.write_index(index, FAISS_INDEX_FILE)
                with open("metadata.npy", "wb") as f:
                    np.save(f, np.array(metadata, dtype=object))
                logger.info(f"Created new empty index at: {FAISS_INDEX_FILE}")
            except Exception as save_e:
                logger.error(f"Error saving new index: {str(save_e)}")
                
            return index

    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")
        # Return the existing in-memory index as fallback
        return index


def get_metadata() -> List[Dict[str, Any]]:
    """Get the current metadata list.

    Returns:
        List of metadata dictionaries
    """
    return metadata


def retrieve_vectors(n=5):
    n = min(n, index.ntotal)
    vectors = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    for i in range(n):
        vectors[i] = index.reconstruct(i)
    return vectors


def inspect_metadata(n: int = 5) -> None:
    """Print metadata information for debugging purposes.

    Args:
        n: Number of entries to inspect
    """
    try:
        metadata_list = get_metadata()
        logger.info(f"Inspecting the first {n} metadata entries:")
        for i, data in enumerate(metadata_list[:n]):
            logger.info(f"Entry {i}:")
            logger.info(f"  Filename: {data['filename']}")
            logger.info(f"  Filepath: {data['filepath']}")
            logger.info(
                f"  Content: {data['content'][:100]}..."
            )  # Show the first 100 characters
    except Exception as e:
        logger.error(f"Error inspecting metadata: {str(e)}")
