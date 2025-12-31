import logging
from typing import Any, Dict, List

import faiss

from coderag.embeddings import generate_embeddings
from coderag.index import get_metadata, load_index

logger = logging.getLogger(__name__)


def search_code(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search the FAISS index using a text query.

    Args:
        query: The search query text
        k: Number of results to return (default: 5)

    Returns:
        List of search results with filename, filepath, content, and distance
    """
    try:
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        # Load the FAISS index
        index = load_index()
        if index is None:
            logger.error("Failed to load FAISS index")
            return []

        if index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []

        # Generate embedding for the query
        query_embedding = generate_embeddings(query)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return []
        # Normalize for cosine similarity (IndexFlatIP)
        faiss.normalize_L2(query_embedding)

        # Perform the search in FAISS
        k = min(k, index.ntotal)  # Don't search for more items than exist
        distances, indices = index.search(query_embedding, k)

        results = []
        metadata = get_metadata()

        for i, idx in enumerate(indices[0]):  # Iterate over the search results
            if 0 <= idx < len(metadata):  # Ensure the index is within bounds
                file_data = metadata[idx]
                results.append(
                    {
                        "filename": file_data["filename"],
                        "filepath": file_data["filepath"],
                        "content": file_data["content"],
                        "distance": float(distances[0][i]),  # Convert to Python float
                    }
                )
            else:
                logger.warning(
                    f"Index {idx} is out of bounds for metadata with length "
                    f"{len(metadata)}"
                )

        logger.debug(
            f"Search returned {len(results)} results for query: " f"'{query[:50]}...'"
        )
        return results

    except Exception as e:
        logger.error(f"Error during code search: {str(e)}")
        return []
