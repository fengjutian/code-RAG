import faiss
import numpy as np

from coderag.config import EMBEDDING_DIM
from coderag.index import (
    add_to_index,
    clear_index,
    inspect_metadata,
    load_index,
    retrieve_vectors,
    save_index,
)


def _isolate_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "coderag.index.FAISS_INDEX_FILE",
        str(tmp_path / "coderag_index.faiss"),
        raising=False,
    )
    monkeypatch.setattr("coderag.index.metadata", [], raising=False)
    monkeypatch.setattr(
        "coderag.index.index",
        faiss.IndexFlatIP(EMBEDDING_DIM),
        raising=False,
    )


def test_faiss_index(tmp_path, monkeypatch):
    _isolate_index(tmp_path, monkeypatch)
    # Clear the index before testing
    clear_index()

    # Create a deterministic dummy embedding (no network needed)
    vec = np.ones((1, EMBEDDING_DIM), dtype=np.float32)
    # Add to index with small dummy content
    add_to_index(vec, "dummy content", "test_file.py", "test_file.py")
    save_index()

    # Load the index
    index = load_index()
    assert index is not None, "Failed to load FAISS index."
    # Check if index has vectors
    assert index.ntotal > 0, "FAISS index is empty. No vectors found!"
    print(f"FAISS index has {index.ntotal} vectors.")

    # Retrieve and inspect vectors
    vectors = retrieve_vectors(5)
    print(f"Retrieved {len(vectors)} vectors from the index.")

    # Inspect metadata
    inspect_metadata(5)
