import faiss
import numpy as np

from coderag.config import EMBEDDING_DIM
from coderag.index import add_to_index, clear_index, save_index
from coderag.search import search_code


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


def test_search_returns_result(tmp_path, monkeypatch):
    _isolate_index(tmp_path, monkeypatch)
    clear_index()

    vector = np.ones((1, EMBEDDING_DIM), dtype=np.float32)
    add_to_index(vector, "print('hello world')", "sample.py", "sample.py")
    save_index()

    def fake_generate_embeddings(_: str):
        embedding = np.ones((1, EMBEDDING_DIM), dtype=np.float32)
        faiss.normalize_L2(embedding)
        return embedding

    monkeypatch.setattr("coderag.search.generate_embeddings", fake_generate_embeddings)

    results = search_code("hello", k=1)
    assert len(results) == 1
    first = results[0]
    assert first["filename"] == "sample.py"
    assert "sample.py" in first["filepath"]
    assert first["distance"] > 0

    clear_index()


def test_search_empty_query_returns_empty():
    assert search_code(" ") == []


def test_search_missing_index_returns_empty(tmp_path, monkeypatch):
    _isolate_index(tmp_path, monkeypatch)
    clear_index()
    results = search_code("whatever")
    assert results == []
