import os

import faiss
import numpy as np

from coderag.config import EMBEDDING_DIM
from coderag.index import add_to_index, clear_index, get_metadata, save_index


def _isolate_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("coderag.index.WATCHED_DIR", str(tmp_path), raising=False)
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


def test_add_to_index_tracks_relative_paths(tmp_path, monkeypatch):
    _isolate_index(tmp_path, monkeypatch)

    clear_index()

    embeddings = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
    file_path = tmp_path / "pkg" / "module.py"
    add_to_index(embeddings, "print('hi')", "module.py", str(file_path))

    metadata = get_metadata()
    assert metadata
    assert metadata[0]["filepath"] == os.path.join("pkg", "module.py")

    save_index()
    assert os.path.exists(tmp_path / "coderag_index.faiss")

    clear_index()
