"""Unit tests for semantic and exact search."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from pytest_mock import MockerFixture

from nexus.indexer import FileChunk, write_index
from nexus.searcher import exact_search, semantic_search


def test_semantic_search_ranking_uses_cosine_similarity(tmp_path: Path, mocker: MockerFixture) -> None:
    """Semantic search should return chunks ordered by vector similarity.

    Args:
        tmp_path: Pytest temporary directory fixture.
        mocker: pytest-mock fixture.
    """

    chunks = [
        FileChunk("auth", str(tmp_path / "auth.py"), 1, 2, "authentication token bug"),
        FileChunk("billing", str(tmp_path / "billing.py"), 1, 2, "invoice payment flow"),
        FileChunk("mixed", str(tmp_path / "mixed.py"), 1, 2, "auth payment edge case"),
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype="float32")
    faiss.normalize_L2(vectors)
    write_index(chunks, vectors, index_dir=tmp_path / "index", indexed_root=tmp_path)
    query = np.array([[1.0, 0.0]], dtype="float32")
    faiss.normalize_L2(query)
    mocker.patch("nexus.searcher.embed_texts", return_value=query)

    results = semantic_search("authentication bug", top=3, index_dir=tmp_path / "index")

    assert [Path(result.file_path).name for result in results] == ["auth.py", "mixed.py", "billing.py"]


def test_exact_search_returns_regex_matches(sample_project: Path) -> None:
    """Exact search should return regex matches with line snippets.

    Args:
        sample_project: Sample project fixture.
    """

    results = exact_search("authenticate_user", root=sample_project, extensions=("py",))

    assert len(results) == 1
    assert Path(results[0].file_path).name == "auth.py"
    assert "authenticate_user" in results[0].snippet
