"""Search a nexus FAISS index or perform exact regex search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from nexus.config import DEFAULT_INDEX_DIR, DEFAULT_MODEL, normalize_extensions
from nexus.indexer import FileChunk, embed_texts, is_binary_file, iter_indexable_files, load_index_metadata, read_text_file


@dataclass(frozen=True)
class SearchResult:
    """A ranked search hit."""

    rank: int
    file_path: str
    line_start: int
    line_end: int
    score: float
    snippet: str


def load_metadata(index_dir: Path = DEFAULT_INDEX_DIR) -> tuple[str, list[FileChunk]]:
    """Load chunk metadata for a FAISS index.

    Args:
        index_dir: Directory containing metadata.json.

    Returns:
        Embedding model name and chunks in vector order.
    """

    data = load_index_metadata(index_dir)
    chunks = [FileChunk(**item) for item in data.get("chunks", [])]
    return data.get("model", DEFAULT_MODEL), chunks


def semantic_search(query: str, top: int = 5, index_dir: Path = DEFAULT_INDEX_DIR, ext: str | None = None) -> list[SearchResult]:
    """Search indexed chunks by semantic similarity.

    Args:
        query: Natural language query.
        top: Maximum number of results.
        index_dir: Directory containing the FAISS index.
        ext: Optional extension filter.

    Returns:
        Ranked semantic matches.
    """

    index_path = index_dir / "faiss.index"
    model, chunks = load_metadata(index_dir)
    index = faiss.read_index(str(index_path))
    query_vector = embed_texts([query], model=model)
    scores, indices = index.search(query_vector, min(top * 4, len(chunks)))

    results: list[SearchResult] = []
    normalized_ext = ext.lower().lstrip(".") if ext else None
    for vector_index, score in zip(indices[0], scores[0], strict=False):
        if vector_index < 0:
            continue
        chunk = chunks[int(vector_index)]
        if normalized_ext and Path(chunk.file_path).suffix.lower().lstrip(".") != normalized_ext:
            continue
        results.append(
            SearchResult(
                rank=len(results) + 1,
                file_path=chunk.file_path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                score=float(score),
                snippet=chunk.text,
            )
        )
        if len(results) >= top:
            break
    return results


def exact_search(query: str, root: Path, top: int = 5, extensions: Sequence[str] | None = None) -> list[SearchResult]:
    """Search files with a regular expression.

    Args:
        query: Regex pattern to match.
        root: File or directory to search.
        top: Maximum number of results.
        extensions: Optional extension allow-list.

    Returns:
        Regex matches as ranked results.
    """

    pattern = re.compile(query, re.IGNORECASE)
    results: list[SearchResult] = []
    for path in iter_indexable_files(root.expanduser().resolve(), normalize_extensions(extensions)):
        try:
            if is_binary_file(path):
                continue
            text = read_text_file(path)
        except (OSError, PermissionError):
            continue
        if text is None:
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines, start=1):
            if not pattern.search(line):
                continue
            start = max(1, index - 2)
            end = min(len(lines), index + 2)
            snippet = "\n".join(lines[start - 1 : end])
            results.append(
                SearchResult(
                    rank=len(results) + 1,
                    file_path=str(path.resolve()),
                    line_start=start,
                    line_end=end,
                    score=1.0,
                    snippet=snippet,
                )
            )
            if len(results) >= top:
                return results
    return results


def index_exists(index_dir: Path = DEFAULT_INDEX_DIR) -> bool:
    """Return whether a usable semantic index exists.

    Args:
        index_dir: Directory containing index files.

    Returns:
        True when both FAISS and metadata files exist.
    """

    return (index_dir / "faiss.index").exists() and (index_dir / "metadata.json").exists()


def cosine_order_for_vectors(query_vector: np.ndarray, matrix: np.ndarray) -> list[int]:
    """Rank normalized vectors by cosine similarity.

    Args:
        query_vector: A single query vector.
        matrix: Candidate vectors.

    Returns:
        Candidate indices ordered from most to least similar.
    """

    scores = matrix @ query_vector
    return list(np.argsort(scores)[::-1])
