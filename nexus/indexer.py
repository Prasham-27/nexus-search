"""Index local text files into a FAISS vector store."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import faiss
import numpy as np
import requests
import tiktoken
from openai import OpenAI

from nexus.config import DEFAULT_EXTENSIONS, DEFAULT_INDEX_DIR, DEFAULT_MODEL, normalize_extensions

LARGE_FILE_BYTES = 1_000_000
MAX_TOKENS_PER_CHUNK = 350
LARGE_FILE_MAX_TOKENS = 700
CHUNK_OVERLAP_LINES = 4
_TOKEN_ENCODING: tiktoken.Encoding | None = None


@dataclass(frozen=True)
class FileChunk:
    """A text chunk extracted from a source file."""

    chunk_id: str
    file_path: str
    line_start: int
    line_end: int
    text: str


@dataclass(frozen=True)
class IndexWarning:
    """A non-fatal indexing warning that can be displayed by the CLI."""

    path: str
    message: str


@dataclass(frozen=True)
class IndexResult:
    """Summary of an indexing run."""

    indexed_root: Path
    files_indexed: int
    chunks_indexed: int
    index_path: Path
    metadata_path: Path
    warnings: tuple[IndexWarning, ...]


@dataclass(frozen=True)
class IndexStats:
    """Statistics about the persisted local index."""

    total_files: int
    total_chunks: int
    last_updated: str | None
    index_size_bytes: int
    file_type_counts: tuple[tuple[str, int], ...]
    index_dir: Path
    indexed_root: str | None
    model: str | None


def iter_indexable_files(root: Path, extensions: Sequence[str]) -> Iterable[Path]:
    """Yield readable candidate files below a root path.

    Args:
        root: File or directory to index.
        extensions: Extension allow-list without leading dots.

    Yields:
        Paths that match the extension filter.
    """

    if root.is_file():
        if root.suffix.lower().lstrip(".") in set(extensions):
            yield root
        return

    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {".git", ".hg", ".svn", "__pycache__", ".venv", "node_modules"}
        ]
        for filename in filenames:
            path = Path(directory) / filename
            if path.suffix.lower().lstrip(".") in set(extensions):
                yield path


def is_binary_file(path: Path, sample_size: int = 2048) -> bool:
    """Return whether a file appears to be binary.

    Args:
        path: File path to inspect.
        sample_size: Number of bytes to sample.

    Returns:
        True when the sample contains null bytes.
    """

    with path.open("rb") as handle:
        return b"\0" in handle.read(sample_size)


def read_text_file(path: Path) -> str | None:
    """Read a UTF-8 text file, returning None when decoding fails.

    Args:
        path: File path to read.

    Returns:
        Decoded text or None for non-UTF-8 content.
    """

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _encoding() -> tiktoken.Encoding:
    """Return a tokenizer suitable for OpenAI embedding models.

    Returns:
        A tiktoken encoding instance.
    """

    global _TOKEN_ENCODING
    if _TOKEN_ENCODING is None:
        _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    return _TOKEN_ENCODING


def _count_tokens(text: str) -> int:
    """Count tokens with tiktoken, falling back when encoding data is unavailable.

    Args:
        text: Text to count.

    Returns:
        Approximate token count.
    """

    try:
        return max(1, len(_encoding().encode(text)))
    except (OSError, requests.RequestException):
        return max(1, len(text.split()))


def chunk_text(text: str, file_path: Path, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list[FileChunk]:
    """Split text into line-preserving token-limited chunks.

    Args:
        text: Source file content.
        file_path: Source file path.
        max_tokens: Maximum approximate tokens per chunk.

    Returns:
        A list of file chunks with stable IDs and line ranges.
    """

    lines = text.splitlines()
    chunks: list[FileChunk] = []
    current_lines: list[str] = []
    current_start = 1
    current_tokens = 0

    for index, line in enumerate(lines, start=1):
        token_count = _count_tokens(line)
        if current_lines and current_tokens + token_count > max_tokens:
            chunk_text_value = "\n".join(current_lines)
            chunks.append(_make_chunk(file_path, current_start, index - 1, chunk_text_value))
            overlap = current_lines[-CHUNK_OVERLAP_LINES:]
            current_start = max(current_start, index - len(overlap))
            current_lines = overlap.copy()
            current_tokens = sum(_count_tokens(item) for item in current_lines)

        current_lines.append(line)
        current_tokens += token_count

    if current_lines:
        chunks.append(_make_chunk(file_path, current_start, len(lines) or 1, "\n".join(current_lines)))

    return chunks


def _make_chunk(file_path: Path, line_start: int, line_end: int, text: str) -> FileChunk:
    """Create a chunk with a stable content-derived ID.

    Args:
        file_path: Source file path.
        line_start: First line in the chunk.
        line_end: Last line in the chunk.
        text: Chunk content.

    Returns:
        A populated FileChunk.
    """

    raw_id = f"{file_path.resolve()}:{line_start}:{line_end}:{hashlib.sha1(text.encode()).hexdigest()}"
    chunk_id = hashlib.sha1(raw_id.encode()).hexdigest()
    return FileChunk(
        chunk_id=chunk_id,
        file_path=str(file_path.resolve()),
        line_start=line_start,
        line_end=line_end,
        text=text,
    )


def collect_chunks(
    root: Path,
    extensions: Sequence[str],
    on_file: Callable[[Path], None] | None = None,
) -> tuple[list[FileChunk], list[IndexWarning], int]:
    """Collect indexable chunks from a file tree.

    Args:
        root: File or directory to index.
        extensions: Extension allow-list without leading dots.
        on_file: Optional callback invoked after each candidate file is processed.

    Returns:
        Chunks, warnings, and the number of files successfully indexed.
    """

    chunks: list[FileChunk] = []
    warnings: list[IndexWarning] = []
    files_indexed = 0

    for path in iter_indexable_files(root, extensions):
        try:
            if is_binary_file(path):
                if on_file:
                    on_file(path)
                continue
            max_tokens = MAX_TOKENS_PER_CHUNK
            if path.stat().st_size > LARGE_FILE_BYTES:
                warnings.append(IndexWarning(str(path), "large file; chunking conservatively"))
                max_tokens = LARGE_FILE_MAX_TOKENS
            text = read_text_file(path)
            if text is None:
                warnings.append(IndexWarning(str(path), "skipped non-UTF-8 text"))
                if on_file:
                    on_file(path)
                continue
            file_chunks = chunk_text(text, path, max_tokens=max_tokens)
        except PermissionError:
            warnings.append(IndexWarning(str(path), "permission denied"))
            if on_file:
                on_file(path)
            continue
        except OSError as exc:
            warnings.append(IndexWarning(str(path), f"skipped: {exc}"))
            if on_file:
                on_file(path)
            continue

        if file_chunks:
            chunks.extend(file_chunks)
            files_indexed += 1
        if on_file:
            on_file(path)

    return chunks, warnings, files_indexed


def embed_texts(texts: Sequence[str], model: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed text chunks with OpenAI.

    Args:
        texts: Text values to embed.
        model: OpenAI embedding model name.

    Returns:
        A float32 matrix of L2-normalized embeddings.
    """

    client = OpenAI()
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), 128):
        response = client.embeddings.create(model=model, input=list(texts[start : start + 128]))
        embeddings.extend(item.embedding for item in response.data)
    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def write_index(
    chunks: Sequence[FileChunk],
    vectors: np.ndarray,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model: str = DEFAULT_MODEL,
    indexed_root: Path | None = None,
) -> tuple[Path, Path]:
    """Persist a FAISS index and metadata sidecar.

    Args:
        chunks: Chunk metadata in vector order.
        vectors: Embedding matrix.
        index_dir: Directory where index files are written.
        model: Embedding model used for the vectors.
        indexed_root: Root path used to build the index.

    Returns:
        Paths to the FAISS index and metadata file.
    """

    index_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"
    faiss.write_index(index, str(index_path))
    files = sorted({chunk.file_path for chunk in chunks})
    chunk_map = {
        chunk.chunk_id: {
            "file_path": chunk.file_path,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "text": chunk.text,
        }
        for chunk in chunks
    }
    metadata = {
        "model": model,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "indexed_root": str(indexed_root.resolve()) if indexed_root else None,
        "files": files,
        "chunks": [asdict(chunk) for chunk in chunks],
        "chunk_map": chunk_map,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return index_path, metadata_path


def build_index(
    root: Path,
    extensions: Sequence[str] | None = None,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model: str = DEFAULT_MODEL,
    on_file: Callable[[Path], None] | None = None,
) -> IndexResult:
    """Build and persist a semantic search index for a path.

    Args:
        root: File or directory to index.
        extensions: Optional file extension allow-list.
        index_dir: Directory where index files are written.
        model: OpenAI embedding model name.
        on_file: Optional callback invoked as files are processed.

    Returns:
        Indexing summary.

    Raises:
        FileNotFoundError: If the root path does not exist.
        ValueError: If no indexable chunks are found.
    """

    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    chunks, warnings, files_indexed = collect_chunks(root, normalize_extensions(extensions), on_file=on_file)
    if not chunks:
        raise ValueError("no indexable text chunks found")

    vectors = embed_texts([chunk.text for chunk in chunks], model=model)
    index_path, metadata_path = write_index(chunks, vectors, index_dir=index_dir, model=model, indexed_root=root)
    return IndexResult(
        indexed_root=root,
        files_indexed=files_indexed,
        chunks_indexed=len(chunks),
        index_path=index_path,
        metadata_path=metadata_path,
        warnings=tuple(warnings),
    )


def load_index_metadata(index_dir: Path = DEFAULT_INDEX_DIR) -> dict[str, object]:
    """Load persisted index metadata.

    Args:
        index_dir: Directory containing metadata.json.

    Returns:
        Parsed metadata dictionary.
    """

    return json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))


def index_size_bytes(index_dir: Path = DEFAULT_INDEX_DIR) -> int:
    """Calculate the size of an index directory on disk.

    Args:
        index_dir: Directory containing index files.

    Returns:
        Total bytes for regular files in the index directory.
    """

    if not index_dir.exists():
        return 0
    return sum(path.stat().st_size for path in index_dir.rglob("*") if path.is_file())


def get_index_stats(index_dir: Path = DEFAULT_INDEX_DIR) -> IndexStats:
    """Return statistics for the local semantic index.

    Args:
        index_dir: Directory containing index files.

    Returns:
        Index statistics for display.

    Raises:
        FileNotFoundError: If metadata does not exist.
    """

    metadata = load_index_metadata(index_dir)
    files = [str(path) for path in metadata.get("files", [])]
    if not files:
        files = sorted({str(chunk["file_path"]) for chunk in metadata.get("chunks", []) if isinstance(chunk, dict)})
    counts: dict[str, int] = {}
    for file_path in files:
        extension = Path(file_path).suffix.lower().lstrip(".") or "[none]"
        counts[extension] = counts.get(extension, 0) + 1
    top_counts = tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5])
    return IndexStats(
        total_files=len(files),
        total_chunks=len(metadata.get("chunks", [])),
        last_updated=metadata.get("updated_at") if isinstance(metadata.get("updated_at"), str) else None,
        index_size_bytes=index_size_bytes(index_dir),
        file_type_counts=top_counts,
        index_dir=index_dir,
        indexed_root=metadata.get("indexed_root") if isinstance(metadata.get("indexed_root"), str) else None,
        model=metadata.get("model") if isinstance(metadata.get("model"), str) else None,
    )


def clear_index(index_dir: Path = DEFAULT_INDEX_DIR) -> bool:
    """Delete the local index directory.

    Args:
        index_dir: Directory to remove.

    Returns:
        True when an index directory was removed.
    """

    if not index_dir.exists():
        return False
    shutil.rmtree(index_dir)
    return True
