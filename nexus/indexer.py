"""Index local text files into a FAISS vector store."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import faiss
import numpy as np
import tiktoken
from openai import OpenAI

DEFAULT_EXTENSIONS = {"py", "js", "ts", "md", "txt", "json", "yaml", "yml"}
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_INDEX_DIR = Path.home() / ".nexus" / "index"
LARGE_FILE_BYTES = 1_000_000
MAX_TOKENS_PER_CHUNK = 350
CHUNK_OVERLAP_LINES = 4


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

    files_indexed: int
    chunks_indexed: int
    index_path: Path
    metadata_path: Path
    warnings: tuple[IndexWarning, ...]


def normalize_extensions(extensions: Sequence[str] | None) -> set[str]:
    """Normalize extension filters to bare lowercase names.

    Args:
        extensions: Optional user-provided extensions with or without dots.

    Returns:
        A set of normalized extension names.
    """

    if not extensions:
        return set(DEFAULT_EXTENSIONS)
    return {ext.lower().lstrip(".") for ext in extensions if ext.strip()}


def iter_indexable_files(root: Path, extensions: set[str]) -> Iterable[Path]:
    """Yield readable candidate files below a root path.

    Args:
        root: File or directory to index.
        extensions: Extension allow-list without leading dots.

    Yields:
        Paths that match the extension filter.
    """

    if root.is_file():
        if root.suffix.lower().lstrip(".") in extensions:
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
            if path.suffix.lower().lstrip(".") in extensions:
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

    return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, file_path: Path, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list[FileChunk]:
    """Split text into line-preserving token-limited chunks.

    Args:
        text: Source file content.
        file_path: Source file path.
        max_tokens: Maximum approximate tokens per chunk.

    Returns:
        A list of file chunks with stable IDs and line ranges.
    """

    encoder = _encoding()
    lines = text.splitlines()
    chunks: list[FileChunk] = []
    current_lines: list[str] = []
    current_start = 1
    current_tokens = 0

    for index, line in enumerate(lines, start=1):
        token_count = max(1, len(encoder.encode(line)))
        if current_lines and current_tokens + token_count > max_tokens:
            chunk_text_value = "\n".join(current_lines)
            chunks.append(_make_chunk(file_path, current_start, index - 1, chunk_text_value))
            overlap = current_lines[-CHUNK_OVERLAP_LINES:]
            current_start = max(current_start, index - len(overlap))
            current_lines = overlap.copy()
            current_tokens = sum(max(1, len(encoder.encode(item))) for item in current_lines)

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


def collect_chunks(root: Path, extensions: set[str]) -> tuple[list[FileChunk], list[IndexWarning], int]:
    """Collect indexable chunks from a file tree.

    Args:
        root: File or directory to index.
        extensions: Extension allow-list without leading dots.

    Returns:
        Chunks, warnings, and the number of files successfully indexed.
    """

    chunks: list[FileChunk] = []
    warnings: list[IndexWarning] = []
    files_indexed = 0

    for path in iter_indexable_files(root, extensions):
        try:
            if is_binary_file(path):
                continue
            if path.stat().st_size > LARGE_FILE_BYTES:
                warnings.append(IndexWarning(str(path), "large file; chunking conservatively"))
            text = read_text_file(path)
            if text is None:
                warnings.append(IndexWarning(str(path), "skipped non-UTF-8 text"))
                continue
            file_chunks = chunk_text(text, path)
        except PermissionError:
            warnings.append(IndexWarning(str(path), "permission denied"))
            continue
        except OSError as exc:
            warnings.append(IndexWarning(str(path), f"skipped: {exc}"))
            continue

        if file_chunks:
            chunks.extend(file_chunks)
            files_indexed += 1

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
    response = client.embeddings.create(model=model, input=list(texts))
    vectors = np.array([item.embedding for item in response.data], dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def write_index(
    chunks: Sequence[FileChunk],
    vectors: np.ndarray,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model: str = DEFAULT_MODEL,
) -> tuple[Path, Path]:
    """Persist a FAISS index and metadata sidecar.

    Args:
        chunks: Chunk metadata in vector order.
        vectors: Embedding matrix.
        index_dir: Directory where index files are written.
        model: Embedding model used for the vectors.

    Returns:
        Paths to the FAISS index and metadata file.
    """

    index_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"
    faiss.write_index(index, str(index_path))
    metadata = {
        "model": model,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return index_path, metadata_path


def build_index(
    root: Path,
    extensions: Sequence[str] | None = None,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model: str = DEFAULT_MODEL,
) -> IndexResult:
    """Build and persist a semantic search index for a path.

    Args:
        root: File or directory to index.
        extensions: Optional file extension allow-list.
        index_dir: Directory where index files are written.
        model: OpenAI embedding model name.

    Returns:
        Indexing summary.

    Raises:
        FileNotFoundError: If the root path does not exist.
        ValueError: If no indexable chunks are found.
    """

    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    chunks, warnings, files_indexed = collect_chunks(root, normalize_extensions(extensions))
    if not chunks:
        raise ValueError("no indexable text chunks found")

    vectors = embed_texts([chunk.text for chunk in chunks], model=model)
    index_path, metadata_path = write_index(chunks, vectors, index_dir=index_dir, model=model)
    return IndexResult(
        files_indexed=files_indexed,
        chunks_indexed=len(chunks),
        index_path=index_path,
        metadata_path=metadata_path,
        warnings=tuple(warnings),
    )
