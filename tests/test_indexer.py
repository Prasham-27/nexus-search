"""Unit tests for file walking and chunk collection."""

from __future__ import annotations

from pathlib import Path

from nexus.indexer import chunk_text, collect_chunks, iter_indexable_files


def test_file_walking_respects_extension_filters(sample_project: Path) -> None:
    """File walking should only yield files with allowed extensions.

    Args:
        sample_project: Sample project fixture.
    """

    files = {path.name for path in iter_indexable_files(sample_project, ("md",))}

    assert files == {"README.md", "incidents.md"}


def test_binary_files_are_skipped(sample_project: Path) -> None:
    """Binary files with allowed extensions should not become chunks.

    Args:
        sample_project: Sample project fixture.
    """

    chunks, warnings, files_indexed = collect_chunks(sample_project, ("py",))
    indexed_paths = {Path(chunk.file_path).name for chunk in chunks}

    assert "binary.py" not in indexed_paths
    assert files_indexed == 3
    assert warnings == []


def test_chunk_count_is_deterministic_for_known_file(sample_project: Path) -> None:
    """A small known file should produce exactly one chunk.

    Args:
        sample_project: Sample project fixture.
    """

    text = (sample_project / "auth.py").read_text(encoding="utf-8")
    chunks = chunk_text(text, sample_project / "auth.py")

    assert len(chunks) == 1
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 4
