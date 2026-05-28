"""Integration tests for the Typer CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pytest_mock import MockerFixture
from typer.testing import CliRunner

from nexus.cli import app
from nexus.indexer import FileChunk, get_index_stats, write_index

runner = CliRunner()


def test_status_output_format(tmp_path: Path, mocker: MockerFixture) -> None:
    """The status command should render index counts in a Rich panel.

    Args:
        tmp_path: Pytest temporary directory fixture.
        mocker: pytest-mock fixture.
    """

    index_dir = tmp_path / "index"
    chunk = FileChunk("one", str(tmp_path / "auth.py"), 1, 1, "def authenticate_user(): pass")
    write_index(
        [chunk],
        np.array([[1.0, 0.0]], dtype="float32"),
        index_dir=index_dir,
        indexed_root=tmp_path,
    )
    mocker.patch("nexus.cli.get_index_stats", lambda: get_index_stats(index_dir))

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "nexus status" in result.output
    assert "Files indexed" in result.output
    assert "Chunks indexed" in result.output


def test_search_exact_returns_correct_matches(sample_project: Path) -> None:
    """The CLI should return exact search matches.

    Args:
        sample_project: Sample project fixture.
    """

    result = runner.invoke(
        app,
        ["search", "authenticate_user", "--exact", "--path", str(sample_project), "--ext", "py"],
    )

    assert result.exit_code == 0
    assert "auth.py" in result.output
    assert "authenticate_user" in result.output
