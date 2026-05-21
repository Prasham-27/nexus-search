"""Command-line interface for nexus-search."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from nexus import __version__
from nexus.display import indexing_progress, show_error, show_index_result, show_index_started, show_results
from nexus.indexer import build_index
from nexus.searcher import exact_search, index_exists, semantic_search

app = typer.Typer(help="Semantic and keyword search for your local filesystem.", no_args_is_help=True)


def version_callback(value: bool) -> None:
    """Print the package version and exit.

    Args:
        value: Whether the version flag was provided.
    """

    if value:
        typer.echo(f"nexus-search {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the version and exit.",
    )
) -> None:
    """Configure global CLI behavior.

    Args:
        version: Optional version flag.
    """

    load_dotenv()


@app.command()
def index(
    path: Path = typer.Argument(..., help="File or directory to index recursively."),
    ext: list[str] | None = typer.Option(None, "--ext", "-e", help="Extension to include; can be repeated."),
) -> None:
    """Index a directory for semantic search.

    Args:
        path: File or directory to index.
        ext: Optional extension allow-list.
    """

    show_index_started(path)
    try:
        with indexing_progress() as progress:
            progress.add_task("Embedding chunks with OpenAI and writing FAISS index...", total=None)
            result = build_index(path, extensions=ext)
    except Exception as exc:
        show_error(str(exc))
        raise typer.Exit(code=1) from exc
    show_index_result(result)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language query or exact regex pattern."),
    exact: bool = typer.Option(False, "--exact", help="Use keyword/regex search instead of embeddings."),
    top: int = typer.Option(5, "--top", "-n", min=1, help="Number of results to show."),
    ext: str | None = typer.Option(None, "--ext", "-e", help="Filter results by extension."),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Root path for --exact search."),
) -> None:
    """Search indexed files semantically or with an exact regex.

    Args:
        query: Natural language query or regex pattern.
        exact: Whether to bypass embeddings and use regex search.
        top: Maximum number of results to show.
        ext: Optional extension filter.
        path: Root path used for exact search.
    """

    try:
        if exact:
            results = exact_search(query, root=path, top=top, extensions=[ext] if ext else None)
        else:
            if not index_exists():
                show_error("no semantic index found; run 'nexus index <path>' first")
                raise typer.Exit(code=1)
            results = semantic_search(query, top=top, ext=ext)
    except re.error as exc:  # type: ignore[name-defined]
        show_error(f"invalid regex: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        show_error(str(exc))
        raise typer.Exit(code=1) from exc
    show_results(results)


if __name__ == "__main__":
    app()
