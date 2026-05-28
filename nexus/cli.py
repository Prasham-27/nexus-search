"""Command-line interface for nexus-search."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from nexus import __version__
from nexus.config import DEFAULT_INDEX_DIR, load_config
from nexus.display import (
    indexing_progress,
    show_clear_result,
    show_error,
    show_index_result,
    show_index_started,
    show_info,
    show_results,
    show_status,
    show_version,
)
from nexus.indexer import build_index, clear_index, get_index_stats, iter_indexable_files, load_index_metadata
from nexus.searcher import exact_search, index_exists, load_metadata, semantic_search

app = typer.Typer(help="Semantic and keyword search for your local filesystem.", no_args_is_help=True)


def version_callback(value: bool) -> None:
    """Print the package version and exit.

    Args:
        value: Whether the version flag was provided.
    """

    if value:
        show_version(__version__)
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
    watch: bool = typer.Option(False, "--watch", help="Keep running and reindex on file changes."),
) -> None:
    """Index a directory for semantic search.

    Args:
        path: File or directory to index.
        ext: Optional extension allow-list.
        watch: Whether to watch and reindex on file changes.
    """

    config = load_config()
    extensions = ext or list(config.default_extensions)
    _run_index_once(path, extensions, config.openai_model)
    if watch:
        _watch_and_reindex(path, extensions, config.openai_model)


def _run_index_once(path: Path, extensions: list[str], model: str) -> None:
    """Run one indexing pass with a Rich progress bar.

    Args:
        path: File or directory to index.
        extensions: Extension allow-list.
        model: OpenAI embedding model name.
    """

    show_index_started(path)
    try:
        candidate_count = sum(1 for _ in iter_indexable_files(path.expanduser().resolve(), extensions))
        with indexing_progress() as progress:
            task_id = progress.add_task("Walking files and chunking text...", total=candidate_count or None)

            def advance(_: Path) -> None:
                """Advance the indexing progress bar for one file.

                Args:
                    _: Processed file path.
                """

                progress.advance(task_id)

            result = build_index(path, extensions=extensions, model=model, on_file=advance)
            progress.update(task_id, description="Embedding chunks with OpenAI and writing FAISS index...")
    except Exception as exc:
        show_error(str(exc))
        raise typer.Exit(code=1) from exc
    show_index_result(result)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language query or exact regex pattern."),
    exact: bool = typer.Option(False, "--exact", help="Use keyword/regex search instead of embeddings."),
    top: int | None = typer.Option(None, "--top", "-n", help="Number of results to show."),
    ext: str | None = typer.Option(None, "--ext", "-e", help="Filter results by extension."),
    path: Path | None = typer.Option(None, "--path", "-p", help="Root path for --exact search."),
) -> None:
    """Search indexed files semantically or with an exact regex.

    Args:
        query: Natural language query or regex pattern.
        exact: Whether to bypass embeddings and use regex search.
        top: Maximum number of results to show.
        ext: Optional extension filter.
        path: Optional root path used for exact search.
    """

    config = load_config()
    limit = top or config.default_top_k
    if limit < 1:
        show_error("--top must be at least 1")
        raise typer.Exit(code=1)
    try:
        if exact:
            exact_root = path or _default_exact_root()
            results = exact_search(query, root=exact_root, top=limit, extensions=[ext] if ext else config.default_extensions)
        else:
            if not index_exists():
                show_error("no semantic index found; run 'nexus index <path>' first")
                raise typer.Exit(code=1)
            results = semantic_search(query, top=limit, ext=ext)
    except re.error as exc:
        show_error(f"invalid regex: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        show_error(str(exc))
        raise typer.Exit(code=1) from exc
    show_results(results)


@app.command()
def status() -> None:
    """Show local index statistics."""

    try:
        show_status(get_index_stats())
    except FileNotFoundError as exc:
        show_error("no semantic index found; run 'nexus index <path>' first")
        raise typer.Exit(code=1) from exc


@app.command()
def clear() -> None:
    """Delete the local semantic index."""

    show_clear_result(clear_index())


def _default_exact_root() -> Path:
    """Return the indexed root when available, otherwise the current directory.

    Returns:
        Root path for exact search.
    """

    if not index_exists():
        return Path(".")
    try:
        metadata = load_index_metadata(DEFAULT_INDEX_DIR)
    except FileNotFoundError:
        return Path(".")
    indexed_root = metadata.get("indexed_root")
    if isinstance(indexed_root, str) and indexed_root:
        return Path(indexed_root)
    _, chunks = load_metadata(DEFAULT_INDEX_DIR)
    if chunks:
        return Path(chunks[0].file_path).parent
    return Path(".")


def _watch_and_reindex(path: Path, extensions: list[str], model: str) -> None:
    """Watch a path and rebuild the index when matching files change.

    Args:
        path: File or directory to watch.
        extensions: Extension allow-list.
        model: OpenAI embedding model name.
    """

    watch_root = path.expanduser().resolve()
    if watch_root.is_file():
        watch_root = watch_root.parent
    handler = ReindexEventHandler(path, extensions, model)
    observer = Observer()
    observer.schedule(handler, str(watch_root), recursive=True)
    observer.start()
    show_info(f"Watching {watch_root} for changes. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        show_info("Stopping watcher.")
    finally:
        observer.stop()
        observer.join()


class ReindexEventHandler(FileSystemEventHandler):
    """Watchdog event handler that rebuilds the nexus index."""

    def __init__(self, path: Path, extensions: list[str], model: str) -> None:
        """Initialize the handler.

        Args:
            path: Original path to reindex.
            extensions: Extension allow-list.
            model: OpenAI embedding model name.
        """

        self.path = path
        self.extensions = {extension.lower().lstrip(".") for extension in extensions}
        self.model = model
        self.last_run = 0.0

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Reindex after a relevant file-system event.

        Args:
            event: Watchdog file-system event.
        """

        if event.is_directory:
            return
        event_path = Path(str(event.src_path))
        if event_path.suffix.lower().lstrip(".") not in self.extensions:
            return
        now = time.monotonic()
        if now - self.last_run < 1.0:
            return
        self.last_run = now
        show_info(f"Change detected in {event_path}; rebuilding index.")
        _run_index_once(self.path, list(self.extensions), self.model)


if __name__ == "__main__":
    app()
