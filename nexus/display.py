"""Rich display helpers for nexus-search."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table

from nexus.indexer import IndexResult, IndexStats, IndexWarning
from nexus.searcher import SearchResult

console = Console()


def show_index_started(path: Path) -> None:
    """Display a short indexing start message.

    Args:
        path: Path being indexed.
    """

    console.print(f"[bold cyan]Indexing[/bold cyan] {path.expanduser()}")


def indexing_progress() -> Progress:
    """Create a Rich progress indicator for indexing.

    Returns:
        A configured Progress instance.
    """

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def show_index_result(result: IndexResult) -> None:
    """Display an indexing summary.

    Args:
        result: Indexing result to render.
    """

    console.print(
        Panel.fit(
            f"[bold green]Index complete[/bold green]\n"
            f"Files indexed: [bold]{result.files_indexed}[/bold]\n"
            f"Chunks indexed: [bold]{result.chunks_indexed}[/bold]\n"
            f"Index: {result.index_path}",
            title="nexus",
        )
    )
    show_warnings(result.warnings)


def show_warnings(warnings: tuple[IndexWarning, ...]) -> None:
    """Display non-fatal indexing warnings.

    Args:
        warnings: Warning records emitted during indexing.
    """

    for warning in warnings[:10]:
        console.print(f"[yellow]warning:[/yellow] {warning.path}: {warning.message}")
    if len(warnings) > 10:
        console.print(f"[yellow]warning:[/yellow] {len(warnings) - 10} more warnings omitted")


def show_error(message: str) -> None:
    """Display an error message.

    Args:
        message: Error text.
    """

    console.print(f"[bold red]error:[/bold red] {message}")


def show_info(message: str) -> None:
    """Display an informational message.

    Args:
        message: Message text.
    """

    console.print(f"[cyan]{message}[/cyan]")


def show_version(version: str) -> None:
    """Display the package version.

    Args:
        version: Version string.
    """

    console.print(f"nexus-search {version}")


def show_results(results: list[SearchResult]) -> None:
    """Render search results as a Rich table.

    Args:
        results: Search hits to render.
    """

    if not results:
        console.print("[yellow]No matches found.[/yellow]")
        return

    table = Table(title="nexus results", show_lines=True)
    table.add_column("Rank", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("File", overflow="fold")
    table.add_column("Lines", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Snippet", overflow="fold")

    for result in results:
        lexer = _lexer_for_path(Path(result.file_path))
        syntax = Syntax(result.snippet, lexer, line_numbers=False, word_wrap=True)
        table.add_row(
            str(result.rank),
            result.file_path,
            f"{result.line_start}-{result.line_end}",
            f"{result.score:.3f}",
            syntax,
        )
    console.print(table)


def show_status(stats: IndexStats) -> None:
    """Render index statistics as a Rich panel.

    Args:
        stats: Index statistics to display.
    """

    file_types = ", ".join(f".{extension}: {count}" for extension, count in stats.file_type_counts)
    if not file_types:
        file_types = "none"
    body = (
        f"Files indexed: [bold]{stats.total_files}[/bold]\n"
        f"Chunks indexed: [bold]{stats.total_chunks}[/bold]\n"
        f"Last updated: {stats.last_updated or 'unknown'}\n"
        f"Index size: {_format_bytes(stats.index_size_bytes)}\n"
        f"Indexed root: {stats.indexed_root or 'unknown'}\n"
        f"Model: {stats.model or 'unknown'}\n"
        f"Top file types: {file_types}"
    )
    console.print(Panel.fit(body, title="nexus status"))


def show_clear_result(removed: bool) -> None:
    """Display the result of clearing the index.

    Args:
        removed: Whether an index was removed.
    """

    if removed:
        console.print("[bold green]Local nexus index deleted.[/bold green]")
    else:
        console.print("[yellow]No local nexus index found.[/yellow]")


def _lexer_for_path(path: Path) -> str:
    """Infer a Pygments lexer from a file extension.

    Args:
        path: File path used for lexer selection.

    Returns:
        A lexer name supported by Rich/Pygments.
    """

    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    return mapping.get(path.suffix.lower(), "text")


def _format_bytes(size: int) -> str:
    """Format bytes in a compact human-readable form.

    Args:
        size: Number of bytes.

    Returns:
        Human-readable size string.
    """

    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"
