"""Rich display helpers for nexus-search."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from nexus.indexer import IndexResult, IndexWarning
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

    return Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console)


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
