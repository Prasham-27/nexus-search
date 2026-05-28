# nexus-search

*Semantic and keyword search for your local filesystem.*

## Quick Demo

```text
$ nexus index ~/projects
Indexing /Users/you/projects
╭──────────── nexus ────────────╮
│ Index complete                │
│ Files indexed: 128            │
│ Chunks indexed: 742           │
│ Index: ~/.nexus/index/faiss.index │
╰───────────────────────────────╯

$ nexus search "authentication bug"
┌──────┬──────────────────────────────┬────────┬───────┬──────────────────────────────┐
│ Rank │ File                         │ Lines  │ Score │ Snippet                      │
├──────┼──────────────────────────────┼────────┼───────┼──────────────────────────────┤
│    1 │ ~/projects/api/auth.py       │ 42-61  │ 0.842 │ def authenticate_user(...):  │
│    2 │ ~/projects/docs/incidents.md │ 11-24  │ 0.817 │ Token refresh failure...     │
└──────┴──────────────────────────────┴────────┴───────┴──────────────────────────────┘

$ nexus search "def authenticate" --exact --ext py
┌──────┬────────────────────────┬───────┬───────┬────────────────────────────┐
│ Rank │ File                   │ Lines │ Score │ Snippet                    │
├──────┼────────────────────────┼───────┼───────┼────────────────────────────┤
│    1 │ ~/projects/api/auth.py │ 40-44 │ 1.000 │ def authenticate_user(...):│
└──────┴────────────────────────┴───────┴───────┴────────────────────────────┘
```

## Why nexus?

Developers often remember what code does before they remember its exact symbol names. nexus makes that memory useful by combining semantic embeddings for concept-level search with regex search for exact matches. It keeps the vector index local in FAISS, stores transparent metadata beside it, and renders results in a terminal-native interface that is quick to scan. The result is a practical "grep meets AI" workflow for codebases, docs, notes, and configuration files.

## Features

- Recursive indexing with a Rich progress bar
- Semantic search powered by OpenAI `text-embedding-3-small`
- Local FAISS vector index stored at `~/.nexus/index/faiss.index`
- Exact keyword and regex search with `--exact`
- Extension filtering for indexing and search with `--ext`
- `nexus status` index statistics with file counts, chunk counts, size, timestamp, and top file types
- `nexus clear` to remove the local index
- `nexus index --watch` to reindex when files change
- Graceful handling for binary files, permission errors, non-UTF-8 files, and very large files
- Configurable defaults in `~/.nexus/config.toml`

## Installation

From source:

```bash
git clone https://github.com/Prasham-27/nexus-search.git
cd nexus-search
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Future PyPI install:

```bash
pip install nexus-search
```

Set your OpenAI key before semantic indexing or search:

```bash
export OPENAI_API_KEY="sk-..."
```

## Usage

Index a directory recursively:

```bash
nexus index ~/projects
```

Index only selected extensions:

```bash
nexus index ~/projects --ext py --ext md
```

Keep the index fresh while working:

```bash
nexus index ~/projects --watch
```

Run semantic search against the local FAISS index:

```bash
nexus search "authentication bug"
```

Return more semantic results:

```bash
nexus search "database migration rollback" --top 10
```

Filter semantic results by extension:

```bash
nexus search "retry policy" --ext py
```

Run exact keyword or regex search without embeddings:

```bash
nexus search "def authenticate" --exact --path ~/projects --ext py
```

Show index statistics:

```bash
nexus status
```

Delete the local index:

```bash
nexus clear
```

## Architecture

Indexing pipeline:

```text
Files → Walker → Chunker → Embedder (OpenAI) → FAISS Index → ~/.nexus/
```

```text
Files
  |
  v
Walker
  |
  v
Chunker
  |
  v
Embedder (OpenAI)
  |
  v
FAISS Index
  |
  v
~/.nexus/
```

Search pipeline:

```text
Query → Embedder → FAISS ANN Search → Ranked Chunks → Rich Table
```

```text
Query
  |
  v
Embedder
  |
  v
FAISS ANN Search
  |
  v
Ranked Chunks
  |
  v
Rich Table
```

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| CLI | Typer | Typed commands, polished help output, and straightforward packaging |
| Terminal UI | Rich | Tables, panels, progress bars, and syntax-highlighted snippets |
| Embeddings | OpenAI `text-embedding-3-small` | High-quality general-purpose semantic representations |
| Vector Index | FAISS | Fast local similarity search without running a vector database |
| Chunking | tiktoken | Token-aware chunk boundaries for embedding model inputs |
| File Watching | watchdog | Cross-platform file-system events for `--watch` |
| Environment | python-dotenv | Simple local API key workflow |
| Testing | pytest + pytest-mock | Deterministic tests with mocked embedding calls |

## Configuration

nexus creates `~/.nexus/config.toml` on first run with these defaults:

```toml
# Number of search results to show by default.
default_top_k = 5

# File extensions indexed when --ext is not provided.
default_extensions = ["py", "js", "ts", "md", "txt", "json", "yaml", "yml"]

# OpenAI embedding model used for semantic indexing and querying.
openai_model = "text-embedding-3-small"
```

## Running Tests

```bash
pytest tests/ -v
```

The test suite mocks OpenAI embeddings, writes FAISS indexes into temporary directories, and exercises the CLI through Typer's `CliRunner`.

## Design Decisions

- FAISS over a hosted vector database keeps the developer workflow private, local, fast, and dependency-light.
- Typer over argparse gives the CLI strong type hints, readable command declarations, and excellent generated help.
- Chunking preserves line ranges and uses token-aware limits so results stay readable while embedding inputs remain model-friendly.

Built by [Prasham Sheth](https://github.com/Prasham-27) · Part of an AI engineering portfolio
