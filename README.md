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

Developers remember intent before they remember exact code. nexus bridges that gap by combining semantic embeddings for fuzzy, concept-level lookup with regex search for exact matches. It keeps the index local, stores vectors in FAISS, and presents results in a terminal-native interface that is fast to scan. The result is a practical "grep meets AI" workflow for codebases, notes, docs, and configuration files.

## Features

- Recursive indexing for common developer file types
- Semantic search powered by OpenAI `text-embedding-3-small`
- Local FAISS vector index stored under `~/.nexus/index/`
- Exact keyword and regex search with `--exact`
- Extension filtering with `--ext`
- Rich terminal tables, panels, progress indicators, and syntax-highlighted snippets
- Graceful skipping for binary and non-UTF-8 files

## Installation

From source:

```bash
git clone <your-repo-url> nexus-search
cd nexus-search
python -m venv .venv
source .venv/bin/activate
pip install -e .
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

## Architecture

Indexing pipeline:

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
| CLI | Typer | Clean typed commands, polished help output, easy packaging |
| Terminal UI | Rich | Tables, panels, progress, and syntax highlighting |
| Embeddings | OpenAI `text-embedding-3-small` | High-quality general-purpose semantic representations |
| Vector Index | FAISS | Fast local similarity search without running a database |
| Chunking | tiktoken | Token-aware chunk boundaries for embedding model inputs |
| Environment | python-dotenv | Simple local API key workflow |

## Configuration

The MVP reads environment variables via `.env`. A future `~/.nexus/config.toml` will expose defaults like this:

```toml
# Number of search results to show by default.
default_top_k = 5

# File extensions indexed when --ext is not provided.
default_extensions = ["py", "js", "ts", "md", "txt", "json", "yaml", "yml"]

# OpenAI embedding model used for semantic indexing and querying.
openai_model = "text-embedding-3-small"
```

## Running Tests

Tests are intentionally out of scope for this MVP. Once the test suite lands, run:

```bash
pytest tests/ -v
```

## Design Decisions

- FAISS over a hosted vector database keeps the developer workflow private, local, and dependency-light.
- Typer over argparse gives the CLI strong type hints, readable command declarations, and excellent generated help.
- Chunking preserves line ranges and uses token-aware limits so results stay understandable and embedding inputs remain model-friendly.
