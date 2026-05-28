"""Configuration loading for nexus-search."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

NEXUS_HOME = Path.home() / ".nexus"
DEFAULT_INDEX_DIR = NEXUS_HOME / "index"
DEFAULT_CONFIG_PATH = NEXUS_HOME / "config.toml"
DEFAULT_TOP_K = 5
DEFAULT_EXTENSIONS = ("py", "js", "ts", "md", "txt", "json", "yaml", "yml")
DEFAULT_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class NexusConfig:
    """Runtime configuration for the nexus CLI."""

    default_top_k: int = DEFAULT_TOP_K
    default_extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    openai_model: str = DEFAULT_MODEL


def normalize_extensions(extensions: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize extension filters to bare lowercase names.

    Args:
        extensions: Optional extensions with or without leading dots.

    Returns:
        Normalized extension names.
    """

    if not extensions:
        return DEFAULT_EXTENSIONS
    return tuple(ext.lower().lstrip(".") for ext in extensions if ext and ext.strip())


def default_config_text() -> str:
    """Return the default TOML configuration with comments.

    Returns:
        TOML text suitable for writing to disk.
    """

    extensions = ", ".join(f'"{extension}"' for extension in DEFAULT_EXTENSIONS)
    return (
        "# Number of search results to show by default.\n"
        f"default_top_k = {DEFAULT_TOP_K}\n\n"
        "# File extensions indexed when --ext is not provided.\n"
        f"default_extensions = [{extensions}]\n\n"
        "# OpenAI embedding model used for semantic indexing and querying.\n"
        f'openai_model = "{DEFAULT_MODEL}"\n'
    )


def ensure_config_file(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """Create the default config file when it is missing.

    Args:
        config_path: Config file path.

    Returns:
        The config file path.
    """

    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(default_config_text(), encoding="utf-8")
    return config_path


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> NexusConfig:
    """Load configuration from TOML, creating defaults if needed.

    Args:
        config_path: Config file path.

    Returns:
        Parsed runtime configuration.
    """

    try:
        ensure_config_file(config_path)
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return NexusConfig()
    return NexusConfig(
        default_top_k=int(data.get("default_top_k", DEFAULT_TOP_K)),
        default_extensions=normalize_extensions(data.get("default_extensions", DEFAULT_EXTENSIONS)),
        openai_model=str(data.get("openai_model", DEFAULT_MODEL)),
    )
