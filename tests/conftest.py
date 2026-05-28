"""Shared pytest fixtures for nexus-search."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def sample_project(tmp_path: Path) -> Path:
    """Create a deterministic sample project tree.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Root path containing sample files.
    """

    root = tmp_path / "project"
    root.mkdir()
    (root / "auth.py").write_text(
        "\n".join(
            [
                "def authenticate_user(token):",
                "    if not token:",
                "        raise ValueError('missing token')",
                "    return token == 'valid'",
            ]
        ),
        encoding="utf-8",
    )
    (root / "billing.py").write_text(
        "\n".join(
            [
                "def charge_customer(customer_id):",
                "    return {'status': 'paid', 'customer_id': customer_id}",
            ]
        ),
        encoding="utf-8",
    )
    (root / "utils.py").write_text(
        "\n".join(
            [
                "def normalize_email(value):",
                "    return value.strip().lower()",
            ]
        ),
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Demo\n\nThis project documents authentication failures and billing flows.",
        encoding="utf-8",
    )
    (root / "incidents.md").write_text(
        "# Incidents\n\nA token refresh bug caused an authentication outage.",
        encoding="utf-8",
    )
    (root / "binary.py").write_bytes(b"\x00\x01not text")
    (root / "ignored.png").write_bytes(b"\x89PNG\r\n")
    return root
