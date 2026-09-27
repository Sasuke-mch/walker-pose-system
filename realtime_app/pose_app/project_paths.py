"""Path helpers shared by runtime and offline tools.

The repository is often launched from an IDE, a different working directory,
or a copied checkout.  Paths that belong to the project must therefore be
anchored to this file's repository location instead of the process CWD or a
machine-specific absolute path.
"""

from __future__ import annotations

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent


def project_path(*parts: str | Path) -> Path:
    """Return a path under the current checkout's repository root."""

    return PROJECT_ROOT.joinpath(*parts)


def repo_root() -> Path:
    """Return the absolute repository root without consulting the CWD."""

    return PROJECT_ROOT
