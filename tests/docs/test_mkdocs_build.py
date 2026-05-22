"""Smoke test: `mkdocs build --strict` succeeds.

Skipped automatically on environments without `mkdocs` installed (e.g., the
minimal CI image used by ``.github/workflows/ci.yml``). The dedicated
``docs.yml`` workflow installs ``.[docs]`` so this test runs there.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mkdocs")


def test_mkdocs_build_strict_succeeds(tmp_path: Path) -> None:
    """`mkdocs build --strict` exits 0 against the committed config."""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--site-dir",
            str(tmp_path / "site"),
            "--config-file",
            str(repo_root / "mkdocs.yml"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"mkdocs build --strict failed:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert (tmp_path / "site" / "index.html").exists()
