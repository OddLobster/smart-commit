"""Shared test fixtures and module loader for smart-commit."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def smart_commit_module():
    import smart_commit  # noqa: PLC0415

    return smart_commit


@pytest.fixture
def sc(smart_commit_module):
    """Short alias for the smart-commit module."""
    return smart_commit_module


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def tmp_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an empty git repo with a single initial commit; cd into it."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "README.md").write_text("# test\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-q", "-m", "initial commit")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    monkeypatch.delenv("SMART_COMMIT_AUTO", raising=False)
    monkeypatch.delenv("SMART_COMMIT_MODEL", raising=False)
    return tmp_path


def stage_files(repo: Path, files: dict[str, str]) -> None:
    """Write and stage a dict of {relative_path: content}."""
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(repo, "add", *files.keys())


def commit_log_subjects(repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", "log", "--format=%s"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def staged_files(repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def commit_body(repo: Path, ref: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "log", "-1", "--format=%B", ref],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
