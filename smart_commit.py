#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic>=0.40"]
# ///
"""smart-commit: split staged git changes into atomic commits using Claude."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
CONFIG_FILENAME = ".smart-commit.toml"
MAX_DIFF_BYTES = 100_000
DIFF_HEAD_TAIL_LINES = 50
MAX_TOKENS = 4096

EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_API_ERROR = 2
EXIT_GIT_ERROR = 3
EXIT_VALIDATION_ERROR = 4


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    auto: bool = False
    dry_run: bool = False
    verbose_messages: bool = False
    trailers: list[str] = field(default_factory=list)
    conventions: str = ""


@dataclass
class CommitGroup:
    message: str
    files: list[str]
    body: str = ""
    reasoning: str = ""


# ======================================================================
# Git helpers
# ======================================================================


class GitError(RuntimeError):
    pass


def run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *args],
            check=check,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip() or f"exit {e.returncode}"
        raise GitError(f"git {' '.join(args)}: {msg}") from e
    except FileNotFoundError as e:
        raise GitError("git is not installed or not on PATH") from e


def git_repo_root() -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"]).stdout.strip())


def git_staged_status() -> tuple[list[str], dict[str, str]]:
    """Return (paths, rename_old_to_new) for staged files.

    Renames map old-path -> new-path; the new path is what callers see in
    the file list. When staging a group that contains a renamed new-path,
    the old path must also be added so git can re-detect the rename.
    """
    result = run_git(["diff", "--cached", "--name-status", "-z"])
    paths: list[str] = []
    rename_pairs: dict[str, str] = {}
    tokens = result.stdout.split("\0")
    i = 0
    while i < len(tokens):
        status = tokens[i]
        i += 1
        if not status:
            continue
        if status[0] in ("R", "C") and i + 1 < len(tokens):
            old_path = tokens[i]
            new_path = tokens[i + 1]
            i += 2
            paths.append(new_path)
            rename_pairs[old_path] = new_path
        elif i < len(tokens):
            paths.append(tokens[i])
            i += 1
    return paths, rename_pairs


def git_staged_diff() -> str:
    return run_git(["diff", "--cached"]).stdout


def git_recent_log(n: int = 20) -> str:
    result = run_git(["log", f"-{n}", "--oneline"], check=False)
    return result.stdout if result.returncode == 0 else ""


def git_in_progress_state() -> str | None:
    """Return 'merge' / 'rebase' / 'cherry-pick' if one is in progress."""
    git_dir = Path(run_git(["rev-parse", "--git-dir"]).stdout.strip())
    if (git_dir / "MERGE_HEAD").exists():
        return "merge"
    if (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists():
        return "rebase"
    if (git_dir / "CHERRY_PICK_HEAD").exists():
        return "cherry-pick"
    return None


def git_reset_index() -> None:
    run_git(["reset", "HEAD", "--"], check=False)


def git_add_all(paths: list[str]) -> None:
    if not paths:
        return
    run_git(["add", "-A", "--", *paths])


def git_commit_with_message(message: str) -> None:
    run_git(["commit", "-m", message])


def git_commit_with_file(path: str) -> None:
    run_git(["commit", "-F", path])


# ======================================================================
# Diff truncation
# ======================================================================


def truncate_diff(diff: str, max_bytes: int = MAX_DIFF_BYTES, head_tail: int = DIFF_HEAD_TAIL_LINES) -> str:
    """Truncate per-file diff sections when the full diff is too large."""
    if len(diff.encode("utf-8")) <= max_bytes:
        return diff

    sections = re.split(r"(?m)^(?=diff --git )", diff)
    parts: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        lines = section.splitlines()
        if len(lines) <= 2 * head_tail + 5:
            parts.append(section.rstrip("\n"))
            continue
        head = lines[:head_tail]
        tail = lines[-head_tail:]
        omitted = len(lines) - 2 * head_tail
        parts.append(
            "\n".join(head)
            + f"\n[... {omitted} lines truncated ...]\n"
            + "\n".join(tail)
        )
    return "\n".join(parts) + "\n"


# ======================================================================
# Config
# ======================================================================


def load_config(repo_root: Path) -> dict:
    path = repo_root / CONFIG_FILENAME
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"warning: failed to parse {CONFIG_FILENAME}: {e}", file=sys.stderr)
        return {}


def build_config(args: argparse.Namespace, repo_root: Path) -> Config:
    raw = load_config(repo_root)
    cfg = Config()
    cfg.model = os.environ.get("SMART_COMMIT_MODEL", DEFAULT_MODEL)
    cfg.auto = bool(args.auto) or os.environ.get("SMART_COMMIT_AUTO", "").lower() in ("1", "true", "yes")
    cfg.dry_run = bool(args.dry_run)
    if cfg.dry_run:
        cfg.auto = False  # dry-run wins

    if args.verbose is not None:
        cfg.verbose_messages = args.verbose
    else:
        cfg.verbose_messages = bool(raw.get("verbose_messages", False))

    trailers = raw.get("trailers", [])
    if isinstance(trailers, list):
        cfg.trailers = [str(t) for t in trailers if str(t).strip()]
    cfg.conventions = str(raw.get("conventions", ""))

    if raw.get("colocate") or raw.get("chore_glob"):
        print(
            "warning: 'colocate' and 'chore_glob' config keys are not yet supported.",
            file=sys.stderr,
        )
    return cfg


# ======================================================================
# Claude API
# ======================================================================


SYSTEM_PROMPT_BASE = """You are a git commit splitter. Given a unified diff of staged changes, group the changed files into semantically distinct commits.

Rules:
- Each group represents ONE logical change (feature, bugfix, refactor, chore, test, docs).
- Order commits so dependencies come first (e.g. a new utility before the feature that uses it).
- Use conventional commit format for the subject: type(scope): short imperative description.
- Match the style of the recent commit log when one is provided.
- If all changes are genuinely one concern, return a single commit.
- Never include files that are not in the staged file list.
- Each staged file MUST appear in EXACTLY ONE group.
- Always include a "reasoning" field explaining why these files belong together (one short sentence).
"""

SYSTEM_PROMPT_VERBOSE = '- Set "body" to 1-3 sentences explaining what the change does and why. Add context beyond the subject line.\n'
SYSTEM_PROMPT_TERSE = '- Leave "body" as an empty string.\n'


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "commits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "body": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["message", "files", "body", "reasoning"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["commits"],
    "additionalProperties": False,
}


def build_system_prompt(verbose: bool) -> str:
    return SYSTEM_PROMPT_BASE + (SYSTEM_PROMPT_VERBOSE if verbose else SYSTEM_PROMPT_TERSE)


def build_user_message(diff: str, files: list[str], recent_log: str, conventions: str) -> str:
    parts: list[str] = []
    if recent_log.strip():
        parts.append(f"## Recent commit style\n{recent_log.strip()}")
    if conventions.strip():
        parts.append(f"## Project conventions\n{conventions.strip()}")
    parts.append("## Staged files\n" + "\n".join(files))
    parts.append(f"## Full diff\n{diff}")
    return "\n\n".join(parts)


def request_grouping(
    client: anthropic.Anthropic,
    config: Config,
    diff: str,
    files: list[str],
    recent_log: str,
) -> list[CommitGroup]:
    response = client.messages.create(
        model=config.model,
        max_tokens=MAX_TOKENS,
        system=build_system_prompt(config.verbose_messages),
        messages=[{"role": "user", "content": build_user_message(diff, files, recent_log, config.conventions)}],
        output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    data = json.loads(text)
    groups: list[CommitGroup] = []
    for item in data["commits"]:
        groups.append(
            CommitGroup(
                message=str(item["message"]).strip(),
                files=[str(f) for f in item["files"]],
                body=str(item.get("body", "")).strip(),
                reasoning=str(item.get("reasoning", "")).strip(),
            )
        )
    return groups


# ======================================================================
# Validation
# ======================================================================


def validate_groups(groups: list[CommitGroup], staged: list[str]) -> list[str]:
    """Return a list of human-readable validation errors. Empty list = OK."""
    errors: list[str] = []
    staged_set = set(staged)
    seen: dict[str, int] = {}
    for i, g in enumerate(groups, 1):
        if not g.message.strip():
            errors.append(f"Commit {i}: empty message")
        if not g.files:
            errors.append(f"Commit {i}: no files")
        for f in g.files:
            if f not in staged_set:
                errors.append(f"Commit {i}: '{f}' is not in the staged file list")
            if f in seen:
                errors.append(f"File '{f}' appears in commits {seen[f]} and {i}")
            else:
                seen[f] = i
    unassigned = sorted(staged_set - set(seen.keys()))
    if unassigned:
        errors.append("Files not in any commit: " + ", ".join(unassigned))
    return errors


# ======================================================================
# Renderer + interactive prompt
# ======================================================================


def render_plan(groups: list[CommitGroup], verbose: bool) -> None:
    n = len(groups)
    suffix = "" if n == 1 else "s"
    print(f"Claude suggests {n} commit{suffix}:\n")
    for i, g in enumerate(groups, 1):
        print(f"  {i}. {g.message}")
        if verbose and g.body:
            wrapped = textwrap.fill(
                g.body,
                width=72,
                initial_indent="     ",
                subsequent_indent="     ",
            )
            print()
            print(wrapped)
            print()
        for f in g.files:
            print(f"     - {f}")
        print()


def prompt_action() -> str:
    """Return one of 'a', 'e', 'q'."""
    while True:
        try:
            response = input("[A]ccept all / [E]dit / [Q]uit? ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "q"
        if not response:
            continue
        ch = response[0]
        if ch in ("a", "e", "q"):
            return ch


# ======================================================================
# TOML emit/parse for edit mode
# ======================================================================


def _toml_escape_basic(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )


def toml_str(s: str) -> str:
    """Emit a TOML string. Use multi-line literal when the value has newlines."""
    if "\n" in s:
        # Triple-quoted basic string: escape backslashes and triple-quotes.
        body = s.replace("\\", "\\\\").replace('"""', '\\"""')
        return f'"""\n{body}\n"""'
    return f'"{_toml_escape_basic(s)}"'


def groups_to_toml(groups: list[CommitGroup]) -> str:
    lines = [
        "# Edit the commit plan below and save to apply.",
        "# Each [[commit]] block becomes one git commit, in order.",
        "# Reorder, edit, or delete blocks as needed.",
        "",
    ]
    for g in groups:
        lines.append("[[commit]]")
        lines.append(f"message = {toml_str(g.message)}")
        lines.append(f"body = {toml_str(g.body)}")
        lines.append("files = [")
        for f in g.files:
            lines.append(f"    {toml_str(f)},")
        lines.append("]")
        lines.append(f"reasoning = {toml_str(g.reasoning)}")
        lines.append("")
    return "\n".join(lines)


def parse_toml_plan(text: str) -> list[CommitGroup]:
    data = tomllib.loads(text)
    commits = data.get("commit", [])
    if not isinstance(commits, list):
        raise ValueError("expected an array of [[commit]] tables")
    groups: list[CommitGroup] = []
    for idx, item in enumerate(commits, 1):
        if not isinstance(item, dict):
            raise ValueError(f"commit {idx}: not a table")
        if "message" not in item or "files" not in item:
            raise ValueError(f"commit {idx}: missing 'message' or 'files'")
        groups.append(
            CommitGroup(
                message=str(item["message"]).strip(),
                files=[str(f) for f in item["files"]],
                body=str(item.get("body", "")).strip(),
                reasoning=str(item.get("reasoning", "")).strip(),
            )
        )
    return groups


def edit_plan(groups: list[CommitGroup], staged: list[str]) -> list[CommitGroup] | None:
    """Open the plan in $EDITOR, validate, return updated groups or None on abort."""
    editor_cmd = os.environ.get("EDITOR", "vi")
    text = groups_to_toml(groups)
    for attempt in range(2):
        with tempfile.NamedTemporaryFile("w", suffix=".smart-commit.toml", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            try:
                subprocess.run([*editor_cmd.split(), path], check=False)
            except FileNotFoundError:
                print(f"error: editor '{editor_cmd}' not found", file=sys.stderr)
                return None
            with open(path) as f:
                edited = f.read()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        if not edited.strip():
            print("error: edited plan is empty", file=sys.stderr)
            return None

        try:
            new_groups = parse_toml_plan(edited)
        except (tomllib.TOMLDecodeError, ValueError, KeyError) as e:
            print(f"  parse error: {e}", file=sys.stderr)
            if attempt == 0:
                print("  re-opening editor (one retry)...", file=sys.stderr)
                text = edited
                continue
            return None

        errors = validate_groups(new_groups, staged)
        if errors:
            print("  validation errors:", file=sys.stderr)
            for err in errors:
                print(f"    - {err}", file=sys.stderr)
            if attempt == 0:
                text = edited
                continue
            return None
        return new_groups
    return None


# ======================================================================
# Commit message assembly + executor
# ======================================================================


def build_commit_message(group: CommitGroup, config: Config) -> str:
    sections: list[str] = [group.message.strip()]
    if config.verbose_messages and group.body.strip():
        sections.append(textwrap.fill(group.body.strip(), width=72))
    if config.trailers:
        sections.append("\n".join(config.trailers))
    return "\n\n".join(sections)


def execute_plan(
    groups: list[CommitGroup],
    all_staged: list[str],
    rename_pairs: dict[str, str],
    config: Config,
) -> tuple[int, int]:
    """Run the commit plan. Return (commits_made, total_groups)."""
    total = len(groups)
    made = 0
    committed_paths: set[str] = set()

    for i, group in enumerate(groups, 1):
        try:
            git_reset_index()
            paths_to_stage = list(group.files)
            for old, new in rename_pairs.items():
                if new in group.files:
                    paths_to_stage.append(old)
            git_add_all(paths_to_stage)

            message = build_commit_message(group, config)
            if "\n" in message:
                tmp = tempfile.NamedTemporaryFile(
                    "w", suffix=".smart-commit-msg", delete=False
                )
                try:
                    tmp.write(message)
                    tmp.flush()
                    tmp.close()
                    git_commit_with_file(tmp.name)
                finally:
                    try:
                        os.unlink(tmp.name)
                    except OSError:
                        pass
            else:
                git_commit_with_message(message)
            made += 1
            committed_paths.update(group.files)
            for old, new in rename_pairs.items():
                if new in group.files:
                    committed_paths.add(old)
            print(f"  ✓ ({i}/{total}) {group.message}")
        except GitError as e:
            print(f"  ✗ ({i}/{total}) {group.message}: {e}", file=sys.stderr)
            # Re-stage everything that wasn't committed yet.
            remaining: list[str] = [p for p in all_staged if p not in committed_paths]
            for old, new in rename_pairs.items():
                if new in remaining and old not in remaining:
                    remaining.append(old)
            try:
                git_reset_index()
                git_add_all(remaining)
            except GitError as e2:
                print(f"  also failed to re-stage remaining files: {e2}", file=sys.stderr)
            return made, total

    return made, total


# ======================================================================
# Main
# ======================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="smart-commit",
        description="Split staged changes into atomic commits using Claude.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without committing.",
    )
    parser.add_argument(
        "-y",
        "--auto",
        action="store_true",
        help="Skip the prompt and execute all commits (also: SMART_COMMIT_AUTO=1).",
    )
    verbose = parser.add_mutually_exclusive_group()
    verbose.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=None,
        help="Generate multi-line commit messages with a body paragraph.",
    )
    verbose.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Force single-line commit messages (override config).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        repo_root = git_repo_root()
    except GitError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USER_ERROR

    in_progress = git_in_progress_state()
    if in_progress:
        print(f"error: {in_progress} in progress; resolve it first.", file=sys.stderr)
        return EXIT_USER_ERROR

    config = build_config(args, repo_root)

    try:
        staged, rename_pairs = git_staged_status()
    except GitError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_GIT_ERROR

    if not staged:
        print("Nothing staged. Stage changes first (e.g. `git add ...`).", file=sys.stderr)
        return EXIT_USER_ERROR

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "error: ANTHROPIC_API_KEY is not set. "
            "Get one at https://console.anthropic.com/ and `export ANTHROPIC_API_KEY=...`",
            file=sys.stderr,
        )
        return EXIT_USER_ERROR

    n = len(staged)
    suffix = "" if n == 1 else "s"
    print(f"Analyzing {n} staged file{suffix}...\n")

    diff = truncate_diff(git_staged_diff())
    recent_log = git_recent_log()

    client = anthropic.Anthropic(api_key=api_key)
    try:
        groups = request_grouping(client, config, diff, staged, recent_log)
    except anthropic.APIError as e:
        print(f"error: API call failed: {e}", file=sys.stderr)
        return EXIT_API_ERROR
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"error: unexpected response shape from API: {e}", file=sys.stderr)
        return EXIT_API_ERROR

    errors = validate_groups(groups, staged)
    if errors:
        print("Validation errors in Claude's plan:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        if config.auto or config.dry_run:
            return EXIT_VALIDATION_ERROR
        print("\nOpen in editor to fix?")
        try:
            choice = input("[E]dit / [Q]uit? ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return EXIT_VALIDATION_ERROR
        if not choice.startswith("e"):
            return EXIT_VALIDATION_ERROR
        fixed = edit_plan(groups, staged)
        if fixed is None:
            return EXIT_VALIDATION_ERROR
        groups = fixed

    render_plan(groups, config.verbose_messages)

    if config.dry_run:
        print("(dry run — no commits were made)")
        return EXIT_OK

    if not config.auto:
        while True:
            action = prompt_action()
            if action == "a":
                break
            if action == "q":
                print("Aborted. Files remain staged.")
                return EXIT_OK
            if action == "e":
                edited = edit_plan(groups, staged)
                if edited is not None:
                    groups = edited
                    print()
                    render_plan(groups, config.verbose_messages)

    print()
    made, total = execute_plan(groups, staged, rename_pairs, config)
    print()
    if made == total:
        print(f"✓ Committed {made}/{total}")
    else:
        print(
            f"⚠ Committed {made}/{total}. Remaining files re-staged.",
            file=sys.stderr,
        )

    if made > 0:
        log = run_git(["log", f"-{made}", "--oneline"], check=False).stdout
        if log.strip():
            print()
            for line in log.splitlines():
                print(f"  {line}")

    return EXIT_OK if made == total else EXIT_GIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
