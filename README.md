# smart-commit

Replace `git commit` when you have mixed-concern staged changes. `smart-commit`
asks Claude to group your staged files into semantically distinct commits,
shows the plan, and (with your approval) executes the commits in order.

```
$ git add .
$ smart-commit

Analyzing 7 staged files...

Claude suggests 3 commits:

  1. feat(api): add license validation endpoint
     - server/routes/license.py
     - server/models/license.py
     - tests/test_license.py

  2. fix(installer): handle spaces in Windows paths
     - installer/setup.py
     - installer/path_utils.py

  3. chore: update CI config
     - .github/workflows/ci.yml

[A]ccept all / [E]dit / [Q]uit? a

  ✓ (1/3) feat(api): add license validation endpoint
  ✓ (2/3) fix(installer): handle spaces in Windows paths
  ✓ (3/3) chore: update CI config

✓ Committed 3/3
```

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (the script declares its dependency on
  `anthropic` via inline script metadata)
- An `ANTHROPIC_API_KEY` from <https://console.anthropic.com/>

## Install

Pick one:

```bash
# Run directly via uv inline script metadata (no install)
uv run /path/to/smart-commit/smart_commit.py

# Install as a tool on $PATH (creates `smart-commit` binary)
uv tool install /path/to/smart-commit
smart-commit

# Wire up as a git alias against either form above
git config --global alias.sc '!smart-commit'        # after uv tool install
git config --global alias.sc '!uv run /path/to/smart-commit/smart_commit.py'
git sc
```

## Usage

```
smart-commit [-n | --dry-run] [-y | --auto] [-v | --verbose | --no-verbose]
```

| Flag | Description |
|---|---|
| `-n`, `--dry-run` | Print the plan and exit without committing. |
| `-y`, `--auto` | Skip the interactive prompt and execute all commits. Equivalent to `SMART_COMMIT_AUTO=1`. |
| `-v`, `--verbose` | Generate multi-line commit messages with a body paragraph. |
| `--no-verbose` | Force single-line messages, overriding `verbose_messages = true` in config. |

`--dry-run` and `--auto` are mutually exclusive; if both are passed, `--dry-run`
wins (no commits are made).

## Environment

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. |
| `SMART_COMMIT_MODEL` | `claude-sonnet-4-6` | Override the model. |
| `SMART_COMMIT_AUTO` | unset | Truthy values (`1`, `true`, `yes`) imply `--auto`. |
| `EDITOR` | `vi` | Used for `[E]dit` mode. |

## Configuration

Optional `.smart-commit.toml` at the repo root. See `.smart-commit.toml.example`
for the full schema. Highlights:

```toml
conventions = """
We use conventional commits.
Scopes: api, installer, ui, infra, docs.
"""
verbose_messages = false
trailers = ["Co-authored-by: Claude <noreply@anthropic.com>"]
```

`verbose_messages` is overridden by `--verbose` / `--no-verbose` for a single
run. `trailers` is repo-level only — there is no CLI override.

## Edit mode

`[E]dit` opens the plan in `$EDITOR` as TOML:

```toml
[[commit]]
message = "feat(api): add license validation endpoint"
body = ""
files = [
    "server/routes/license.py",
    "server/models/license.py",
]
reasoning = "These files together implement the license check feature"
```

Reorder, edit, drop, or split commits. On save, the plan is re-validated
(every staged file present in exactly one commit, no unknown files). If
parsing or validation fails the editor reopens once before aborting.

## Scope (v1)

- File-level grouping only. A file is fully in one commit or fully out — no
  hunk splitting. Files with mixed concerns may produce a less-than-ideal
  split; fix in `[E]dit` mode.
- `colocate` and `chore_glob` config keys are accepted but warned and ignored;
  they are planned for a follow-up.
- `[S]kip` and `1`-`9` step-through actions from the original spec are
  deferred. Use `[E]dit` to drop or reorder commits.

## Development

```bash
uv sync --group dev
uv run pytest
```

Tests use a temp git repo and monkeypatch `request_grouping` so no real
Anthropic API calls are made.
