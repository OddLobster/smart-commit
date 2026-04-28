#!/usr/bin/env -S uv run --script
# PYTHON_ARGCOMPLETE_OK
# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic>=0.40", "httpx>=0.27", "argcomplete>=3.0"]
# ///
"""smart-commit: split staged git changes into atomic commits using an LLM."""

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
import argcomplete
import httpx

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENROUTER = "openrouter"
VALID_PROVIDERS = (PROVIDER_ANTHROPIC, PROVIDER_OPENROUTER)

DEFAULT_MODEL_BY_PROVIDER = {
    PROVIDER_ANTHROPIC: "claude-sonnet-4-6",
    PROVIDER_OPENROUTER: "anthropic/claude-sonnet-4.5",
}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Short aliases for the --model flag. Resolve provider-aware so `haiku` Just
# Works whether the user is on Anthropic direct or via OpenRouter. Anything
# not in this map (e.g. `openai/gpt-5`, `qwen/qwen3-vl-plus`) passes through.
MODEL_ALIASES: dict[str, dict[str, str]] = {
    PROVIDER_ANTHROPIC: {
        "haiku": "claude-haiku-4-5",
        "sonnet": "claude-sonnet-4-6",
        "opus": "claude-opus-4-7",
    },
    PROVIDER_OPENROUTER: {
        "haiku": "anthropic/claude-haiku-4.5",
        "sonnet": "anthropic/claude-sonnet-4.5",
        "opus": "anthropic/claude-opus-4.7",
    },
}

# Curated suggestions for shell tab-completion on `--model`. Not authoritative
# — anything OpenRouter exposes is still a valid value, this is just the list
# we surface on tab. Aliases come first so `<tab><tab>` short-cycles them.
MODEL_COMPLETION_HINTS: dict[str, list[str]] = {
    PROVIDER_ANTHROPIC: [
        "haiku",
        "sonnet",
        "opus",
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-opus-4-6",
    ],
    PROVIDER_OPENROUTER: [
        "haiku",
        "sonnet",
        "opus",
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-opus-4.7",
        "openai/gpt-5",
        "openai/gpt-4.1",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "meta-llama/llama-3.3-70b-instruct",
        "qwen/qwen3-coder",
        "qwen/qwen3-vl-plus",
        "deepseek/deepseek-r1",
        "x-ai/grok-4",
        "mistralai/mistral-large",
    ],
}

# Prices in USD per million tokens (input, output). Used only for the
# Anthropic provider — OpenRouter returns actual cost on the response.
ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

CONFIG_FILENAME = ".smart-commit.toml"

INIT_TEMPLATE = """\
# smart-commit configuration. All keys are optional — defaults work fine.
# Full reference: .smart-commit.toml.example in the smart-commit repo.

# --- Provider & model ---------------------------------------------------

# provider = "anthropic"     # or "openrouter"
# model    = "sonnet"        # alias or full ID. SMART_COMMIT_MODEL env wins.

# --- Per-branch context (optional) --------------------------------------

# Persistent baseline that's prepended to per-run -m / --context strings.
# Uncomment when this branch has a stable, durable intent.
# context = "v2 migration: new API routes + DB schema changes"

# --- Prompt customization -----------------------------------------------

# Free-form text injected into the prompt under "## Project conventions".
# Teach the model your repo's commit-message house style.
# conventions = \"\"\"
# We use conventional commits.
# Scopes: api, ui, infra, docs.
# Always lowercase. No period at end of subject line.
# \"\"\"

# --- Commit message style -----------------------------------------------

# When true, generates subject + body per commit. Override per-run with
# --verbose / --no-verbose.
# verbose_messages = false

# Trailers appended to every commit message. Common uses:
#   - Co-authored-by for AI attribution
#   - Signed-off-by for DCO compliance
# trailers = [
#     "Co-authored-by: Claude <noreply@anthropic.com>",
# ]
"""
MAX_DIFF_BYTES = 100_000
DIFF_HEAD_TAIL_LINES = 50
MAX_TOKENS = 4096
HTTP_TIMEOUT_SECONDS = 120.0

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
    provider: str = PROVIDER_ANTHROPIC
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    auto: bool = False
    dry_run: bool = False
    verbose_messages: bool = False
    trailers: list[str] = field(default_factory=list)
    conventions: str = ""
    context: str = ""


class APICallError(RuntimeError):
    """Provider-agnostic wrapper for API errors."""


@dataclass
class CommitGroup:
    message: str
    files: list[str]
    body: str = ""
    reasoning: str = ""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # None when pricing is unknown
    estimated: bool = False  # True when cost is derived from a local price table, not the API


# ======================================================================
# Model + pricing helpers
# ======================================================================


def resolve_model(name: str, provider: str) -> str:
    """Expand a short alias (`haiku`, `sonnet`, `opus`) to a provider-specific ID."""
    if not name:
        return name
    aliases = MODEL_ALIASES.get(provider, {})
    return aliases.get(name.strip().lower(), name)


def _resolve_provider_for_completion(parsed_args) -> str:
    """Best-effort provider resolution during shell completion (no config-file read)."""
    p = (
        getattr(parsed_args, "provider", None)
        or os.environ.get("SMART_COMMIT_PROVIDER", "").strip().lower()
    )
    if p in VALID_PROVIDERS:
        return p
    if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        return PROVIDER_OPENROUTER
    return PROVIDER_ANTHROPIC


def _model_completer(prefix, parsed_args, **kwargs):  # noqa: ARG001
    """argcomplete hook for `--model`: suggest provider-appropriate model IDs."""
    return MODEL_COMPLETION_HINTS[_resolve_provider_for_completion(parsed_args)]


def anthropic_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate Anthropic API cost in USD, or None if the model isn't priced here."""
    price = ANTHROPIC_PRICING.get(model)
    if price is None:
        return None
    in_per_m, out_per_m = price
    return (input_tokens / 1_000_000) * in_per_m + (output_tokens / 1_000_000) * out_per_m


def fmt_tokens(n: int) -> str:
    if n >= 10_000:
        return f"~{n / 1000:.0f}k"
    if n >= 1000:
        return f"~{n / 1000:.1f}k"
    return str(n)


def fmt_cost(usd: float | None) -> str | None:
    if usd is None:
        return None
    if usd <= 0:
        return "$0"
    if usd < 0.001:
        return "<$0.001"
    if usd < 1:
        return f"${usd:.3f}"
    if usd < 100:
        return f"${usd:.2f}"
    return f"${usd:.0f}"


def model_display_name(model: str) -> str:
    """Strip the provider namespace from an OpenRouter-style ID for display."""
    if not model:
        return "model"
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def format_usage_line(usage: Usage) -> str:
    parts = [
        f"{fmt_tokens(usage.input_tokens)} in",
        f"{fmt_tokens(usage.output_tokens)} out",
    ]
    cost = fmt_cost(usage.cost_usd)
    if cost is not None:
        # Tilde marks an estimate — cost was derived from a local price table
        # rather than reported by the provider (e.g. BYOK on Anthropic).
        parts.append(f"~{cost}" if usage.estimated else cost)
    return "(" + " · ".join(parts) + ")"


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


def _resolve_provider(args: argparse.Namespace, raw: dict) -> str:
    """Resolve provider from CLI > env > config > auto-detect."""
    if getattr(args, "provider", None):
        return args.provider
    env = os.environ.get("SMART_COMMIT_PROVIDER", "").strip().lower()
    if env:
        return env
    if raw.get("provider"):
        return str(raw["provider"]).strip().lower()
    # Auto-detect: if exactly one key is set, use that provider.
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    if has_openrouter and not has_anthropic:
        return PROVIDER_OPENROUTER
    return PROVIDER_ANTHROPIC


def build_config(args: argparse.Namespace, repo_root: Path) -> Config:
    raw = load_config(repo_root)
    cfg = Config()

    cfg.provider = _resolve_provider(args, raw)
    if cfg.provider not in VALID_PROVIDERS:
        raise ValueError(
            f"unknown provider '{cfg.provider}'. Valid: {', '.join(VALID_PROVIDERS)}"
        )

    raw_model = (
        getattr(args, "model", None)
        or os.environ.get("SMART_COMMIT_MODEL")
        or str(raw.get("model", "")).strip()
        or DEFAULT_MODEL_BY_PROVIDER[cfg.provider]
    )
    cfg.model = resolve_model(raw_model, cfg.provider)

    if cfg.provider == PROVIDER_ANTHROPIC:
        cfg.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        cfg.base_url = ""
    else:  # openrouter
        cfg.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        cfg.base_url = (
            os.environ.get("SMART_COMMIT_BASE_URL")
            or str(raw.get("base_url", "")).strip()
            or OPENROUTER_BASE_URL
        )

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

    # Context resolution. The config file is a persistent baseline (e.g. a
    # long-running feature branch's intent); CLI / env are per-run overlays.
    # CLI wins over env. Final string = config_baseline + " " + per_run_overlay.
    config_context = str(raw.get("context", "")).strip()
    cli_chunks = [c.strip() for c in (getattr(args, "context", None) or []) if c and c.strip()]
    cli_context = " ".join(cli_chunks)
    env_context = os.environ.get("SMART_COMMIT_CONTEXT", "").strip()
    per_run_context = cli_context or env_context
    cfg.context = " ".join(p for p in (config_context, per_run_context) if p).strip()

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

SYSTEM_PROMPT_SCHEMA = """
Respond with a single JSON object matching this schema, and nothing else (no markdown fences, no commentary):

{
  "commits": [
    {
      "message": "feat(api): add license validation endpoint",
      "files": ["server/routes/license.py", "server/models/license.py"],
      "body": "",
      "reasoning": "These files together implement the license check feature."
    }
  ]
}
"""


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
    return (
        SYSTEM_PROMPT_BASE
        + (SYSTEM_PROMPT_VERBOSE if verbose else SYSTEM_PROMPT_TERSE)
        + SYSTEM_PROMPT_SCHEMA
    )


def build_user_message(
    diff: str,
    files: list[str],
    recent_log: str,
    conventions: str,
    context: str = "",
) -> str:
    parts: list[str] = []
    if context.strip():
        parts.append(
            "## Developer context\n"
            "The developer describes this session:\n"
            f'"{context.strip()}"\n\n'
            "Use this to inform your grouping decisions. Files related to the "
            "same described concern should be grouped together; if the "
            "developer mentions multiple concerns, use them as grouping "
            "anchors. Still validate against the actual diff — don't invent "
            "groups that aren't supported by the changes, and don't drop "
            "files just because the developer didn't mention them."
        )
    if recent_log.strip():
        parts.append(f"## Recent commit style\n{recent_log.strip()}")
    if conventions.strip():
        parts.append(f"## Project conventions\n{conventions.strip()}")
    parts.append("## Staged files\n" + "\n".join(files))
    parts.append(f"## Full diff\n{diff}")
    return "\n\n".join(parts)


def _strip_json_fences(text: str) -> str:
    """Remove markdown ``` / ```json fences if a model adds them despite instructions."""
    text = text.strip()
    text = re.sub(r"^```(?:json|JSON)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _items_to_groups(items: list[dict]) -> list[CommitGroup]:
    groups: list[CommitGroup] = []
    for item in items:
        groups.append(
            CommitGroup(
                message=str(item["message"]).strip(),
                files=[str(f) for f in item["files"]],
                body=str(item.get("body", "")).strip(),
                reasoning=str(item.get("reasoning", "")).strip(),
            )
        )
    return groups


def _request_anthropic(config: Config, system: str, user: str) -> tuple[list[dict], Usage]:
    client = anthropic.Anthropic(api_key=config.api_key)
    try:
        response = client.messages.create(
            model=config.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        )
    except anthropic.APIError as e:
        raise APICallError(str(e)) from e
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    data = json.loads(text)
    in_tok = getattr(response.usage, "input_tokens", 0) or 0
    out_tok = getattr(response.usage, "output_tokens", 0) or 0
    usage = Usage(
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=anthropic_cost(config.model, in_tok, out_tok),
    )
    return list(data["commits"]), usage


def _request_openrouter(config: Config, system: str, user: str) -> tuple[list[dict], Usage]:
    """Call OpenRouter (or any OpenAI-compatible endpoint) with one parse retry."""
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/oddlobster/smart-commit",
        "X-Title": "smart-commit",
    }
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    payload = {
        "model": config.model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "commits",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
    }
    # Note: OpenRouter's `usage: {include: true}` and OpenAI's `stream_options:
    # {include_usage: true}` are now deprecated/no-ops — usage is always returned.

    last_error: str | None = None
    last_text: str | None = None
    last_data: dict | None = None
    for attempt in range(2):
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
                response = client.post(
                    f"{config.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response is not None else ""
            raise APICallError(
                f"{e.response.status_code} from {config.base_url}: {body}"
            ) from e
        except httpx.HTTPError as e:
            raise APICallError(f"HTTP error talking to {config.base_url}: {e}") from e

        last_data = data
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise APICallError(f"Unexpected response shape from {config.base_url}: {e}") from e
        last_text = text
        cleaned = _strip_json_fences(text)
        try:
            parsed = json.loads(cleaned)
            usage = _openrouter_usage(data, model=config.model)
            return list(parsed["commits"]), usage
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            last_error = str(e)
            if attempt == 0:
                # Append the bad assistant turn and ask for a clean retry.
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON matching the schema. "
                        "Respond with ONLY the JSON object, no markdown fences, no commentary."
                    ),
                })
                payload = {**payload, "messages": messages}
                continue
    snippet = (last_text or "")[:200].replace("\n", " ")
    raise APICallError(f"Could not parse JSON from model after retry ({last_error}): {snippet!r}")


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _openrouter_usage(data: dict, model: str = "") -> Usage:
    """Extract usage from an OpenRouter chat-completion response.

    OpenRouter reports cost in two places (per their usage-accounting docs):
      - `usage.cost`: what OpenRouter charges the user (zero for BYOK and
        some free-tier promos).
      - `usage.cost_details.upstream_inference_cost`: present only on BYOK
        requests — the cost the upstream provider charges.

    Strategy: prefer `cost`, fall back to upstream, then fall back to a
    local price-table estimate when the model is a Claude one. The
    `estimated` flag tells the renderer to prefix the value with `~`.
    """
    raw = data.get("usage") or {}
    in_tok = int(raw.get("prompt_tokens") or 0)
    out_tok = int(raw.get("completion_tokens") or 0)

    reported = _safe_float(raw.get("cost"))
    details = raw.get("cost_details") or {}
    upstream = _safe_float(details.get("upstream_inference_cost"))

    if reported is not None and reported > 0:
        return Usage(input_tokens=in_tok, output_tokens=out_tok, cost_usd=reported)
    if upstream is not None and upstream > 0:
        # BYOK: the user paid the upstream, not OpenRouter.
        return Usage(input_tokens=in_tok, output_tokens=out_tok, cost_usd=upstream)

    # cost is 0 or missing. If the model maps to our Anthropic price table,
    # estimate locally so the user still sees a number on BYOK / free tiers.
    if model.startswith("anthropic/"):
        bare = model.split("/", 1)[1]
        # OpenRouter uses dot-separated versions ("claude-sonnet-4.5"), our
        # table uses dashes ("claude-sonnet-4-5"). Try both.
        for candidate in (bare.replace(".", "-"), bare):
            est = anthropic_cost(candidate, in_tok, out_tok)
            if est is not None and est > 0:
                return Usage(
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=est,
                    estimated=True,
                )

    # cost is genuinely zero (or unknown). Surface tokens only —
    # `cost_usd=None` makes fmt_cost return None and the dollar figure drops.
    return Usage(input_tokens=in_tok, output_tokens=out_tok, cost_usd=None)


def request_grouping(
    config: Config,
    diff: str,
    files: list[str],
    recent_log: str,
) -> tuple[list[CommitGroup], Usage]:
    system = build_system_prompt(config.verbose_messages)
    user = build_user_message(diff, files, recent_log, config.conventions, config.context)
    if config.provider == PROVIDER_OPENROUTER:
        items, usage = _request_openrouter(config, system, user)
    else:
        items, usage = _request_anthropic(config, system, user)
    return _items_to_groups(items), usage


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


def render_plan(groups: list[CommitGroup], verbose: bool, model: str = "") -> None:
    n = len(groups)
    suffix = "" if n == 1 else "s"
    who = model_display_name(model) if model else "model"
    print(f"{who} suggests {n} commit{suffix}:\n")
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
        description="Split staged changes into atomic commits using an LLM.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["init", "setup"],
        default=None,
        help=(
            "Optional subcommand. "
            "'init' scaffolds a .smart-commit.toml at the repo root. "
            "'setup' installs shell tab-completion (bash / zsh / fish)."
        ),
    )
    parser.add_argument(
        "--shell",
        choices=["bash", "zsh", "fish"],
        default=None,
        help="Override shell auto-detection for the `setup` subcommand.",
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
    parser.add_argument(
        "--provider",
        choices=VALID_PROVIDERS,
        default=None,
        help="Override SMART_COMMIT_PROVIDER for this run (anthropic | openrouter).",
    )
    model_arg = parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override the model for this run. Accepts full IDs "
            "(e.g. 'claude-opus-4-7', 'openai/gpt-5') or short aliases "
            "('haiku', 'sonnet', 'opus'), which resolve to provider-specific IDs."
        ),
    )
    model_arg.completer = _model_completer  # type: ignore[attr-defined]

    parser.add_argument(
        "-m",
        "--context",
        action="append",
        default=None,
        metavar="TEXT",
        help=(
            "Free-form description of what this session covered (mirrors "
            "`git commit -m`). Can be repeated; multiple flags are joined "
            "with a space. Dramatically improves grouping accuracy on "
            "ambiguous diffs. 1-3 sentences is the sweet spot."
        ),
    )

    parser.add_argument(
        "--print-completion",
        choices=["bash", "zsh", "fish", "tcsh"],
        default=None,
        metavar="SHELL",
        help=(
            "Print the shell completion script for the chosen shell and exit. "
            "Add `eval \"$(smart-commit --print-completion zsh)\"` to your shell rc."
        ),
    )

    argcomplete.autocomplete(parser)
    return parser.parse_args(argv)


SETUP_MARKER_START = "# >>> smart-commit completion (managed by `smart-commit setup`) >>>"
SETUP_MARKER_END = "# <<< smart-commit completion <<<"


def _detect_shell() -> str:
    """Return the basename of $SHELL, or '' if undetectable."""
    sh = os.environ.get("SHELL", "")
    return Path(sh).name if sh else ""


def _setup_eval_shell(shell: str) -> int:
    """Append a guarded eval block to ~/.bashrc or ~/.zshrc. Idempotent."""
    rc_name = ".zshrc" if shell == "zsh" else ".bashrc"
    rc_path = Path.home() / rc_name
    eval_line = f'eval "$(smart-commit --print-completion {shell})"'

    if rc_path.exists():
        text = rc_path.read_text()
        if SETUP_MARKER_START in text or eval_line in text:
            print(f"smart-commit completion is already installed in {rc_path}.")
            print("(open a new shell to activate, or `source` the file)")
            return EXIT_OK
    else:
        text = ""

    block = "\n".join([
        "",
        SETUP_MARKER_START,
        eval_line,
        SETUP_MARKER_END,
        "",
    ])

    needs_leading_newline = text and not text.endswith("\n")
    with open(rc_path, "a") as f:
        if needs_leading_newline:
            f.write("\n")
        f.write(block)

    print(f"✓ Added smart-commit completion to {rc_path}")
    print(f"  Activate now: `source {rc_path}` (or open a new shell)")
    print("  To remove: delete the block between the smart-commit markers.")
    return EXIT_OK


def _setup_fish() -> int:
    """Write a dedicated completion file under fish's completions dir."""
    completions_dir = Path.home() / ".config" / "fish" / "completions"
    completions_dir.mkdir(parents=True, exist_ok=True)
    target = completions_dir / "smart-commit.fish"
    target.write_text(argcomplete.shellcode(["smart-commit"], shell="fish"))
    print(f"✓ Wrote fish completion to {target}")
    print("  Activate now: open a new fish shell.")
    return EXIT_OK


def cmd_setup(shell_override: str | None) -> int:
    """Install shell tab-completion for smart-commit. Idempotent."""
    shell = shell_override or _detect_shell()
    if not shell:
        print(
            "error: could not detect shell from $SHELL. "
            "Pass --shell {bash|zsh|fish}.",
            file=sys.stderr,
        )
        return EXIT_USER_ERROR
    if shell == "fish":
        return _setup_fish()
    if shell in ("bash", "zsh"):
        return _setup_eval_shell(shell)
    print(
        f"error: shell '{shell}' isn't supported by automatic setup. "
        f"Manually add to your shell rc: "
        f'eval "$(smart-commit --print-completion {shell})"',
        file=sys.stderr,
    )
    return EXIT_USER_ERROR


def cmd_init(repo_root: Path) -> int:
    """Scaffold a `.smart-commit.toml` at the repo root."""
    target = repo_root / CONFIG_FILENAME
    if target.exists():
        print(
            f"{CONFIG_FILENAME} already exists at {target}. Not overwriting.",
            file=sys.stderr,
        )
        return EXIT_USER_ERROR
    target.write_text(INIT_TEMPLATE)
    print(f"Created {target}")
    print(
        "Edit it to customize provider, model, conventions, trailers, etc. "
        "All keys are optional — leaving the file empty also silences the setup hint."
    )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.print_completion:
        print(argcomplete.shellcode(["smart-commit"], shell=args.print_completion))
        return EXIT_OK

    # `setup` doesn't need a git repo — it modifies the user's shell config.
    if args.command == "setup":
        return cmd_setup(args.shell)

    try:
        repo_root = git_repo_root()
    except GitError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USER_ERROR

    if args.command == "init":
        return cmd_init(repo_root)

    in_progress = git_in_progress_state()
    if in_progress:
        print(f"error: {in_progress} in progress; resolve it first.", file=sys.stderr)
        return EXIT_USER_ERROR

    try:
        config = build_config(args, repo_root)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USER_ERROR

    try:
        staged, rename_pairs = git_staged_status()
    except GitError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_GIT_ERROR

    if not staged:
        print("Nothing staged. Stage changes first (e.g. `git add ...`).", file=sys.stderr)
        return EXIT_USER_ERROR

    if not config.api_key:
        if config.provider == PROVIDER_OPENROUTER:
            print(
                "error: OPENROUTER_API_KEY is not set. "
                "Get one at https://openrouter.ai/keys and `export OPENROUTER_API_KEY=...`",
                file=sys.stderr,
            )
        else:
            print(
                "error: ANTHROPIC_API_KEY is not set. "
                "Get one at https://console.anthropic.com/ and `export ANTHROPIC_API_KEY=...`",
                file=sys.stderr,
            )
        return EXIT_USER_ERROR

    n = len(staged)
    suffix = "" if n == 1 else "s"
    print(f"Analyzing {n} staged file{suffix} via {config.provider} ({config.model})...")
    if config.context:
        print(f'Context: "{config.context}"')
    print()

    diff = truncate_diff(git_staged_diff())
    recent_log = git_recent_log()

    try:
        groups, usage = request_grouping(config, diff, staged, recent_log)
    except APICallError as e:
        print(f"error: API call failed: {e}", file=sys.stderr)
        return EXIT_API_ERROR
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"error: unexpected response shape from API: {e}", file=sys.stderr)
        return EXIT_API_ERROR

    errors = validate_groups(groups, staged)
    if errors:
        print(f"Validation errors in {model_display_name(config.model)}'s plan:", file=sys.stderr)
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

    render_plan(groups, config.verbose_messages, config.model)
    print(format_usage_line(usage))
    print()

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
                    render_plan(groups, config.verbose_messages, config.model)

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

    # Discoverability hint: nudge users toward a per-repo config when they
    # don't have one yet. Suppressed by an existing config file (even an
    # empty one — `touch .smart-commit.toml` works as an opt-out).
    if made > 0 and not (repo_root / CONFIG_FILENAME).exists():
        print()
        print("(tip: `smart-commit init` to customize behavior per-repo)")

    return EXIT_OK if made == total else EXIT_GIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
