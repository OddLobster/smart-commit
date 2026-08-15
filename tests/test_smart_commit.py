"""Unit + integration tests for smart-commit."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from conftest import (
    commit_body,
    commit_log_subjects,
    stage_binary_file,
    stage_files,
    staged_files,
)


# ----------------------------------------------------------------------
# Unit tests — pure functions
# ----------------------------------------------------------------------


def test_validate_groups_happy(sc):
    groups = [
        sc.CommitGroup(message="feat: x", files=["a.py", "b.py"]),
        sc.CommitGroup(message="chore: y", files=["c.py"]),
    ]
    assert sc.validate_groups(groups, ["a.py", "b.py", "c.py"]) == []


def test_validate_groups_duplicate_file(sc):
    groups = [
        sc.CommitGroup(message="feat: x", files=["a.py"]),
        sc.CommitGroup(message="feat: y", files=["a.py"]),
    ]
    errors = sc.validate_groups(groups, ["a.py"])
    assert any("appears in commits 1 and 2" in e for e in errors)


def test_validate_groups_unknown_file(sc):
    groups = [sc.CommitGroup(message="feat: x", files=["a.py", "ghost.py"])]
    errors = sc.validate_groups(groups, ["a.py"])
    assert any("'ghost.py' is not in the staged file list" in e for e in errors)


def test_validate_groups_unassigned(sc):
    groups = [sc.CommitGroup(message="feat: x", files=["a.py"])]
    errors = sc.validate_groups(groups, ["a.py", "b.py"])
    assert any("Files not in any commit: b.py" in e for e in errors)


def test_validate_groups_empty_message(sc):
    groups = [sc.CommitGroup(message="   ", files=["a.py"])]
    errors = sc.validate_groups(groups, ["a.py"])
    assert any("empty message" in e for e in errors)


def test_validate_groups_no_files(sc):
    groups = [sc.CommitGroup(message="feat: x", files=[])]
    errors = sc.validate_groups(groups, [])
    assert any("no files" in e for e in errors)


def test_truncate_diff_passes_small_through(sc):
    diff = "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -0 +1 @@\n+hello\n"
    assert sc.truncate_diff(diff) == diff


def test_truncate_diff_large_file(sc):
    body = "+x\n" * 500
    diff = "diff --git a/big b/big\n--- a/big\n+++ b/big\n@@ -0 +1,500 @@\n" + body
    out = sc.truncate_diff(diff, max_bytes=100, head_tail=10)
    assert "[... " in out and "lines truncated ...]" in out
    assert out.startswith("diff --git a/big b/big")


def test_truncate_diff_per_file(sc):
    a = "diff --git a/a b/a\n" + "+x\n" * 200
    b = "diff --git a/b b/b\n" + "+y\n" * 200
    out = sc.truncate_diff(a + b, max_bytes=100, head_tail=10)
    assert out.count("diff --git") == 2
    assert out.count("[... ") == 2


def test_truncate_diff_clips_few_pathologically_long_lines(sc):
    # A handful of lines (fewer than the head/tail line-count threshold) but
    # each individually huge — e.g. binary content forced into a text diff
    # by a gitattributes override. Line-count-based truncation alone would
    # never touch this; it must be caught by the byte-size check instead.
    huge_line = "+" + ("x" * 2_000_000)
    diff = f"diff --git a/weird.bin b/weird.bin\n--- a/weird.bin\n+++ b/weird.bin\n@@ -0,0 +1 @@\n{huge_line}\n"
    out = sc.truncate_diff(diff, max_bytes=1000)
    assert len(out.encode("utf-8")) < 5000
    assert "more chars truncated" in out
    assert out.startswith("diff --git a/weird.bin b/weird.bin")


def test_build_commit_message_subject_only(sc):
    cfg = sc.Config(verbose_messages=False, trailers=[])
    g = sc.CommitGroup(message="feat: hello", files=["a.py"], body="ignored")
    assert sc.build_commit_message(g, cfg) == "feat: hello"


def test_build_commit_message_with_body(sc):
    cfg = sc.Config(verbose_messages=True, trailers=[])
    g = sc.CommitGroup(
        message="feat: hello",
        files=["a.py"],
        body="A short body that should appear after a blank line.",
    )
    msg = sc.build_commit_message(g, cfg)
    lines = msg.split("\n")
    assert lines[0] == "feat: hello"
    assert lines[1] == ""
    assert "blank line" in msg


def test_build_commit_message_with_trailers(sc):
    cfg = sc.Config(verbose_messages=False, trailers=["Co-authored-by: X <x@example.com>"])
    g = sc.CommitGroup(message="feat: hello", files=["a.py"])
    msg = sc.build_commit_message(g, cfg)
    assert msg == "feat: hello\n\nCo-authored-by: X <x@example.com>"


def test_build_commit_message_body_and_trailers(sc):
    cfg = sc.Config(
        verbose_messages=True,
        trailers=["Signed-off-by: A <a@example.com>", "Co-authored-by: B <b@example.com>"],
    )
    g = sc.CommitGroup(message="feat: x", files=["a.py"], body="why and what")
    msg = sc.build_commit_message(g, cfg)
    parts = msg.split("\n\n")
    assert parts[0] == "feat: x"
    assert "why and what" in parts[1]
    assert parts[2] == "Signed-off-by: A <a@example.com>\nCo-authored-by: B <b@example.com>"


def test_toml_roundtrip(sc):
    groups = [
        sc.CommitGroup(
            message='feat(api): "quoted" thing',
            files=["a/b.py", "c.py"],
            body="line one\nline two",
            reasoning="because",
        ),
        sc.CommitGroup(message="chore: x", files=["d.py"], body="", reasoning="r"),
    ]
    text = sc.groups_to_toml(groups)
    parsed = sc.parse_toml_plan(text)
    assert len(parsed) == 2
    assert parsed[0].message == 'feat(api): "quoted" thing'
    assert parsed[0].body == "line one\nline two"
    assert parsed[0].files == ["a/b.py", "c.py"]
    assert parsed[1].message == "chore: x"
    assert parsed[1].body == ""


def test_toml_parse_missing_required(sc):
    with pytest.raises(ValueError):
        sc.parse_toml_plan('[[commit]]\nmessage = "x"\n')  # no files


# ----------------------------------------------------------------------
# Integration tests — real git, mocked Claude
# ----------------------------------------------------------------------


def _patch_grouping(monkeypatch, sc, groups, usage=None):
    if usage is None:
        usage = sc.Usage(input_tokens=1234, output_tokens=567, cost_usd=0.0042)

    def fake(config, diff, files, recent_log):
        return list(groups), usage

    def fake_repair(config, diff, files, recent_log, groups_in, errors):
        # Default: the repair round can't fix it either, so validation
        # failures still fall through to the Edit/Quit path.
        return list(groups), usage

    monkeypatch.setattr(sc, "request_grouping", fake)
    monkeypatch.setattr(sc, "request_grouping_repair", fake_repair)


def test_happy_path_two_groups(tmp_git_repo, sc, monkeypatch, capsys):
    stage_files(tmp_git_repo, {
        "feature.py": "def f(): pass\n",
        "feature_test.py": "def test_f(): pass\n",
        "ci.yml": "ci: yes\n",
    })
    groups = [
        sc.CommitGroup(message="feat: add feature", files=["feature.py", "feature_test.py"]),
        sc.CommitGroup(message="chore: add ci config", files=["ci.yml"]),
    ]
    _patch_grouping(monkeypatch, sc, groups)
    monkeypatch.setattr("builtins.input", lambda *_: "a")

    rc = sc.main([])
    assert rc == sc.EXIT_OK

    subjects = commit_log_subjects(tmp_git_repo)
    assert subjects[:2] == ["chore: add ci config", "feat: add feature"]
    assert staged_files(tmp_git_repo) == []


def test_dry_run_makes_no_commits(tmp_git_repo, sc, monkeypatch):
    stage_files(tmp_git_repo, {"a.py": "1\n", "b.py": "2\n"})
    groups = [
        sc.CommitGroup(message="feat: a", files=["a.py"]),
        sc.CommitGroup(message="feat: b", files=["b.py"]),
    ]
    _patch_grouping(monkeypatch, sc, groups)
    initial_count = len(commit_log_subjects(tmp_git_repo))

    rc = sc.main(["--dry-run"])
    assert rc == sc.EXIT_OK
    assert len(commit_log_subjects(tmp_git_repo)) == initial_count
    assert sorted(staged_files(tmp_git_repo)) == ["a.py", "b.py"]


def test_auto_skips_prompt(tmp_git_repo, sc, monkeypatch):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    groups = [sc.CommitGroup(message="feat: a", files=["a.py"])]
    _patch_grouping(monkeypatch, sc, groups)

    def no_input(*_):
        raise AssertionError("should not be prompted")

    monkeypatch.setattr("builtins.input", no_input)
    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_OK
    assert "feat: a" in commit_log_subjects(tmp_git_repo)


def test_quit_leaves_files_staged(tmp_git_repo, sc, monkeypatch):
    stage_files(tmp_git_repo, {"a.py": "1\n", "b.py": "2\n"})
    groups = [
        sc.CommitGroup(message="feat: a", files=["a.py"]),
        sc.CommitGroup(message="feat: b", files=["b.py"]),
    ]
    _patch_grouping(monkeypatch, sc, groups)
    monkeypatch.setattr("builtins.input", lambda *_: "q")
    initial = len(commit_log_subjects(tmp_git_repo))

    rc = sc.main([])
    assert rc == sc.EXIT_OK
    assert len(commit_log_subjects(tmp_git_repo)) == initial
    assert sorted(staged_files(tmp_git_repo)) == ["a.py", "b.py"]


def test_validation_failure_aborts_without_committing(tmp_git_repo, sc, monkeypatch, capsys):
    stage_files(tmp_git_repo, {"a.py": "1\n", "b.py": "2\n"})
    # Both groups claim a.py — duplicate.
    groups = [
        sc.CommitGroup(message="feat: a", files=["a.py"]),
        sc.CommitGroup(message="feat: b", files=["a.py", "b.py"]),
    ]
    _patch_grouping(monkeypatch, sc, groups)
    monkeypatch.setattr("builtins.input", lambda *_: "q")  # don't enter editor
    initial = len(commit_log_subjects(tmp_git_repo))

    rc = sc.main([])
    assert rc == sc.EXIT_VALIDATION_ERROR
    assert len(commit_log_subjects(tmp_git_repo)) == initial
    assert sorted(staged_files(tmp_git_repo)) == ["a.py", "b.py"]


def test_no_staged_changes_exits_cleanly(tmp_git_repo, sc, monkeypatch):
    rc = sc.main([])
    assert rc == sc.EXIT_USER_ERROR


def test_missing_api_key(tmp_git_repo, sc, monkeypatch):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    monkeypatch.delenv("SMART_COMMIT_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    rc = sc.main([])
    assert rc == sc.EXIT_USER_ERROR


def test_merge_in_progress_blocks_run(tmp_git_repo, sc, monkeypatch):
    # Simulate an in-progress merge by creating MERGE_HEAD in .git/.
    (tmp_git_repo / ".git" / "MERGE_HEAD").write_text(
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=tmp_git_repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    stage_files(tmp_git_repo, {"a.py": "1\n"})

    rc = sc.main([])
    assert rc == sc.EXIT_USER_ERROR


def test_rename_kept_in_one_group(tmp_git_repo, sc, monkeypatch):
    # Set up: commit a file, then rename it.
    (tmp_git_repo / "old.py").write_text("hello\n")
    subprocess.run(["git", "add", "old.py"], cwd=tmp_git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "add old"], cwd=tmp_git_repo, check=True
    )
    subprocess.run(
        ["git", "mv", "old.py", "new.py"], cwd=tmp_git_repo, check=True
    )

    paths, renames = sc.git_staged_status()
    assert paths == ["new.py"]
    assert renames == {"old.py": "new.py"}

    groups = [sc.CommitGroup(message="refactor: rename", files=["new.py"])]
    _patch_grouping(monkeypatch, sc, groups)
    monkeypatch.setattr("builtins.input", lambda *_: "a")

    rc = sc.main([])
    assert rc == sc.EXIT_OK
    assert "refactor: rename" in commit_log_subjects(tmp_git_repo)
    assert staged_files(tmp_git_repo) == []


def test_git_staged_binary_paths_detects_binary_file(tmp_git_repo, sc):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    stage_binary_file(tmp_git_repo, "icon.png")

    binary = sc.git_staged_binary_paths()
    assert binary == {"icon.png"}


def test_git_staged_diff_excludes_binary_content(tmp_git_repo, sc):
    stage_files(tmp_git_repo, {"feature.py": "def f(): pass\n"})
    stage_binary_file(tmp_git_repo, "icon.png", size=50_000)

    diff = sc.git_staged_diff()

    assert "def f(): pass" in diff
    assert "diff --git a/icon.png b/icon.png" in diff
    assert "Binary file (contents not included)" in diff
    # Raw binary bytes must never appear — only the placeholder line.
    assert len(diff) < 2000


def test_binary_files_never_reach_the_model(tmp_git_repo, sc, monkeypatch):
    """The diff string handed to request_grouping must stay small and
    placeholder-only for staged binary files, even for a large PNG."""
    stage_files(tmp_git_repo, {"feature.py": "def f(): pass\n"})
    stage_binary_file(tmp_git_repo, "assets/icon.png", size=500_000)
    captured: dict = {}

    def fake_request_grouping(config, diff, files, recent_log):
        captured["diff"] = diff
        return (
            [
                sc.CommitGroup(message="feat: a", files=["feature.py"]),
                sc.CommitGroup(message="chore: add icon", files=["assets/icon.png"]),
            ],
            sc.Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
        )

    monkeypatch.setattr(sc, "request_grouping", fake_request_grouping)
    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_OK
    assert len(captured["diff"]) < 2000
    assert "Binary file (contents not included)" in captured["diff"]


def test_gitignored_but_staged_file_commits(tmp_git_repo, sc, monkeypatch):
    # A user can force-stage a path matched by .gitignore (e.g. generated
    # snapshot files explicitly checked in). smart-commit must round-trip
    # those through reset+re-add without git refusing the re-add.
    (tmp_git_repo / ".gitignore").write_text("build/\n")
    (tmp_git_repo / "build").mkdir()
    (tmp_git_repo / "build" / "snapshot.json").write_text("{}\n")
    subprocess.run(
        ["git", "add", "-f", "build/snapshot.json"],
        cwd=tmp_git_repo, check=True,
    )

    groups = [
        sc.CommitGroup(message="test: refresh snapshot", files=["build/snapshot.json"])
    ]
    _patch_grouping(monkeypatch, sc, groups)

    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_OK
    assert "test: refresh snapshot" in commit_log_subjects(tmp_git_repo)
    assert staged_files(tmp_git_repo) == []


def test_verbose_writes_body_into_commit(tmp_git_repo, sc, monkeypatch):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    groups = [
        sc.CommitGroup(
            message="feat: a",
            files=["a.py"],
            body="Adds the a module which does the a thing.",
        )
    ]
    _patch_grouping(monkeypatch, sc, groups)

    rc = sc.main(["--verbose", "--auto"])
    assert rc == sc.EXIT_OK
    body = commit_body(tmp_git_repo)
    assert body.startswith("feat: a\n\n")
    assert "Adds the a module" in body


def test_trailers_appended_from_config(tmp_git_repo, sc, monkeypatch):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    (tmp_git_repo / ".smart-commit.toml").write_text(
        'trailers = ["Co-authored-by: Claude <noreply@anthropic.com>"]\n'
    )
    # The config file is unstaged, so it won't appear in staged_files.
    groups = [sc.CommitGroup(message="feat: a", files=["a.py"])]
    _patch_grouping(monkeypatch, sc, groups)

    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_OK
    body = commit_body(tmp_git_repo)
    assert body.rstrip().endswith("Co-authored-by: Claude <noreply@anthropic.com>")


def test_dry_run_wins_over_auto(tmp_git_repo, sc, monkeypatch):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    groups = [sc.CommitGroup(message="feat: a", files=["a.py"])]
    _patch_grouping(monkeypatch, sc, groups)
    initial = len(commit_log_subjects(tmp_git_repo))

    rc = sc.main(["--auto", "--dry-run"])
    assert rc == sc.EXIT_OK
    assert len(commit_log_subjects(tmp_git_repo)) == initial


# ----------------------------------------------------------------------
# Provider resolution + OpenRouter
# ----------------------------------------------------------------------


import argparse


def _ns(**kwargs):
    defaults = dict(
        auto=False,
        dry_run=False,
        verbose=None,
        model=None,
        context=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_build_config_defaults(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "test")
    monkeypatch.delenv("SMART_COMMIT_API_BASE", raising=False)
    monkeypatch.delenv("SMART_COMMIT_MODEL", raising=False)
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.model == sc.DEFAULT_MODEL
    assert cfg.base_url == sc.DEFAULT_API_BASE
    assert cfg.api_key == "test"


def test_build_config_api_base_env_override(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "test")
    monkeypatch.setenv("SMART_COMMIT_API_BASE", "https://litellm.example.com/v1")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.base_url == "https://litellm.example.com/v1"


def test_smart_commit_model_env_overrides_default(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "x")
    monkeypatch.setenv("SMART_COMMIT_MODEL", "anthropic/claude-haiku-4.5")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.model == "anthropic/claude-haiku-4.5"


def test_missing_api_key_message(tmp_git_repo, sc, monkeypatch, capsys):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    monkeypatch.delenv("SMART_COMMIT_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    rc = sc.main([])
    assert rc == sc.EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "SMART_COMMIT_API_KEY" in err


def test_openrouter_api_key_fallback(sc, tmp_path, monkeypatch):
    monkeypatch.delenv("SMART_COMMIT_API_KEY", raising=False)
    monkeypatch.delenv("SMART_COMMIT_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.api_key == "or-key"


def test_openrouter_api_key_ignored_for_other_base(sc, tmp_path, monkeypatch):
    monkeypatch.delenv("SMART_COMMIT_API_KEY", raising=False)
    monkeypatch.setenv("SMART_COMMIT_API_BASE", "https://litellm.example.com/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.api_key == ""


def test_smart_commit_api_key_wins_over_openrouter(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "sc-key")
    monkeypatch.delenv("SMART_COMMIT_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.api_key == "sc-key"


def test_openrouter_request_happy_path(sc, monkeypatch):
    """Mock httpx and verify the OpenRouter path parses a normal response."""
    captured: dict = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

        @property
        def text(self):
            return json.dumps(self._payload)

        @property
        def status_code(self):
            return 200

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResponse({
                "choices": [
                    {
                        "message": {
                            "content": '{"commits":[{"message":"feat: x","files":["a.py"],"body":"","reasoning":"r"}]}'
                        }
                    }
                ]
            })

    monkeypatch.setattr(sc.httpx, "Client", FakeClient)

    cfg = sc.Config(
        model="anthropic/claude-sonnet-4.6",
        api_key="or-test",
        base_url=sc.DEFAULT_API_BASE,
    )
    items, usage = sc._request_completion(cfg, "system text", "user text", ["a.py"])
    assert items == [{"message": "feat: x", "files": ["a.py"], "body": "", "reasoning": "r"}]
    assert isinstance(usage, sc.Usage)
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer or-test"
    assert captured["payload"]["model"] == "anthropic/claude-sonnet-4.6"
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    # The deprecated `usage: {include: true}` opt-in is no longer sent — cost
    # is always included by OpenRouter automatically.
    assert "usage" not in captured["payload"]


def test_openrouter_strips_markdown_fences(sc, monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers, json):
            return FakeResponse({
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"commits":[{"message":"feat: x","files":["a.py"],"body":"","reasoning":"r"}]}\n```'
                        }
                    }
                ]
            })

    monkeypatch.setattr(sc.httpx, "Client", FakeClient)
    cfg = sc.Config(
        model="some/model",
        api_key="or-test",
        base_url=sc.DEFAULT_API_BASE,
    )
    items, _usage = sc._request_completion(cfg, "system", "user", ["a.py"])
    assert items[0]["message"] == "feat: x"


def test_openrouter_retries_on_bad_json(sc, monkeypatch):
    """First response is unparseable; second is valid. Retry should succeed."""

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    calls = {"n": 0}
    bad_payload = {"choices": [{"message": {"content": "not json at all, sorry"}}]}
    good_payload = {
        "choices": [
            {
                "message": {
                    "content": '{"commits":[{"message":"chore: y","files":["b.py"],"body":"","reasoning":"r"}]}'
                }
            }
        ]
    }

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers, json):
            calls["n"] += 1
            return FakeResponse(bad_payload if calls["n"] == 1 else good_payload)

    monkeypatch.setattr(sc.httpx, "Client", FakeClient)
    cfg = sc.Config(
        model="some/model",
        api_key="or-test",
        base_url=sc.DEFAULT_API_BASE,
    )
    items, _usage = sc._request_completion(cfg, "system", "user", ["b.py"])
    assert calls["n"] == 2
    assert items[0]["files"] == ["b.py"]


def test_qwen37_flash_uses_compatible_request_options(sc, monkeypatch):
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"commits":[{"message":"feat: x","files":[1],"body":"","reasoning":"r"}]}'
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers, json):
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(sc.httpx, "Client", FakeClient)
    cfg = sc.Config(
        model="qwen/qwen3.7-flash",
        api_key="or-test",
        base_url=sc.DEFAULT_API_BASE,
    )
    items, _usage = sc._request_completion(cfg, "system", "user", ["a.py"])
    assert items[0]["files"] == ["a.py"]
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["reasoning"] == {"enabled": False}


def test_null_content_retries_without_assistant_null_turn(sc, monkeypatch):
    calls: list[dict] = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    responses = [
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "native_finish_reason": "max_tokens",
                    "message": {"content": None, "reasoning": "thinking"},
                }
            ],
            "usage": {"completion_tokens_details": {"reasoning_tokens": 4096}},
        },
        {
            "choices": [
                {
                    "message": {
                        "content": '{"commits":[{"message":"feat: x","files":[1],"body":"","reasoning":"r"}]}'
                    }
                }
            ]
        },
    ]

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers, json):
            calls.append(json)
            return FakeResponse(responses[len(calls) - 1])

    monkeypatch.setattr(sc.httpx, "Client", FakeClient)
    cfg = sc.Config(model="some/model", api_key="k", base_url=sc.DEFAULT_API_BASE)
    items, _usage = sc._request_completion(cfg, "system", "user", ["a.py"])
    assert items[0]["files"] == ["a.py"]
    assert len(calls) == 2
    retry_messages = calls[1]["messages"]
    assert [m["role"] for m in retry_messages] == ["system", "user", "user"]
    assert "finish_reason='length'" in retry_messages[-1]["content"]
    assert "reasoning_tokens=4096" in retry_messages[-1]["content"]


def test_persistent_null_content_raises_diagnostic_api_error(sc, monkeypatch):
    calls = {"payloads": []}
    _make_fake_client(sc, monkeypatch, [None], calls)
    cfg = sc.Config(model="some/model", api_key="k", base_url=sc.DEFAULT_API_BASE)
    with pytest.raises(sc.APICallError, match="model returned no text content"):
        sc._request_completion(cfg, "system", "user", ["a.py"])
    assert len(calls["payloads"]) == 2


# ----------------------------------------------------------------------
# Numeric file IDs (anti-hallucination)
# ----------------------------------------------------------------------


def test_resolve_file_ref_numeric_ids(sc):
    files = ["src/a.py", "src/b.py", "docs/c.md"]
    assert sc._resolve_file_ref(1, files) == "src/a.py"
    assert sc._resolve_file_ref(3, files) == "docs/c.md"
    assert sc._resolve_file_ref("2", files) == "src/b.py"  # stringified ID


def test_resolve_file_ref_exact_path_fallback(sc):
    files = ["src/a.py", "src/b.py"]
    assert sc._resolve_file_ref("src/b.py", files) == "src/b.py"


def test_resolve_file_ref_rejects_bad_refs(sc):
    files = ["src/a.py"]
    with pytest.raises(ValueError):
        sc._resolve_file_ref(0, files)
    with pytest.raises(ValueError):
        sc._resolve_file_ref(2, files)
    with pytest.raises(ValueError):
        sc._resolve_file_ref(True, files)
    with pytest.raises(ValueError):
        sc._resolve_file_ref("src/hallucinated.py", files)
    with pytest.raises(ValueError):
        sc._resolve_file_ref(None, files)


def _make_fake_client(sc, monkeypatch, responses: list[str], calls: dict):
    """Patch httpx.Client so successive posts return the given message contents."""

    class FakeResponse:
        def __init__(self, content):
            self._content = content

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": self._content}}]}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers, json):
            calls["payloads"].append(json)
            content = responses[min(len(calls["payloads"]) - 1, len(responses) - 1)]
            return FakeResponse(content)

    monkeypatch.setattr(sc.httpx, "Client", FakeClient)


def test_completion_resolves_numeric_ids_to_paths(sc, monkeypatch):
    calls = {"payloads": []}
    _make_fake_client(sc, monkeypatch, [
        '{"commits":[{"message":"feat: x","files":[2,1],"body":"","reasoning":"r"}]}',
    ], calls)
    cfg = sc.Config(model="m", api_key="k", base_url=sc.DEFAULT_API_BASE)
    items, _usage = sc._request_completion(cfg, "system", "user", ["src/a.py", "src/b.py"])
    assert items[0]["files"] == ["src/b.py", "src/a.py"]


def test_completion_retries_on_hallucinated_path(sc, monkeypatch):
    """A path not in the staged list is a parse failure: one corrective retry."""
    calls = {"payloads": []}
    _make_fake_client(sc, monkeypatch, [
        '{"commits":[{"message":"feat: x","files":["src/shim/a.py"],"body":"","reasoning":"r"}]}',
        '{"commits":[{"message":"feat: x","files":[1],"body":"","reasoning":"r"}]}',
    ], calls)
    cfg = sc.Config(model="m", api_key="k", base_url=sc.DEFAULT_API_BASE)
    items, _usage = sc._request_completion(cfg, "system", "user", ["src/a.py"])
    assert len(calls["payloads"]) == 2
    assert items[0]["files"] == ["src/a.py"]
    # The corrective turn names the bad reference and asks for numeric IDs.
    retry_msg = calls["payloads"][1]["messages"][-1]["content"]
    assert "src/shim/a.py" in retry_msg
    assert "numeric ID" in retry_msg


def test_completion_retries_on_out_of_range_id(sc, monkeypatch):
    calls = {"payloads": []}
    _make_fake_client(sc, monkeypatch, [
        '{"commits":[{"message":"x","files":[5],"body":"","reasoning":"r"}]}',
        '{"commits":[{"message":"x","files":[1,2],"body":"","reasoning":"r"}]}',
    ], calls)
    cfg = sc.Config(model="m", api_key="k", base_url=sc.DEFAULT_API_BASE)
    items, _usage = sc._request_completion(cfg, "system", "user", ["a.py", "b.py"])
    assert len(calls["payloads"]) == 2
    assert items[0]["files"] == ["a.py", "b.py"]


def test_completion_fails_after_persistent_hallucination(sc, monkeypatch):
    calls = {"payloads": []}
    _make_fake_client(sc, monkeypatch, [
        '{"commits":[{"message":"x","files":["nope.py"],"body":"","reasoning":"r"}]}',
    ], calls)
    cfg = sc.Config(model="m", api_key="k", base_url=sc.DEFAULT_API_BASE)
    with pytest.raises(sc.APICallError):
        sc._request_completion(cfg, "system", "user", ["a.py"])
    assert len(calls["payloads"]) == 2


def test_schema_files_are_integers(sc):
    items = sc.RESPONSE_SCHEMA["properties"]["commits"]["items"]
    assert items["properties"]["files"]["items"] == {"type": "integer"}


def test_user_message_numbers_staged_files(sc):
    msg = sc.build_user_message(
        diff="d",
        files=["src/a.py", "src/b.py"],
        recent_log="",
        conventions="",
        context="",
    )
    assert "1. src/a.py" in msg
    assert "2. src/b.py" in msg
    assert "numeric ID" in msg


def test_openrouter_extracts_cost_from_response(sc, monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers, json):
            return FakeResponse({
                "choices": [
                    {
                        "message": {
                            "content": '{"commits":[{"message":"x","files":["a"],"body":"","reasoning":"r"}]}'
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 4321,
                    "completion_tokens": 890,
                    "cost": 0.00342,
                },
            })

    monkeypatch.setattr(sc.httpx, "Client", FakeClient)
    cfg = sc.Config(
        model="some/model",
        api_key="or-test",
        base_url=sc.DEFAULT_API_BASE,
    )
    _items, usage = sc._request_completion(cfg, "system", "user", ["a"])
    assert usage.input_tokens == 4321
    assert usage.output_tokens == 890
    assert usage.cost_usd == pytest.approx(0.00342)


def test_openrouter_byok_uses_upstream_inference_cost(sc, monkeypatch):
    """When OpenRouter reports cost=0 but upstream cost is set (BYOK), use upstream."""

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers, json):
            return FakeResponse({
                "choices": [
                    {
                        "message": {
                            "content": '{"commits":[{"message":"x","files":["a"],"body":"","reasoning":"r"}]}'
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 3400,
                    "completion_tokens": 181,
                    "cost": 0,
                    "cost_details": {"upstream_inference_cost": 0.0125},
                },
            })

    monkeypatch.setattr(sc.httpx, "Client", FakeClient)
    cfg = sc.Config(
        model="anthropic/claude-sonnet-4.6",
        api_key="or-test",
        base_url=sc.DEFAULT_API_BASE,
    )
    _items, usage = sc._request_completion(cfg, "system", "user", ["a"])
    assert usage.cost_usd == pytest.approx(0.0125)
    assert usage.estimated is False  # upstream is reported, not estimated


def test_openrouter_byok_falls_back_to_local_estimate_for_anthropic(sc, monkeypatch):
    """BYOK with no upstream cost field: estimate from token counts via the price table."""

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers, json):
            return FakeResponse({
                "choices": [
                    {
                        "message": {
                            "content": '{"commits":[{"message":"x","files":["a"],"body":"","reasoning":"r"}]}'
                        }
                    }
                ],
                # cost=0, no cost_details — typical BYOK shape
                "usage": {"prompt_tokens": 4000, "completion_tokens": 1000, "cost": 0},
            })

    monkeypatch.setattr(sc.httpx, "Client", FakeClient)
    cfg = sc.Config(
        # OpenRouter uses dot-separated versions; our local table uses dashes.
        # The fallback should normalize the ID.
        model="anthropic/claude-sonnet-4.6",
        api_key="or-test",
        base_url=sc.DEFAULT_API_BASE,
    )
    _items, usage = sc._request_completion(cfg, "system", "user", ["a"])
    # 4000 in @ $3/M + 1000 out @ $15/M = 0.012 + 0.015 = 0.027
    assert usage.cost_usd == pytest.approx(0.027)
    assert usage.estimated is True


def test_openrouter_zero_cost_non_anthropic_yields_none(sc, monkeypatch):
    """For non-Anthropic models with cost=0, we have no price table — show no $."""

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers, json):
            return FakeResponse({
                "choices": [
                    {
                        "message": {
                            "content": '{"commits":[{"message":"x","files":["a"],"body":"","reasoning":"r"}]}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0},
            })

    monkeypatch.setattr(sc.httpx, "Client", FakeClient)
    cfg = sc.Config(
        model="qwen/qwen3-coder",
        api_key="or-test",
        base_url=sc.DEFAULT_API_BASE,
    )
    _items, usage = sc._request_completion(cfg, "system", "user", ["a"])
    assert usage.cost_usd is None
    assert usage.estimated is False


def test_format_usage_line_estimated_prefixes_tilde(sc):
    usage = sc.Usage(input_tokens=4000, output_tokens=1000, cost_usd=0.027, estimated=True)
    assert sc.format_usage_line(usage) == "(~4.0k in · ~1.0k out · ~$0.027)"


def test_format_usage_line_not_estimated_no_tilde(sc):
    usage = sc.Usage(input_tokens=4000, output_tokens=1000, cost_usd=0.027, estimated=False)
    assert sc.format_usage_line(usage) == "(~4.0k in · ~1.0k out · $0.027)"


def test_openrouter_missing_cost_yields_none(sc, monkeypatch):
    """When OpenRouter doesn't return cost, usage.cost_usd should be None."""

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers, json):
            return FakeResponse({
                "choices": [
                    {
                        "message": {
                            "content": '{"commits":[{"message":"x","files":["a"],"body":"","reasoning":"r"}]}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            })

    monkeypatch.setattr(sc.httpx, "Client", FakeClient)
    cfg = sc.Config(
        model="x",
        api_key="k",
        base_url=sc.DEFAULT_API_BASE,
    )
    _items, usage = sc._request_completion(cfg, "s", "u", ["a"])
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.cost_usd is None


def test_openrouter_http_error_raises_apicallerror(sc, monkeypatch):
    class FakeResponse:
        status_code = 401
        text = "Unauthorized"

        def raise_for_status(self):
            raise sc.httpx.HTTPStatusError("401", request=None, response=self)

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers, json):
            return FakeResponse()

    monkeypatch.setattr(sc.httpx, "Client", FakeClient)
    cfg = sc.Config(
        model="some/model",
        api_key="bad",
        base_url=sc.DEFAULT_API_BASE,
    )
    with pytest.raises(sc.APICallError):
        sc._request_completion(cfg, "system", "user", [])


# ----------------------------------------------------------------------
# Model aliases + --model flag
# ----------------------------------------------------------------------


def test_resolve_model_aliases(sc):
    assert sc.resolve_model("haiku") == "anthropic/claude-haiku-4.5"
    assert sc.resolve_model("sonnet") == "anthropic/claude-sonnet-4.6"
    assert sc.resolve_model("opus") == "anthropic/claude-opus-4.7"


def test_resolve_model_passes_unknown_through(sc):
    assert sc.resolve_model("openai/gpt-5") == "openai/gpt-5"
    assert sc.resolve_model("claude-haiku-4-5") == "claude-haiku-4-5"


def test_resolve_model_alias_case_insensitive(sc):
    assert sc.resolve_model("HAIKU") == "anthropic/claude-haiku-4.5"
    assert sc.resolve_model(" Sonnet ") == "anthropic/claude-sonnet-4.6"


def test_model_cli_flag_wins_over_env(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "x")
    monkeypatch.setenv("SMART_COMMIT_MODEL", "anthropic/claude-opus-4.7")
    cfg = sc.build_config(_ns(model="haiku"), tmp_path)
    assert cfg.model == "anthropic/claude-haiku-4.5"


def test_model_env_wins_over_config(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "x")
    (tmp_path / sc.CONFIG_FILENAME).write_text('model = "sonnet"\n')
    monkeypatch.setenv("SMART_COMMIT_MODEL", "haiku")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.model == "anthropic/claude-haiku-4.5"


# ----------------------------------------------------------------------
# Cost / usage formatting
# ----------------------------------------------------------------------


def test_anthropic_cost_known_model(sc):
    # 4000 in @ $3/M + 1000 out @ $15/M = 0.012 + 0.015 = 0.027
    assert sc.anthropic_cost("claude-sonnet-4-6", 4000, 1000) == pytest.approx(0.027)


def test_anthropic_cost_unknown_model(sc):
    assert sc.anthropic_cost("not-a-real-model", 1000, 1000) is None


def test_fmt_tokens(sc):
    assert sc.fmt_tokens(42) == "42"
    assert sc.fmt_tokens(890) == "890"
    assert sc.fmt_tokens(4200) == "~4.2k"
    assert sc.fmt_tokens(15000) == "~15k"


def test_fmt_cost(sc):
    assert sc.fmt_cost(None) is None
    assert sc.fmt_cost(0) == "$0"
    assert sc.fmt_cost(0.0005) == "<$0.001"
    assert sc.fmt_cost(0.003) == "$0.003"
    assert sc.fmt_cost(1.23) == "$1.23"
    assert sc.fmt_cost(150) == "$150"


def test_format_usage_line_with_cost(sc):
    usage = sc.Usage(input_tokens=4200, output_tokens=890, cost_usd=0.003)
    assert sc.format_usage_line(usage) == "(~4.2k in · 890 out · $0.003)"


def test_format_usage_line_no_cost(sc):
    usage = sc.Usage(input_tokens=100, output_tokens=50, cost_usd=None)
    assert sc.format_usage_line(usage) == "(100 in · 50 out)"


def test_model_display_name_strips_namespace(sc):
    assert sc.model_display_name("qwen/qwen3.6-flash") == "qwen3.6-flash"
    assert sc.model_display_name("openai/gpt-5") == "gpt-5"
    assert sc.model_display_name("anthropic/claude-sonnet-4.6") == "claude-sonnet-4.6"
    assert sc.model_display_name("claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert sc.model_display_name("") == "model"


def test_model_completer_returns_curated_list(sc):
    suggestions = sc._model_completer("", argparse.Namespace())
    assert "haiku" in suggestions
    assert "sonnet" in suggestions
    assert "opus" in suggestions
    assert "anthropic/claude-sonnet-4.6" in suggestions
    assert "openai/gpt-5" in suggestions
    assert "qwen/qwen3-coder" in suggestions
    # Aliases come first so tab cycles them before full IDs.
    assert suggestions.index("haiku") < suggestions.index("anthropic/claude-sonnet-4.6")


# ----------------------------------------------------------------------
# Spinner
# ----------------------------------------------------------------------


class _FakeTTY:
    """Stream that quacks like a TTY and captures writes."""

    def __init__(self):
        self.buf = []

    def write(self, s):
        self.buf.append(s)
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return True


class _FakeNonTTY:
    def __init__(self):
        self.buf = []

    def write(self, s):
        self.buf.append(s)
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False


def test_spinner_disabled_when_not_tty(sc, monkeypatch):
    monkeypatch.delenv("SMART_COMMIT_NO_SPINNER", raising=False)
    stream = _FakeNonTTY()
    with sc.Spinner("hello", stream=stream):
        pass
    assert stream.buf == []  # nothing written, including no clear sequence


def test_spinner_disabled_via_env(sc, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_NO_SPINNER", "1")
    stream = _FakeTTY()
    with sc.Spinner("hello", stream=stream):
        pass
    assert stream.buf == []


def test_spinner_runs_when_tty(sc, monkeypatch):
    """Smoke test: with a TTY-like stream, the spinner thread writes at least one frame."""
    import time

    monkeypatch.delenv("SMART_COMMIT_NO_SPINNER", raising=False)
    stream = _FakeTTY()
    with sc.Spinner("doing work", stream=stream):
        time.sleep(0.15)  # roughly 2 frames at default 80ms cadence
    output = "".join(stream.buf)
    # At least one frame char and the message should have been written.
    assert any(ch in output for ch in sc.Spinner.FRAMES)
    assert "doing work" in output
    # Cleanup sequence cleared the line on exit.
    assert sc.Spinner.CLEAR_LINE in output


def test_spinner_animates_trailing_dots(sc):
    """Dots cycle through "" / "." / ".." / "..." at DOT_TICKS frames each."""
    s = sc.Spinner("Thinking", stream=_FakeNonTTY())
    ticks = sc.Spinner.DOT_TICKS
    assert s._frame(0).endswith("Thinking")            # i=0 → no dots
    assert s._frame(ticks).endswith("Thinking.")       # i=DOT_TICKS → 1 dot
    assert s._frame(2 * ticks).endswith("Thinking..")  # i=2*DOT_TICKS → 2 dots
    assert s._frame(3 * ticks).endswith("Thinking...") # i=3*DOT_TICKS → 3 dots
    assert s._frame(4 * ticks).endswith("Thinking")    # wraps back


def test_spinner_strips_trailing_dots_from_message(sc):
    """Caller-supplied trailing dots/spaces are stripped — Spinner owns the animation."""
    stream = _FakeNonTTY()
    s = sc.Spinner("Working...", stream=stream)
    assert s.message == "Working"
    s2 = sc.Spinner("Working ... ", stream=stream)
    assert s2.message == "Working"
    # Dots in the middle of the message are preserved.
    s3 = sc.Spinner("Doing thing.foo (bar)", stream=stream)
    assert s3.message == "Doing thing.foo (bar)"


def test_spinner_cleans_up_on_exception(sc, monkeypatch):
    monkeypatch.delenv("SMART_COMMIT_NO_SPINNER", raising=False)
    stream = _FakeTTY()
    with pytest.raises(RuntimeError):
        with sc.Spinner("crashing soon", stream=stream):
            raise RuntimeError("boom")
    output = "".join(stream.buf)
    # The clear sequence must run even on exception so the cursor is parked
    # on a clean line for the error message that follows.
    assert sc.Spinner.CLEAR_LINE in output


# ----------------------------------------------------------------------
# `smart-commit init` subcommand
# ----------------------------------------------------------------------


def test_init_creates_config_file(tmp_git_repo, sc, capsys):
    target = tmp_git_repo / sc.CONFIG_FILENAME
    assert not target.exists()
    rc = sc.main(["init"])
    assert rc == sc.EXIT_OK
    assert target.exists()
    out = capsys.readouterr().out
    assert str(target) in out


def test_init_does_not_clobber(tmp_git_repo, sc, capsys):
    target = tmp_git_repo / sc.CONFIG_FILENAME
    target.write_text("# my custom config\nverbose_messages = true\n")
    original = target.read_text()

    rc = sc.main(["init"])
    assert rc == sc.EXIT_USER_ERROR
    assert target.read_text() == original
    err = capsys.readouterr().err
    assert "already exists" in err


def test_init_template_is_valid_toml(tmp_git_repo, sc):
    """Generated file must parse cleanly with tomllib."""
    import tomllib

    rc = sc.main(["init"])
    assert rc == sc.EXIT_OK
    target = tmp_git_repo / sc.CONFIG_FILENAME
    parsed = tomllib.loads(target.read_text())
    # Everything should be commented out → empty dict
    assert parsed == {}


def test_init_template_passes_through_load_config(tmp_git_repo, sc, monkeypatch):
    """Generated file must round-trip through build_config without errors."""
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "x")
    sc.main(["init"])
    cfg = sc.build_config(_ns(), tmp_git_repo)
    # All-commented template should produce default config.
    assert cfg.context == ""
    assert cfg.trailers == []
    assert cfg.verbose_messages is False


def test_init_outside_repo_errors(tmp_path, sc, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = sc.main(["init"])
    assert rc == sc.EXIT_USER_ERROR


def test_init_writes_config_at_repo_root_not_cwd(tmp_git_repo, sc, monkeypatch):
    """init writes to the repo root, even when invoked from a subdirectory."""
    sub = tmp_git_repo / "deep" / "nested"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    rc = sc.main(["init"])
    assert rc == sc.EXIT_OK
    assert (tmp_git_repo / sc.CONFIG_FILENAME).exists()
    assert not (sub / sc.CONFIG_FILENAME).exists()


# ----------------------------------------------------------------------
# `smart-commit setup` (shell completion installer)
# ----------------------------------------------------------------------


def test_setup_zsh_appends_to_zshrc(tmp_path, sc, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    rc = tmp_path / ".zshrc"
    rc.write_text("# user's existing zshrc\nexport FOO=bar\n")

    rc_pre = rc.read_text()
    rc = sc.main(["setup"])  # rebinds local 'rc' on purpose? no — keep file ref
    # Re-resolve since we shadowed: just use the path again.
    rc_path = tmp_path / ".zshrc"
    text = rc_path.read_text()

    assert text.startswith(rc_pre)  # existing content preserved
    assert "smart-commit --print-completion zsh" in text
    assert sc.SETUP_MARKER_START in text
    assert sc.SETUP_MARKER_END in text


def test_setup_bash_appends_to_bashrc(tmp_path, sc, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/bash")
    rc_path = tmp_path / ".bashrc"
    rc_path.write_text("# bash setup\n")

    code = sc.main(["setup"])
    assert code == sc.EXIT_OK
    text = rc_path.read_text()
    assert "smart-commit --print-completion bash" in text


def test_setup_creates_rc_if_missing(tmp_path, sc, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    rc_path = tmp_path / ".zshrc"
    assert not rc_path.exists()

    code = sc.main(["setup"])
    assert code == sc.EXIT_OK
    assert rc_path.exists()
    assert "smart-commit --print-completion zsh" in rc_path.read_text()


def test_setup_is_idempotent(tmp_path, sc, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    rc_path = tmp_path / ".zshrc"

    assert sc.main(["setup"]) == sc.EXIT_OK
    after_first = rc_path.read_text()
    capsys.readouterr()  # clear

    assert sc.main(["setup"]) == sc.EXIT_OK
    after_second = rc_path.read_text()
    assert after_first == after_second  # no duplicate eval block

    out = capsys.readouterr().out
    assert "already installed" in out


def test_setup_fish_writes_completion_file(tmp_path, sc, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/usr/bin/fish")

    code = sc.main(["setup"])
    assert code == sc.EXIT_OK
    target = tmp_path / ".config" / "fish" / "completions" / "smart-commit.fish"
    assert target.exists()
    assert "smart-commit" in target.read_text()


def test_setup_shell_flag_overrides_detection(tmp_path, sc, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/bash")  # shell says bash...
    code = sc.main(["setup", "--shell", "zsh"])  # ...but flag says zsh
    assert code == sc.EXIT_OK
    assert (tmp_path / ".zshrc").exists()
    assert not (tmp_path / ".bashrc").exists()


def test_setup_unsupported_shell_errors(tmp_path, sc, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/usr/bin/tcsh")
    code = sc.main(["setup"])
    assert code == sc.EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "tcsh" in err
    assert "--print-completion" in err  # tells user the manual fallback


def test_setup_no_shell_env_errors(tmp_path, sc, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SHELL", raising=False)
    code = sc.main(["setup"])
    assert code == sc.EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "could not detect shell" in err


def test_setup_runs_outside_git_repo(tmp_path, sc, monkeypatch):
    """setup modifies user shell config — it shouldn't require a git repo."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.chdir(tmp_path)  # plain dir, no .git
    code = sc.main(["setup"])
    assert code == sc.EXIT_OK


# ----------------------------------------------------------------------
# Post-run setup hint
# ----------------------------------------------------------------------


def test_hint_shown_when_no_config(tmp_git_repo, sc, monkeypatch, capsys):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    groups = [sc.CommitGroup(message="feat: a", files=["a.py"])]
    _patch_grouping(monkeypatch, sc, groups)
    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_OK
    out = capsys.readouterr().out
    assert "smart-commit init" in out


def test_hint_hidden_when_config_exists(tmp_git_repo, sc, monkeypatch, capsys):
    (tmp_git_repo / sc.CONFIG_FILENAME).write_text("# custom config\n")
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    groups = [sc.CommitGroup(message="feat: a", files=["a.py"])]
    _patch_grouping(monkeypatch, sc, groups)
    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_OK
    out = capsys.readouterr().out
    assert "smart-commit init" not in out


def test_hint_hidden_when_empty_config_exists(tmp_git_repo, sc, monkeypatch, capsys):
    """An empty .smart-commit.toml is a valid opt-out for the hint."""
    (tmp_git_repo / sc.CONFIG_FILENAME).touch()
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    groups = [sc.CommitGroup(message="feat: a", files=["a.py"])]
    _patch_grouping(monkeypatch, sc, groups)
    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_OK
    out = capsys.readouterr().out
    assert "smart-commit init" not in out


def test_hint_hidden_on_dry_run(tmp_git_repo, sc, monkeypatch, capsys):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    groups = [sc.CommitGroup(message="feat: a", files=["a.py"])]
    _patch_grouping(monkeypatch, sc, groups)
    rc = sc.main(["--dry-run"])
    assert rc == sc.EXIT_OK
    out = capsys.readouterr().out
    assert "smart-commit init" not in out


def test_hint_hidden_on_quit(tmp_git_repo, sc, monkeypatch, capsys):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    groups = [sc.CommitGroup(message="feat: a", files=["a.py"])]
    _patch_grouping(monkeypatch, sc, groups)
    monkeypatch.setattr("builtins.input", lambda *_: "q")
    rc = sc.main([])
    assert rc == sc.EXIT_OK
    out = capsys.readouterr().out
    assert "smart-commit init" not in out


# ----------------------------------------------------------------------
# Developer context
# ----------------------------------------------------------------------


def test_context_in_prompt_when_provided(sc):
    msg = sc.build_user_message(
        diff="diff",
        files=["a.py"],
        recent_log="abc add foo",
        conventions="conv",
        context="added license endpoint, also windows path fix",
    )
    assert "## Developer context" in msg
    assert "added license endpoint, also windows path fix" in msg
    # Context section appears BEFORE recent commit style and the diff.
    assert msg.index("## Developer context") < msg.index("## Recent commit style")
    assert msg.index("## Developer context") < msg.index("## Full diff")


def test_context_omitted_when_empty(sc):
    msg = sc.build_user_message(
        diff="diff",
        files=["a.py"],
        recent_log="",
        conventions="",
        context="",
    )
    assert "## Developer context" not in msg
    assert "developer describes" not in msg.lower()


def test_context_omitted_when_whitespace_only(sc):
    msg = sc.build_user_message(
        diff="d", files=["a"], recent_log="", conventions="", context="   \n  "
    )
    assert "## Developer context" not in msg


def test_context_priority_cli_wins_over_env(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "x")
    monkeypatch.setenv("SMART_COMMIT_CONTEXT", "from env")
    cfg = sc.build_config(_ns(context=["from cli"]), tmp_path)
    assert cfg.context == "from cli"


def test_context_env_used_when_no_cli(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "x")
    monkeypatch.setenv("SMART_COMMIT_CONTEXT", "from env")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.context == "from env"


def test_context_config_baseline_concatenated_with_per_run(sc, tmp_path, monkeypatch):
    """Config provides a persistent baseline that gets joined with CLI/env per-run."""
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "x")
    (tmp_path / sc.CONFIG_FILENAME).write_text('context = "v2 migration branch"\n')
    cfg = sc.build_config(_ns(context=["new license endpoint"]), tmp_path)
    assert cfg.context == "v2 migration branch new license endpoint"


def test_context_config_only(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "x")
    (tmp_path / sc.CONFIG_FILENAME).write_text('context = "long-running auth rewrite"\n')
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.context == "long-running auth rewrite"


def test_context_multiple_cli_flags_concatenate(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "x")
    cfg = sc.build_config(_ns(context=["new license endpoint", "windows path fix"]), tmp_path)
    assert cfg.context == "new license endpoint windows path fix"


def test_context_empty_cli_flag_treated_as_no_context(sc, tmp_path, monkeypatch):
    """`-m ""` is filtered out — falls back to env / config."""
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "x")
    monkeypatch.setenv("SMART_COMMIT_CONTEXT", "fallback")
    cfg = sc.build_config(_ns(context=["", "  "]), tmp_path)
    assert cfg.context == "fallback"


def test_context_default_is_empty(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "x")
    monkeypatch.delenv("SMART_COMMIT_CONTEXT", raising=False)
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.context == ""


def test_context_cli_parsing_via_argparse(sc):
    """End-to-end: -m flag is appended into a list and accepts repeated use."""
    args = sc.parse_args(["-m", "first part", "-m", "second part"])
    assert args.context == ["first part", "second part"]


def test_context_cli_parsing_long_form(sc):
    args = sc.parse_args(["--context", "single chunk"])
    assert args.context == ["single chunk"]


def test_context_line_appears_in_output(tmp_git_repo, sc, monkeypatch, capsys):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    groups = [sc.CommitGroup(message="feat: a", files=["a.py"])]
    _patch_grouping(monkeypatch, sc, groups)

    rc = sc.main(["--auto", "-m", "added license validation"])
    assert rc == sc.EXIT_OK
    out = capsys.readouterr().out
    assert 'Context: "added license validation"' in out


def test_context_passed_through_to_prompt(tmp_git_repo, sc, monkeypatch):
    """Verify the context string actually reaches build_user_message via request_grouping."""
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    captured: dict = {}

    def fake_request_grouping(config, diff, files, recent_log):
        captured["context"] = config.context
        return ([sc.CommitGroup(message="feat: a", files=["a.py"])],
                sc.Usage(input_tokens=1, output_tokens=1, cost_usd=0.0))

    monkeypatch.setattr(sc, "request_grouping", fake_request_grouping)
    rc = sc.main(["--auto", "-m", "intent here"])
    assert rc == sc.EXIT_OK
    assert captured["context"] == "intent here"


def test_print_completion_zsh(sc, capsys):
    rc = sc.main(["--print-completion", "zsh"])
    assert rc == sc.EXIT_OK
    out = capsys.readouterr().out
    # argcomplete's zsh shellcode includes bashcompinit and references the program name
    assert "smart-commit" in out
    assert len(out) > 100  # non-trivial script


def test_print_completion_bash(sc, capsys):
    rc = sc.main(["--print-completion", "bash"])
    assert rc == sc.EXIT_OK
    out = capsys.readouterr().out
    assert "smart-commit" in out


def test_render_plan_uses_model_not_claude(tmp_git_repo, sc, monkeypatch, capsys):
    """The plan header must reflect the active model, not a hardcoded 'Claude'."""
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    groups = [sc.CommitGroup(message="feat: a", files=["a.py"])]
    _patch_grouping(monkeypatch, sc, groups)
    monkeypatch.setenv("SMART_COMMIT_MODEL", "qwen/qwen3.6-flash")

    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_OK
    out = capsys.readouterr().out
    assert "qwen3.6-flash suggests" in out
    assert "Claude suggests" not in out


def test_usage_line_appears_in_output(tmp_git_repo, sc, monkeypatch, capsys):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    groups = [sc.CommitGroup(message="feat: a", files=["a.py"])]
    _patch_grouping(
        monkeypatch,
        sc,
        groups,
        usage=sc.Usage(input_tokens=4200, output_tokens=890, cost_usd=0.003),
    )

    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_OK
    out = capsys.readouterr().out
    assert "(~4.2k in · 890 out · $0.003)" in out


# ----------------------------------------------------------------------
# Hunk-level splitting (--split-hunks / -p)
# ----------------------------------------------------------------------


def _stage_multi_hunk_file(repo, name="app.py", n=40, edits=(3, 36)):
    """Commit a baseline file, then stage edits far enough apart to form
    one hunk per edit position."""
    base = "".join(f"line{i}\n" for i in range(1, n + 1))
    (repo / name).write_text(base)
    subprocess.run(["git", "add", name], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", f"add {name}"], cwd=repo, check=True
    )
    lines = base.splitlines()
    for pos in edits:
        lines[pos - 1] = f"line{pos} EDITED"
    (repo / name).write_text("\n".join(lines) + "\n")
    subprocess.run(["git", "add", name], cwd=repo, check=True)


def _show_file(repo, ref, name):
    return subprocess.run(
        ["git", "show", f"{ref}:{name}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _patch_hunk_grouping(monkeypatch, sc, groups, usage=None):
    if usage is None:
        usage = sc.Usage(input_tokens=1234, output_tokens=567, cost_usd=0.0042)

    def fake(config, index, recent_log):
        return list(groups), usage

    def fake_repair(config, index, recent_log, groups_in, errors):
        return list(groups), usage

    monkeypatch.setattr(sc, "request_hunk_grouping", fake)
    monkeypatch.setattr(sc, "request_hunk_grouping_repair", fake_repair)


# --- diff parsing ---


def test_parse_staged_hunks_multi_file(tmp_git_repo, sc):
    _stage_multi_hunk_file(tmp_git_repo, "app.py")
    stage_files(tmp_git_repo, {"new.py": "print('hi')\n"})

    patches = sc.parse_staged_hunks(sc.git_staged_diff())
    by_path = {p.path: p for p in patches}
    assert set(by_path) == {"app.py", "new.py"}
    app = by_path["app.py"]
    assert len(app.hunks) == 2
    assert app.header_lines[0].startswith("diff --git a/app.py b/app.py")
    assert all(h.header.startswith("@@ -") for h in app.hunks)
    assert app.hunks[0].hunk_id == "app.py#1"
    assert app.hunks[1].hunk_id == "app.py#2"
    assert any("EDITED" in l for l in app.hunks[0].lines)
    # New file: single hunk with the added content.
    new = by_path["new.py"]
    assert len(new.hunks) == 1
    assert any("print('hi')" in l for l in new.hunks[0].lines)


def test_build_hunk_index_binary_and_rename_are_whole_file(tmp_git_repo, sc):
    # Commit the rename baseline first — committing later would sweep up
    # the other staged files.
    (tmp_git_repo / "old.py").write_text("keep\n")
    subprocess.run(["git", "add", "old.py"], cwd=tmp_git_repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "add old"], cwd=tmp_git_repo, check=True
    )
    _stage_multi_hunk_file(tmp_git_repo, "app.py")
    stage_binary_file(tmp_git_repo, "icon.png")
    subprocess.run(["git", "mv", "old.py", "renamed.py"], cwd=tmp_git_repo, check=True)

    staged, renames = sc.git_staged_status()
    index = sc.build_hunk_index(staged, renames, sc.git_staged_diff())

    assert "app.py#1" in index.units and "app.py#2" in index.units
    assert "icon.png" in index.units  # bare unit, no hunks
    assert "renamed.py" in index.units
    assert "icon.png" not in index.hunks
    assert "renamed.py" not in index.hunks
    assert set(index.headers) == {"app.py"}


def test_hunk_patch_roundtrip_applies(tmp_git_repo, sc):
    _stage_multi_hunk_file(tmp_git_repo, "app.py", edits=(3, 36))
    staged, renames = sc.git_staged_status()
    index = sc.build_hunk_index(staged, renames, sc.git_staged_diff())
    assert index.units == ["app.py#1", "app.py#2"]

    sc.git_reset_index()
    patch = sc.build_file_patch(index.headers["app.py"], [index.hunks["app.py#1"]])
    sc.git_apply_cached(patch)

    staged_diff = sc.git_staged_diff()
    assert "line3 EDITED" in staged_diff
    assert "line36 EDITED" not in staged_diff


# --- the core new behavior: one file split across two commits ---


def test_split_file_across_two_commits(tmp_git_repo, sc, monkeypatch):
    _stage_multi_hunk_file(tmp_git_repo, "app.py", edits=(3, 36))
    groups = [
        sc.CommitGroup(
            message="feat: change a", files=["app.py"], hunks=["app.py#1"]
        ),
        sc.CommitGroup(
            message="fix: change b", files=["app.py"], hunks=["app.py#2"]
        ),
    ]
    _patch_hunk_grouping(monkeypatch, sc, groups)

    rc = sc.main(["--split-hunks", "--auto"])
    assert rc == sc.EXIT_OK

    subjects = commit_log_subjects(tmp_git_repo)
    assert subjects[:2] == ["fix: change b", "feat: change a"]
    first = _show_file(tmp_git_repo, "HEAD^", "app.py")
    assert "line3 EDITED" in first
    assert "line36 EDITED" not in first
    final = _show_file(tmp_git_repo, "HEAD", "app.py")
    assert "line3 EDITED" in final and "line36 EDITED" in final
    assert staged_files(tmp_git_repo) == []


def test_split_survives_offset_drift(tmp_git_repo, sc, monkeypatch):
    """Committing hunk 1 shifts hunk 2's line offsets; git apply must still
    locate it by context when commit 2 is built."""
    base = "".join(f"line{i}\n" for i in range(1, 41))
    (tmp_git_repo / "app.py").write_text(base)
    subprocess.run(["git", "add", "app.py"], cwd=tmp_git_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=tmp_git_repo, check=True)
    lines = base.splitlines()
    lines[3:3] = [f"inserted{j}" for j in range(5)]  # hunk 1 grows the file
    lines[lines.index("line36")] = "line36 EDITED"  # hunk 2, offsets now stale
    (tmp_git_repo / "app.py").write_text("\n".join(lines) + "\n")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_git_repo, check=True)

    groups = [
        sc.CommitGroup(message="feat: insert", files=["app.py"], hunks=["app.py#1"]),
        sc.CommitGroup(message="fix: edit", files=["app.py"], hunks=["app.py#2"]),
    ]
    _patch_hunk_grouping(monkeypatch, sc, groups)

    rc = sc.main(["-p", "--auto"])
    assert rc == sc.EXIT_OK
    first = _show_file(tmp_git_repo, "HEAD^", "app.py")
    assert "inserted0" in first and "EDITED" not in first
    head = _show_file(tmp_git_repo, "HEAD", "app.py")
    assert "inserted0" in head and "line36 EDITED" in head
    assert staged_files(tmp_git_repo) == []


def test_split_hunks_mixed_with_whole_file_units(tmp_git_repo, sc, monkeypatch):
    _stage_multi_hunk_file(tmp_git_repo, "app.py", edits=(3, 36))
    stage_binary_file(tmp_git_repo, "icon.png")
    groups = [
        sc.CommitGroup(
            message="feat: change a",
            files=["app.py", "icon.png"],
            hunks=["app.py#1", "icon.png"],
        ),
        sc.CommitGroup(
            message="fix: change b", files=["app.py"], hunks=["app.py#2"]
        ),
    ]
    _patch_hunk_grouping(monkeypatch, sc, groups)

    rc = sc.main(["-p", "--auto"])
    assert rc == sc.EXIT_OK
    first_files = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD^"],
        cwd=tmp_git_repo, check=True, capture_output=True, text=True,
    ).stdout.split()
    assert sorted(first_files) == ["app.py", "icon.png"]
    assert staged_files(tmp_git_repo) == []


# --- fallback when hunks don't apply ---


def test_hunk_fallback_stages_whole_file_in_current_group(
    tmp_git_repo, sc, monkeypatch, capsys
):
    _stage_multi_hunk_file(tmp_git_repo, "app.py", edits=(3, 36))
    stage_files(tmp_git_repo, {"b.py": "x = 1\n"})
    groups = [
        sc.CommitGroup(
            message="feat: a", files=["app.py", "b.py"], hunks=["app.py#1", "b.py#1"]
        ),
        sc.CommitGroup(message="fix: b", files=["app.py"], hunks=["app.py#2"]),
    ]
    _patch_hunk_grouping(monkeypatch, sc, groups)

    orig_apply = sc.git_apply_cached

    def flaky(patch):
        if "a/app.py" in patch:
            raise sc.GitError("git apply --cached: patch failed")
        return orig_apply(patch)

    monkeypatch.setattr(sc, "git_apply_cached", flaky)

    rc = sc.main(["-p", "--auto"])
    assert rc == sc.EXIT_OK  # commit 2 was consolidated away, not an error

    # Tie between groups 1 and 2 -> earliest wins: whole app.py in commit 1.
    subjects = commit_log_subjects(tmp_git_repo)
    assert subjects[0] == "feat: a"
    assert "fix: b" not in subjects
    head = _show_file(tmp_git_repo, "HEAD", "app.py")
    assert "line3 EDITED" in head and "line36 EDITED" in head
    assert staged_files(tmp_git_repo) == []
    err = capsys.readouterr().err
    assert "did not apply cleanly" in err


def test_hunk_fallback_defers_to_majority_group(tmp_git_repo, sc, monkeypatch, capsys):
    _stage_multi_hunk_file(tmp_git_repo, "app.py", n=90, edits=(3, 45, 87))
    stage_files(tmp_git_repo, {"c.py": "y = 2\n"})
    groups = [
        sc.CommitGroup(
            message="feat: a", files=["app.py", "c.py"], hunks=["app.py#1", "c.py#1"]
        ),
        sc.CommitGroup(
            message="fix: b", files=["app.py"], hunks=["app.py#2", "app.py#3"]
        ),
    ]
    _patch_hunk_grouping(monkeypatch, sc, groups)

    orig_apply = sc.git_apply_cached

    def flaky(patch):
        if "a/app.py" in patch:
            raise sc.GitError("git apply --cached: patch failed")
        return orig_apply(patch)

    monkeypatch.setattr(sc, "git_apply_cached", flaky)

    rc = sc.main(["-p", "--auto"])
    assert rc == sc.EXIT_OK

    subjects = commit_log_subjects(tmp_git_repo)
    assert subjects[:2] == ["fix: b", "feat: a"]
    # Majority of app.py's hunks live in group 2 -> whole file deferred there.
    first = _show_file(tmp_git_repo, "HEAD^", "app.py")
    assert "EDITED" not in first
    head = _show_file(tmp_git_repo, "HEAD", "app.py")
    assert head.count("EDITED") == 3
    assert staged_files(tmp_git_repo) == []
    err = capsys.readouterr().err
    assert "staging the whole file in commit 2 instead" in err


# --- flag interactions ---


def test_split_hunks_dry_run_makes_no_commits(tmp_git_repo, sc, monkeypatch, capsys):
    _stage_multi_hunk_file(tmp_git_repo, "app.py")
    groups = [
        sc.CommitGroup(message="feat: a", files=["app.py"], hunks=["app.py#1"]),
        sc.CommitGroup(message="fix: b", files=["app.py"], hunks=["app.py#2"]),
    ]
    _patch_hunk_grouping(monkeypatch, sc, groups)
    initial = len(commit_log_subjects(tmp_git_repo))

    rc = sc.main(["-p", "--dry-run"])
    assert rc == sc.EXIT_OK
    assert len(commit_log_subjects(tmp_git_repo)) == initial
    assert staged_files(tmp_git_repo) == ["app.py"]
    out = capsys.readouterr().out
    assert "app.py#1" in out and "app.py#2" in out  # plan shows hunk IDs
    assert "(2 hunks)" in out


def test_split_hunks_auto_skips_prompt(tmp_git_repo, sc, monkeypatch):
    _stage_multi_hunk_file(tmp_git_repo, "app.py")
    groups = [
        sc.CommitGroup(
            message="feat: a", files=["app.py"], hunks=["app.py#1", "app.py#2"]
        ),
    ]
    _patch_hunk_grouping(monkeypatch, sc, groups)

    def no_input(*_):
        raise AssertionError("should not be prompted")

    monkeypatch.setattr("builtins.input", no_input)
    rc = sc.main(["-p", "--auto"])
    assert rc == sc.EXIT_OK
    assert "feat: a" in commit_log_subjects(tmp_git_repo)


def test_split_hunks_verbose_writes_body(tmp_git_repo, sc, monkeypatch):
    _stage_multi_hunk_file(tmp_git_repo, "app.py")
    groups = [
        sc.CommitGroup(
            message="feat: a",
            files=["app.py"],
            hunks=["app.py#1", "app.py#2"],
            body="Both edits belong to the same feature.",
        ),
    ]
    _patch_hunk_grouping(monkeypatch, sc, groups)

    rc = sc.main(["-p", "--verbose", "--auto"])
    assert rc == sc.EXIT_OK
    body = commit_body(tmp_git_repo)
    assert body.startswith("feat: a\n\n")
    assert "same feature" in body


def test_no_flag_keeps_file_level_path(tmp_git_repo, sc, monkeypatch):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    groups = [sc.CommitGroup(message="feat: a", files=["a.py"])]
    _patch_grouping(monkeypatch, sc, groups)

    def boom(*_a, **_kw):
        raise AssertionError("hunk path must not run without --split-hunks")

    monkeypatch.setattr(sc, "request_hunk_grouping", boom)
    monkeypatch.setattr(sc, "build_hunk_index", boom)
    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_OK
    assert "feat: a" in commit_log_subjects(tmp_git_repo)


def test_split_hunks_validation_failure_aborts(tmp_git_repo, sc, monkeypatch):
    _stage_multi_hunk_file(tmp_git_repo, "app.py")
    # app.py#2 unassigned -> validation error.
    groups = [
        sc.CommitGroup(message="feat: a", files=["app.py"], hunks=["app.py#1"]),
    ]
    _patch_hunk_grouping(monkeypatch, sc, groups)
    initial = len(commit_log_subjects(tmp_git_repo))

    rc = sc.main(["-p", "--auto"])
    assert rc == sc.EXIT_VALIDATION_ERROR
    assert len(commit_log_subjects(tmp_git_repo)) == initial
    assert staged_files(tmp_git_repo) == ["app.py"]


# --- validation ---


def test_validate_hunk_groups_happy(sc):
    groups = [
        sc.CommitGroup(message="feat: x", files=[], hunks=["a.py#1", "b.py"]),
        sc.CommitGroup(message="chore: y", files=[], hunks=["a.py#2"]),
    ]
    assert sc.validate_hunk_groups(groups, ["a.py#1", "a.py#2", "b.py"]) == []


def test_validate_hunk_groups_duplicate(sc):
    groups = [
        sc.CommitGroup(message="feat: x", files=[], hunks=["a.py#1"]),
        sc.CommitGroup(message="feat: y", files=[], hunks=["a.py#1"]),
    ]
    errors = sc.validate_hunk_groups(groups, ["a.py#1"])
    assert any("appears in commits 1 and 2" in e for e in errors)


def test_validate_hunk_groups_unknown_and_unassigned(sc):
    groups = [sc.CommitGroup(message="feat: x", files=[], hunks=["ghost.py#9"])]
    errors = sc.validate_hunk_groups(groups, ["a.py#1"])
    assert any("'ghost.py#9' is not in the staged hunk list" in e for e in errors)
    assert any("Hunks not in any commit: a.py#1" in e for e in errors)


def test_validate_hunk_groups_no_hunks(sc):
    groups = [sc.CommitGroup(message="feat: x", files=["a.py"], hunks=[])]
    errors = sc.validate_hunk_groups(groups, [])
    assert any("no hunks" in e for e in errors)


# --- prompt + schema ---


def _fake_index(sc):
    h1 = sc.Hunk(
        file="a.py", header="@@ -1,2 +1,3 @@", lines=[" x", "+y"],
        hunk_id="a.py#1", seq=1,
    )
    h2 = sc.Hunk(
        file="a.py", header="@@ -10,2 +11,2 @@", lines=["-old", "+new"],
        hunk_id="a.py#2", seq=2,
    )
    return sc.HunkIndex(
        units=["a.py#1", "a.py#2", "icon.png"],
        hunks={"a.py#1": h1, "a.py#2": h2},
        headers={"a.py": ["diff --git a/a.py b/a.py", "--- a/a.py", "+++ b/a.py"]},
    )


def test_build_hunk_user_message_contents(sc):
    index = _fake_index(sc)
    msg = sc.build_hunk_user_message(index, recent_log="", conventions="", context="")
    assert "## Staged hunks (reference by numeric ID)" in msg
    assert "1. a.py#1\n@@ -1,2 +1,3 @@" in msg
    assert "+y" in msg
    assert "3. icon.png (whole file" in msg


def test_hunk_schema_uses_integer_hunks(sc):
    items = sc.RESPONSE_SCHEMA_HUNKS["properties"]["commits"]["items"]
    assert items["properties"]["hunks"]["items"] == {"type": "integer"}
    assert "hunks" in items["required"]


def test_hunk_system_prompt_allows_same_file_split(sc):
    prompt = sc.build_hunk_system_prompt(verbose=False)
    assert "SAME file CAN go into different commits" in prompt
    assert '"hunks"' in prompt


def test_format_hunks_for_prompt_truncates_when_oversized(sc):
    big = sc.Hunk(
        file="big.py", header="@@ -1,500 +1,500 @@",
        lines=[f"+x{i}" for i in range(500)], hunk_id="big.py#1", seq=1,
    )
    index = sc.HunkIndex(
        units=["big.py#1"], hunks={"big.py#1": big},
        headers={"big.py": ["diff --git a/big.py b/big.py"]},
    )
    out = sc.format_hunks_for_prompt(index, max_bytes=100, head_tail=10)
    assert "lines truncated" in out
    assert out.startswith("1. big.py#1")


def test_request_completion_resolves_hunk_ids(sc, monkeypatch):
    calls = {"payloads": []}
    _make_fake_client(sc, monkeypatch, [
        '{"commits":[{"message":"feat: x","hunks":[2,1],"body":"","reasoning":"r"}]}',
    ], calls)
    cfg = sc.Config(model="m", api_key="k", base_url=sc.DEFAULT_API_BASE)
    items, _usage = sc._request_completion(
        cfg, "system", "user", ["a.py#1", "a.py#2"], ref_key="hunks"
    )
    assert items[0]["hunks"] == ["a.py#2", "a.py#1"]
    schema = calls["payloads"][0]["response_format"]["json_schema"]["schema"]
    assert schema == sc.RESPONSE_SCHEMA_HUNKS


def test_request_completion_retries_on_bad_hunk_ref(sc, monkeypatch):
    calls = {"payloads": []}
    _make_fake_client(sc, monkeypatch, [
        '{"commits":[{"message":"x","hunks":[5],"body":"","reasoning":"r"}]}',
        '{"commits":[{"message":"x","hunks":[1],"body":"","reasoning":"r"}]}',
    ], calls)
    cfg = sc.Config(model="m", api_key="k", base_url=sc.DEFAULT_API_BASE)
    items, _usage = sc._request_completion(
        cfg, "system", "user", ["a.py#1"], ref_key="hunks"
    )
    assert len(calls["payloads"]) == 2
    assert items[0]["hunks"] == ["a.py#1"]
    retry_msg = calls["payloads"][1]["messages"][-1]["content"]
    assert "numeric ID" in retry_msg and "hunk" in retry_msg


# --- TOML edit mode ---


def test_groups_to_toml_hunks_roundtrip(sc):
    index = _fake_index(sc)
    groups = [
        sc.CommitGroup(
            message="feat: x", files=["a.py"], hunks=["a.py#1", "a.py#2"],
            body="", reasoning="r",
        ),
        sc.CommitGroup(
            message="chore: icon", files=["icon.png"], hunks=["icon.png"],
            body="", reasoning="r2",
        ),
    ]
    text = sc.groups_to_toml(groups, index)
    # Catalog of available hunks is in the header comments.
    assert "#   a.py#1  @@ -1,2 +1,3 @@" in text
    assert "#   icon.png (whole file)" in text
    parsed = sc.parse_toml_plan(text)
    assert parsed[0].hunks == ["a.py#1", "a.py#2"]
    assert parsed[1].hunks == ["icon.png"]
    assert sc.validate_hunk_groups(parsed, index.units) == []


def test_parse_toml_plan_requires_files_or_hunks(sc):
    with pytest.raises(ValueError):
        sc.parse_toml_plan('[[commit]]\nmessage = "x"\n')
    parsed = sc.parse_toml_plan('[[commit]]\nmessage = "x"\nhunks = ["a.py#1"]\n')
    assert parsed[0].hunks == ["a.py#1"]
    assert parsed[0].files == []


def test_derive_group_files_from_hunks(sc):
    index = _fake_index(sc)
    groups = [
        sc.CommitGroup(
            message="m", files=[], hunks=["a.py#2", "icon.png", "a.py#1"]
        )
    ]
    sc.derive_group_files(groups, index)
    assert groups[0].files == ["a.py", "icon.png"]


# --- config ---


def test_split_hunks_cli_flag_parses(sc):
    assert sc.parse_args(["-p"]).split_hunks is True
    assert sc.parse_args(["--split-hunks"]).split_hunks is True
    assert sc.parse_args([]).split_hunks is False


def test_split_hunks_config_key(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "x")
    (tmp_path / sc.CONFIG_FILENAME).write_text("split_hunks = true\n")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.split_hunks is True


def test_split_hunks_defaults_off(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_API_KEY", "x")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.split_hunks is False


# --- plan repair ---


def test_sum_usage_adds_tokens_and_cost(sc):
    a = sc.Usage(input_tokens=100, output_tokens=10, cost_usd=0.001)
    b = sc.Usage(input_tokens=250, output_tokens=25, cost_usd=0.002)
    total = sc._sum_usage(a, b)
    assert total.input_tokens == 350
    assert total.output_tokens == 35
    assert total.cost_usd == pytest.approx(0.003)
    assert total.estimated is False


def test_sum_usage_unknown_cost_stays_unknown(sc):
    a = sc.Usage(input_tokens=1, output_tokens=1, cost_usd=None)
    b = sc.Usage(input_tokens=1, output_tokens=1, cost_usd=0.5)
    assert sc._sum_usage(a, b).cost_usd is None


def test_sum_usage_propagates_estimated_flag(sc):
    a = sc.Usage(input_tokens=1, output_tokens=1, cost_usd=0.1, estimated=True)
    b = sc.Usage(input_tokens=1, output_tokens=1, cost_usd=0.1, estimated=False)
    assert sc._sum_usage(a, b).estimated is True


def test_plan_as_json_uses_numeric_ids(sc):
    groups = [
        sc.CommitGroup(message="feat: a", files=["a.py", "c.py"], reasoning="r1"),
        sc.CommitGroup(message="chore: b", files=["b.py"], body="why", reasoning="r2"),
    ]
    import json as _json

    plan = _json.loads(sc._plan_as_json(groups, ["a.py", "b.py", "c.py"], "files"))
    assert plan["commits"][0]["files"] == [1, 3]
    assert plan["commits"][1]["files"] == [2]
    assert plan["commits"][1]["body"] == "why"
    assert plan["commits"][0]["message"] == "feat: a"


def test_plan_as_json_drops_unresolvable_refs(sc):
    groups = [sc.CommitGroup(message="feat: a", files=["a.py", "ghost.py"])]
    import json as _json

    plan = _json.loads(sc._plan_as_json(groups, ["a.py"], "files"))
    assert plan["commits"][0]["files"] == [1]


def test_repair_user_message_lists_errors(sc):
    msg = sc._repair_user_message(["Files not in any commit: build/x.jar"], "files")
    assert "build/x.jar" in msg
    assert "EXACTLY ONE" in msg
    assert "file ID" in msg


def test_request_completion_correction_seeds_conversation(sc, monkeypatch):
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"commits":[{"message":"feat: x","files":[1],"body":"","reasoning":"r"}]}'
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers, json):
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(sc.httpx, "Client", FakeClient)
    cfg = sc.Config(model="m", api_key="k", base_url=sc.DEFAULT_API_BASE)
    sc._request_completion(
        cfg,
        "system",
        "user",
        ["a.py"],
        correction=('{"commits":[]}', "you dropped a.py"),
    )
    roles = [m["role"] for m in captured["payload"]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert captured["payload"]["messages"][2]["content"] == '{"commits":[]}'
    assert captured["payload"]["messages"][3]["content"] == "you dropped a.py"


def test_repair_round_fixes_dropped_file_and_commits(tmp_git_repo, sc, monkeypatch, capsys):
    """The exact failure mode this was built for: the model omits a staged
    artifact deletion, the repair round assigns it, and the run proceeds."""
    artifact = tmp_git_repo / "build" / "junk.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"\x00generated")
    subprocess.run(
        ["git", "add", "build/junk.bin"], cwd=tmp_git_repo, check=True
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "track generated artifact"],
        cwd=tmp_git_repo,
        check=True,
    )
    artifact.unlink()
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    subprocess.run(
        ["git", "add", "-u", "build/junk.bin"], cwd=tmp_git_repo, check=True
    )
    usage = sc.Usage(input_tokens=100, output_tokens=10, cost_usd=0.001)
    calls: dict = {}

    def fake(config, diff, files, recent_log):
        return [sc.CommitGroup(message="feat: a", files=["a.py"])], usage

    def fake_repair(config, diff, files, recent_log, groups_in, errors):
        calls["errors"] = errors
        calls["groups_in"] = [g.message for g in groups_in]
        return (
            [
                sc.CommitGroup(message="feat: a", files=["a.py"]),
                sc.CommitGroup(
                    message="chore: stop tracking build artifact",
                    files=["build/junk.bin"],
                ),
            ],
            usage,
        )

    monkeypatch.setattr(sc, "request_grouping", fake)
    monkeypatch.setattr(sc, "request_grouping_repair", fake_repair)

    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_OK
    assert any("build/junk.bin" in e for e in calls["errors"])
    assert calls["groups_in"] == ["feat: a"]
    subjects = commit_log_subjects(tmp_git_repo)
    assert "feat: a" in subjects
    assert "chore: stop tracking build artifact" in subjects
    assert staged_files(tmp_git_repo) == []


def test_repair_round_usage_is_added_to_total(tmp_git_repo, sc, monkeypatch, capsys):
    stage_files(tmp_git_repo, {"a.py": "1\n", "b.py": "2\n"})

    def fake(config, diff, files, recent_log):
        return (
            [sc.CommitGroup(message="feat: a", files=["a.py"])],
            sc.Usage(input_tokens=100, output_tokens=10, cost_usd=0.001),
        )

    def fake_repair(config, diff, files, recent_log, groups_in, errors):
        return (
            [sc.CommitGroup(message="feat: a", files=["a.py", "b.py"])],
            sc.Usage(input_tokens=250, output_tokens=25, cost_usd=0.002),
        )

    monkeypatch.setattr(sc, "request_grouping", fake)
    monkeypatch.setattr(sc, "request_grouping_repair", fake_repair)

    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_OK
    out = capsys.readouterr().out
    expected = sc.format_usage_line(
        sc.Usage(input_tokens=350, output_tokens=35, cost_usd=0.003)
    )
    assert expected in out


def test_repair_api_failure_falls_back_to_validation_error(tmp_git_repo, sc, monkeypatch, capsys):
    stage_files(tmp_git_repo, {"a.py": "1\n", "b.py": "2\n"})

    def fake(config, diff, files, recent_log):
        return (
            [sc.CommitGroup(message="feat: a", files=["a.py"])],
            sc.Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
        )

    def boom(config, diff, files, recent_log, groups_in, errors):
        raise sc.APICallError("upstream 500")

    monkeypatch.setattr(sc, "request_grouping", fake)
    monkeypatch.setattr(sc, "request_grouping_repair", boom)

    initial = len(commit_log_subjects(tmp_git_repo))
    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_VALIDATION_ERROR
    assert len(commit_log_subjects(tmp_git_repo)) == initial
    assert "repair attempt failed" in capsys.readouterr().err


def test_repair_round_still_invalid_keeps_original_plan(tmp_git_repo, sc, monkeypatch, capsys):
    stage_files(tmp_git_repo, {"a.py": "1\n", "b.py": "2\n"})

    def fake(config, diff, files, recent_log):
        return (
            [sc.CommitGroup(message="feat: original", files=["a.py"])],
            sc.Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
        )

    def fake_repair(config, diff, files, recent_log, groups_in, errors):
        # Still drops b.py.
        return (
            [sc.CommitGroup(message="feat: repaired", files=["a.py"])],
            sc.Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
        )

    monkeypatch.setattr(sc, "request_grouping", fake)
    monkeypatch.setattr(sc, "request_grouping_repair", fake_repair)

    rc = sc.main(["--auto"])
    assert rc == sc.EXIT_VALIDATION_ERROR
    assert "still invalid" in capsys.readouterr().err
    assert sorted(staged_files(tmp_git_repo)) == ["a.py", "b.py"]


def test_repair_round_runs_in_hunk_mode(tmp_git_repo, sc, monkeypatch):
    _stage_multi_hunk_file(tmp_git_repo, "app.py")
    usage = sc.Usage(input_tokens=1, output_tokens=1, cost_usd=0.0)

    def fake(config, index, recent_log):
        # app.py#2 unassigned -> validation error.
        return [sc.CommitGroup(message="feat: a", files=["app.py"], hunks=["app.py#1"])], usage

    def fake_repair(config, index, recent_log, groups_in, errors):
        groups = [
            sc.CommitGroup(message="feat: a", files=["app.py"], hunks=["app.py#1"]),
            sc.CommitGroup(message="fix: b", files=["app.py"], hunks=["app.py#2"]),
        ]
        return groups, usage

    monkeypatch.setattr(sc, "request_hunk_grouping", fake)
    monkeypatch.setattr(sc, "request_hunk_grouping_repair", fake_repair)

    rc = sc.main(["-p", "--auto"])
    assert rc == sc.EXIT_OK
    subjects = commit_log_subjects(tmp_git_repo)
    assert "feat: a" in subjects
    assert "fix: b" in subjects
