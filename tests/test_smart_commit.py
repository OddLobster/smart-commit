"""Unit + integration tests for smart-commit."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from conftest import commit_body, commit_log_subjects, stage_files, staged_files


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


def _patch_grouping(monkeypatch, sc, groups):
    def fake(config, diff, files, recent_log):
        return list(groups)

    monkeypatch.setattr(sc, "request_grouping", fake)


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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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
    defaults = dict(auto=False, dry_run=False, verbose=None, provider=None)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_provider_defaults_to_anthropic(sc, tmp_path, monkeypatch):
    monkeypatch.delenv("SMART_COMMIT_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.provider == sc.PROVIDER_ANTHROPIC
    assert cfg.model == sc.DEFAULT_MODEL_BY_PROVIDER[sc.PROVIDER_ANTHROPIC]
    assert cfg.api_key == "test"


def test_provider_autodetect_openrouter(sc, tmp_path, monkeypatch):
    monkeypatch.delenv("SMART_COMMIT_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.provider == sc.PROVIDER_OPENROUTER
    assert cfg.model == sc.DEFAULT_MODEL_BY_PROVIDER[sc.PROVIDER_OPENROUTER]
    assert cfg.base_url == sc.OPENROUTER_BASE_URL
    assert cfg.api_key == "or-test"


def test_provider_env_var_wins(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("SMART_COMMIT_PROVIDER", "openrouter")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.provider == sc.PROVIDER_OPENROUTER


def test_provider_cli_flag_wins(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("SMART_COMMIT_PROVIDER", "openrouter")
    cfg = sc.build_config(_ns(provider="anthropic"), tmp_path)
    assert cfg.provider == sc.PROVIDER_ANTHROPIC


def test_provider_invalid_value_errors(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_COMMIT_PROVIDER", "made-up")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    with pytest.raises(ValueError):
        sc.build_config(_ns(), tmp_path)


def test_smart_commit_model_env_overrides_default(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("SMART_COMMIT_MODEL", "claude-haiku-4-5")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.model == "claude-haiku-4-5"


def test_openrouter_missing_key_message(tmp_git_repo, sc, monkeypatch, capsys):
    stage_files(tmp_git_repo, {"a.py": "1\n"})
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("SMART_COMMIT_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    rc = sc.main([])
    assert rc == sc.EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "OPENROUTER_API_KEY" in err


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
        provider=sc.PROVIDER_OPENROUTER,
        model="anthropic/claude-sonnet-4.5",
        api_key="or-test",
        base_url=sc.OPENROUTER_BASE_URL,
    )
    items = sc._request_openrouter(cfg, "system text", "user text")
    assert items == [{"message": "feat: x", "files": ["a.py"], "body": "", "reasoning": "r"}]
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer or-test"
    assert captured["payload"]["model"] == "anthropic/claude-sonnet-4.5"
    assert captured["payload"]["response_format"]["type"] == "json_schema"


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
        provider=sc.PROVIDER_OPENROUTER,
        model="some/model",
        api_key="or-test",
        base_url=sc.OPENROUTER_BASE_URL,
    )
    items = sc._request_openrouter(cfg, "system", "user")
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
        provider=sc.PROVIDER_OPENROUTER,
        model="some/model",
        api_key="or-test",
        base_url=sc.OPENROUTER_BASE_URL,
    )
    items = sc._request_openrouter(cfg, "system", "user")
    assert calls["n"] == 2
    assert items[0]["files"] == ["b.py"]


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
        provider=sc.PROVIDER_OPENROUTER,
        model="some/model",
        api_key="bad",
        base_url=sc.OPENROUTER_BASE_URL,
    )
    with pytest.raises(sc.APICallError):
        sc._request_openrouter(cfg, "system", "user")
