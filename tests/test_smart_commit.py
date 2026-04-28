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


def _patch_grouping(monkeypatch, sc, groups, usage=None):
    if usage is None:
        usage = sc.Usage(input_tokens=1234, output_tokens=567, cost_usd=0.0042)

    def fake(config, diff, files, recent_log):
        return list(groups), usage

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
    defaults = dict(
        auto=False,
        dry_run=False,
        verbose=None,
        provider=None,
        model=None,
        context=None,
    )
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
        model="anthropic/claude-sonnet-4.6",
        api_key="or-test",
        base_url=sc.OPENROUTER_BASE_URL,
    )
    items, usage = sc._request_openrouter(cfg, "system text", "user text")
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
        provider=sc.PROVIDER_OPENROUTER,
        model="some/model",
        api_key="or-test",
        base_url=sc.OPENROUTER_BASE_URL,
    )
    items, _usage = sc._request_openrouter(cfg, "system", "user")
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
    items, _usage = sc._request_openrouter(cfg, "system", "user")
    assert calls["n"] == 2
    assert items[0]["files"] == ["b.py"]


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
        provider=sc.PROVIDER_OPENROUTER,
        model="some/model",
        api_key="or-test",
        base_url=sc.OPENROUTER_BASE_URL,
    )
    _items, usage = sc._request_openrouter(cfg, "system", "user")
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
        provider=sc.PROVIDER_OPENROUTER,
        model="anthropic/claude-sonnet-4.6",
        api_key="or-test",
        base_url=sc.OPENROUTER_BASE_URL,
    )
    _items, usage = sc._request_openrouter(cfg, "system", "user")
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
        provider=sc.PROVIDER_OPENROUTER,
        # OpenRouter uses dot-separated versions; our local table uses dashes.
        # The fallback should normalize the ID.
        model="anthropic/claude-sonnet-4.6",
        api_key="or-test",
        base_url=sc.OPENROUTER_BASE_URL,
    )
    _items, usage = sc._request_openrouter(cfg, "system", "user")
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
        provider=sc.PROVIDER_OPENROUTER,
        model="qwen/qwen3-coder",
        api_key="or-test",
        base_url=sc.OPENROUTER_BASE_URL,
    )
    _items, usage = sc._request_openrouter(cfg, "system", "user")
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
        provider=sc.PROVIDER_OPENROUTER,
        model="x",
        api_key="k",
        base_url=sc.OPENROUTER_BASE_URL,
    )
    _items, usage = sc._request_openrouter(cfg, "s", "u")
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
        provider=sc.PROVIDER_OPENROUTER,
        model="some/model",
        api_key="bad",
        base_url=sc.OPENROUTER_BASE_URL,
    )
    with pytest.raises(sc.APICallError):
        sc._request_openrouter(cfg, "system", "user")


# ----------------------------------------------------------------------
# Model aliases + --model flag
# ----------------------------------------------------------------------


def test_resolve_model_aliases_anthropic(sc):
    assert sc.resolve_model("haiku", sc.PROVIDER_ANTHROPIC) == "claude-haiku-4-5"
    assert sc.resolve_model("sonnet", sc.PROVIDER_ANTHROPIC) == "claude-sonnet-4-6"
    assert sc.resolve_model("opus", sc.PROVIDER_ANTHROPIC) == "claude-opus-4-7"


def test_resolve_model_aliases_openrouter(sc):
    assert sc.resolve_model("haiku", sc.PROVIDER_OPENROUTER) == "anthropic/claude-haiku-4.5"
    assert sc.resolve_model("sonnet", sc.PROVIDER_OPENROUTER) == "anthropic/claude-sonnet-4.6"
    assert sc.resolve_model("opus", sc.PROVIDER_OPENROUTER) == "anthropic/claude-opus-4.7"


def test_resolve_model_passes_unknown_through(sc):
    assert sc.resolve_model("openai/gpt-5", sc.PROVIDER_OPENROUTER) == "openai/gpt-5"
    assert sc.resolve_model("claude-haiku-4-5", sc.PROVIDER_ANTHROPIC) == "claude-haiku-4-5"


def test_resolve_model_alias_case_insensitive(sc):
    assert sc.resolve_model("HAIKU", sc.PROVIDER_ANTHROPIC) == "claude-haiku-4-5"
    assert sc.resolve_model(" Sonnet ", sc.PROVIDER_ANTHROPIC) == "claude-sonnet-4-6"


def test_model_cli_flag_wins_over_env(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("SMART_COMMIT_MODEL", "claude-opus-4-7")
    cfg = sc.build_config(_ns(model="haiku"), tmp_path)
    assert cfg.model == "claude-haiku-4-5"


def test_model_env_wins_over_config(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    (tmp_path / sc.CONFIG_FILENAME).write_text('model = "sonnet"\n')
    monkeypatch.setenv("SMART_COMMIT_MODEL", "haiku")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.model == "claude-haiku-4-5"


def test_model_alias_resolves_against_active_provider(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("SMART_COMMIT_PROVIDER", "openrouter")
    cfg = sc.build_config(_ns(model="opus"), tmp_path)
    assert cfg.model == "anthropic/claude-opus-4.7"


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


def test_model_completer_anthropic(sc, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SMART_COMMIT_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    ns = argparse.Namespace(provider=None)
    suggestions = sc._model_completer("", ns)
    assert "haiku" in suggestions
    assert "sonnet" in suggestions
    assert "opus" in suggestions
    assert "claude-sonnet-4-6" in suggestions
    # Aliases come first so tab cycles them before full IDs
    assert suggestions.index("haiku") < suggestions.index("claude-sonnet-4-6")


def test_model_completer_openrouter_autodetect(sc, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SMART_COMMIT_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    ns = argparse.Namespace(provider=None)
    suggestions = sc._model_completer("", ns)
    assert "openai/gpt-5" in suggestions
    assert "anthropic/claude-sonnet-4.6" in suggestions
    assert "qwen/qwen3-coder" in suggestions


def test_model_completer_explicit_provider_wins(sc, monkeypatch):
    """When --provider is on the command line, it overrides env-based auto-detect."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    ns = argparse.Namespace(provider="openrouter")
    suggestions = sc._model_completer("", ns)
    assert "openai/gpt-5" in suggestions
    assert "claude-sonnet-4-6" not in suggestions


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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("SMART_COMMIT_CONTEXT", "from env")
    cfg = sc.build_config(_ns(context=["from cli"]), tmp_path)
    assert cfg.context == "from cli"


def test_context_env_used_when_no_cli(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("SMART_COMMIT_CONTEXT", "from env")
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.context == "from env"


def test_context_config_baseline_concatenated_with_per_run(sc, tmp_path, monkeypatch):
    """Config provides a persistent baseline that gets joined with CLI/env per-run."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    (tmp_path / sc.CONFIG_FILENAME).write_text('context = "v2 migration branch"\n')
    cfg = sc.build_config(_ns(context=["new license endpoint"]), tmp_path)
    assert cfg.context == "v2 migration branch new license endpoint"


def test_context_config_only(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    (tmp_path / sc.CONFIG_FILENAME).write_text('context = "long-running auth rewrite"\n')
    cfg = sc.build_config(_ns(), tmp_path)
    assert cfg.context == "long-running auth rewrite"


def test_context_multiple_cli_flags_concatenate(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = sc.build_config(_ns(context=["new license endpoint", "windows path fix"]), tmp_path)
    assert cfg.context == "new license endpoint windows path fix"


def test_context_empty_cli_flag_treated_as_no_context(sc, tmp_path, monkeypatch):
    """`-m ""` is filtered out — falls back to env / config."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("SMART_COMMIT_CONTEXT", "fallback")
    cfg = sc.build_config(_ns(context=["", "  "]), tmp_path)
    assert cfg.context == "fallback"


def test_context_default_is_empty(sc, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
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
