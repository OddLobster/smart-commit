# Handoff

**Date:** 2026-08-15
**Branch:** `main`

## What was done

- Added a one-shot corrective model request when a parsed commit plan fails
  file/hunk coverage validation.
- Preserved the existing manual edit and validation-error fallback if repair fails.
- Added usage aggregation and regression coverage, including a staged deletion of a
  tracked binary build artifact.
- Verified all 157 tests pass, `git diff --check` passes, and `smart_commit.py`
  compiles.
- Reinstalled `smart-commit` v0.1.0 from the updated working tree and verified the
  user-level executable starts successfully.
- Added OpenRouter compatibility handling for `qwen/qwen3.7-flash`: JSON-object
  output with reasoning disabled for commit-plan extraction.
- Null or blank assistant content now gets one retry and then a diagnostic
  `APICallError` instead of an `AttributeError` traceback.
- Verified all 160 tests pass, reinstalled the CLI again, and confirmed the
  installed module activates the Qwen 3.7 compatibility path.

## Next steps

- Review and commit the working-tree changes.
- Repeat a real `sc --dry-run` when a repository has staged changes; the attempted
  `microbot-shim` verification found only unstaged and untracked files.
