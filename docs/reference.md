# Reference

## Model plan validation

- `_request_completion` retries malformed JSON or invalid numeric references once.
- `main` separately retries a schema-valid plan once when coverage validation finds
  missing, duplicate, unknown, or empty commit assignments.
- Repair usage is added to the displayed token and cost totals.
- File and hunk modes both use numeric IDs in corrective prompts.
- OpenRouter's `qwen/qwen3.7-flash` uses JSON-object mode with reasoning disabled;
  other models continue to receive the strict JSON Schema request.
- Null or blank `message.content` is retried once. Persistent empty responses become
  `APICallError` messages that include available finish and reasoning-token details.

## Installing / packaging

- `build/lib/smart_commit.py` is a stale copy left by setuptools. `build_py` copies
  only when the source mtime is newer than the destination, so a `build/lib` copy with
  a newer mtime than `smart_commit.py` silently poisons every wheel — `uv tool install`
  (even `--reinstall`) then ships old code with a fresh install timestamp.
- `build/` is gitignored, so a clean `git status` proves nothing about what gets built.
- Always `rm -rf build smart_commit.egg-info __pycache__` before `make reinstall`.
- Verify the install by diffing the artifact, not by launching it:
  `diff -q ~/.local/share/uv/tools/smart-commit/lib/python3.12/site-packages/smart_commit.py smart_commit.py`
