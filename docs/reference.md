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
