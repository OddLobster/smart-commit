# Decisions

## 2026-08-15: Retry invalid commit plans once with validation feedback

**Context:** Models can return valid JSON that omits low-signal staged entries,
especially binary files and deletions. The existing retry only handled malformed
JSON and invalid references.

**Decision:** On a plan validation failure, make one corrective request containing
the parsed prior plan and exact validation errors. If the corrected plan validates,
continue; otherwise preserve the original manual-edit/error path.

**Tradeoff:** Invalid plans can cost one additional model request, but valid plans
remain single-request and users retain the existing fallback behavior.

## 2026-08-15: Use compatibility mode for Qwen 3.7 Flash on OpenRouter

**Context:** Qwen 3.7 Flash enables reasoning by default and does not advertise
strict structured-output support. With smart-commit's 4,096-token completion
budget it repeatedly returned `message.content = null` before plan validation.

**Decision:** For that model on OpenRouter, disable reasoning and request JSON-object
output. Continue enforcing the plan shape, numeric references, and complete coverage
locally. Treat null or blank content as a retryable parse failure for every model.

**Tradeoff:** Qwen 3.7 Flash loses hidden reasoning for this narrow extraction task
and JSON Schema is no longer provider-enforced, but local parsing and validation
retain correctness while avoiding empty completions.
