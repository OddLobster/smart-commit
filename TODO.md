# TODO

## Active
- [x] Finish and verify one-shot corrective retry for invalid model commit plans
- [x] Reinstall the local CLI from the updated working tree
- [x] Handle null model content and Qwen 3.7 Flash response compatibility
- [ ] Add `make clean` and wire it into `install`/`reinstall` so a stale `build/lib`
      can never ship old code again
- [ ] Decide the fallback when the repair round also fails under `-y` (currently
      exits 4 with no plan and no editor)
