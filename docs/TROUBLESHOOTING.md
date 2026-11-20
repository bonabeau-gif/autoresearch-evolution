# Setup and recovery

**`No CUDA GPU detected`:** Train requires one NVIDIA GPU with compatible PyTorch CUDA builds. The CPU-only test environment supports `python -m unittest`, but is not a valid training platform. Run `nvidia-smi` and `uv run python -c "import torch; print(torch.cuda.is_available())"`.

**`Missing prepared data`:** Run `uv run prepare.py` with internet access. A pinned validation shard is downloaded separately. Large parquet downloads need sufficient disk and network quota. Files are cached in `~/.cache/autoresearch/`.

**`OPENAI_API_KEY not set`:** Set the key and `OPENAI_MODEL` or pass `--model`. This backend uses the Responses API. The model must support returning complete multi-thousand-token Python files. `--provider mock` is only for orchestration diagnostics and does not conduct semantic research.

**`Branch already exists`:** Pick a new `--tag`, or resume the existing experiment with `run`. Never initialize over an existing `state.json`.

**`prepare.py checksum changed`:** Restore the exact fixed evaluator from the original baseline commit and verify the data manifest. Do not silently accept changed metrics mid-run.

**`Tracked files modified`:** Commit or restore changes before running the controller. The controller aggressively resets its experiment branch to selected parent commits and deliberately refuses dirty tracked trees.

**Repeated OOM:** Reduce model width, context memory or batch sizes in the next candidates. Failures are recorded with blank metrics as specified. A candidate with three or more failed descendants is deprioritized during parent selection.

**Restart after power loss:** Keep `.git/`, `state.json`, both TSV files, `checkpoint.txt`, and the cached dataset. Run exactly the same `run` command. A candidate committed immediately before a crash but not yet logged can remain pinned in Git while its slot is retried.

**Create a clean new study:** Archive logs and state, return to `master` in a separate directory or Git worktree, and choose a fresh run tag. The system intentionally refuses to recycle a tag within the same repository.
