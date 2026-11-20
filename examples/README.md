# Examples

The end-to-end test in `tests/test_e2e.py` runs a complete two-generation experiment on a **fake trainer** in a temporary Git repository. It exercises lineage tracking, selection, checkpoint and resume semantics without claiming a CUDA experiment or a real language-model `val_bpb` result.

For a demonstration on any platform, run `python -m unittest tests.test_e2e -v` from the repository root. For actual research, use `uv run prepare.py` and the live LLM configuration from the main README on a CUDA GPU.
