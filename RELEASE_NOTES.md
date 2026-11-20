# Release 0.1.0 — November 20, 2026

## Included

- The supplied `program.md`, byte-for-byte preserved.
- Independent single-GPU PyTorch Transformer `train.py` baseline with seed 42, a fixed 300-second training budget, and machine-readable `val_bpb`/VRAM output.
- Fixed `prepare.py` benchmark with ClimbMix training shards, a disjoint held-out validation shard, tiktoken GPT-2 tokenizer, cache manifest, and fixed bits-per-byte evaluator.
- A full evolutionary controller implementing initialization, population of six, top-two elites, tournaments, intelligent mutation, crossover, repair, GPU process timeouts, generation summaries, and pause/resume.
- An OpenAI Responses API LLM backend with no new runtime dependency, plus an explicitly non-scientific deterministic mock backend for tests.
- Permanent Git individual refs, per-generation tags, environment snapshot and data-integrity checks.
- TSV ledgers, progress checkpoints, raw run logs, best-candidate summaries, and standalone offline HTML dashboard with interactive trends/lineage/table.
- Configuration file, agent instructions, CI workflow, docs, Makefile and experiment audit script.

## Validation performed

- Python source compilation of core modules and train/prepare code.
- Ten passing automated tests, including a two-generation real-Git end-to-end experiment with a fake (non-ML) training script and a restart from saved state.
- CPU-only forward pass on the bundled Transformer, producing finite logits in shape `(batch, sequence, vocabulary)`.

## Not validated in this build environment

- Real 300-second CUDA training or reference `val_bpb` score: no CUDA GPU/data shards are available here.
- `uv sync` dependency resolution on the user's target GPU machine, dataset download, and a live LLM API call.
- Numerical comparability with the original Karpathy autoresearch benchmark. This repo uses different tokenizer/model/evaluation constants and must be treated as its own controlled benchmark.

## First-run action

Run `uv sync`, then `uv run prepare.py` and `uv run train.py` on the target CUDA machine. Once this hardware/data baseline works, initialize `autoresearch/nov20` with the CLI and start the autonomous LLM evolution process. Commit the generated `uv.lock` to pin the resolved package set for future work.
