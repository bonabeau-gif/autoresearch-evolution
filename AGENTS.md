# Agent operational rules

Read [`program.md`](program.md) first. It is the canonical research specification for this repo. The executable orchestration lives in `autoresearch_evo/`; its LLM operators are instructed to modify **only** `train.py` in candidate commits.

* **Objective:** strictly minimize fixed held-out `val_bpb`. Do not alter the metric, validation shard, `prepare.py`, data cache, or evaluation parameters to improve it.
* **Budget:** exactly five minutes of model training per evaluated individual, on one GPU. A 10-minute process timeout is a failure guard, not extra allowed training.
* **Reproduction:** seed 42 stays fixed in `train.py`; record every evaluated variant as a Git commit with both TSV ledgers. Failed variants keep blank score/memory fields.
* **Search:** population size 6, top 2 elites, tournament size 3, 0.3 crossover, 40/40/20 conservative/moderate/radical mutation spread. Encourage diversity, learn from failed and near-miss ideas, prefer simpler code only when measured scores tie.
* **Generated code:** Return complete Python source, not patches, commentary or fences. Use only dependencies in `pyproject.toml`. `model(x)` must return token logits to the fixed evaluator.
* **Execution:** `uv run python -m autoresearch_evo init --tag <unique-tag>` followed by `uv run python -m autoresearch_evo run --provider openai --hours 8`. Initialization must run once per tag; subsequent `run` invocations resume.
* **Stop:** No human confirmation required between generations. Stop after five generations with no best-score improvement, three majority-failure generations, interruption, or specified resource limit. Always leave the experiment branch at its best measured candidate.

**Security note:** Running LLM-produced Python is unsafe on machines with sensitive data. Use isolated GPU infrastructure. For package provenance and score comparability, read `docs/BENCHMARK.md` and `docs/SAFETY.md`.
