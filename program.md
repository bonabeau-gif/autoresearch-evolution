# autoresearch

An evolutionary approach to LLM-driven ML research. Instead of greedy hill-climbing, we maintain a **population** of `train.py` variants and evolve them through selection, mutation, and recombination.

## Principles

These apply throughout the experiment and are not repeated elsewhere:

1. **Minimize val_bpb.** That is the only objective. The time budget is fixed at 5 minutes, so throughput is not a variable — only final metric quality matters.
2. **Simplicity wins ties.** When two variants achieve similar val_bpb, prefer the simpler code. Deleting code for equal performance is always a good outcome.
3. **Diversity over convergence.** Each generation should include a spread of conservative, moderate, and radical changes. Don't shy away from ideas that might crash — a 50% crash rate on radical mutations is acceptable if the survivors occasionally break through. Spread parentage across top performers rather than always selecting the single best.
4. **Build on signal, don't repeat noise.** Before generating a new variant, review what has been tried. Avoid duplicating failed ideas. Combine near-misses. Revisit crashed ideas that might work with small fixes.
5. **VRAM is a soft constraint.** Some increase is acceptable for meaningful val_bpb gains, but memory should not blow up dramatically. Note memory in logs so lineages can self-correct.

## Setup

Work with the user to complete these steps:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Capture environment**: Record the output of `nvidia-smi --query-gpu=name,memory.total --format=csv` and `python -c "import torch; print(torch.__version__, torch.version.cuda)"` in a file `environment.txt` at the repo root (untracked). This is a reproducibility anchor.
6. **Initialize tracking files**: Create `results.tsv` and `population.tsv` with their header rows (see Logging).
7. **Run the baseline**: Run the unmodified `train.py` to establish the baseline. Record it in both TSVs.
8. **Seed the initial population**: Generate `POPULATION_SIZE - 1` mutants from the baseline to fill out generation 0.
9. **Confirm and go**: Confirm setup looks good, then begin evolution.

## Core Constraints

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). Launch with: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Architecture, optimizer, hyperparameters, training loop, batch size, model size — all fair game.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies beyond `pyproject.toml`.
- Modify the evaluation harness (`evaluate_bpb` in `prepare.py`).

## Evolutionary Concepts

| Biology | autoresearch |
|---|---|
| Genome | The source code of `train.py` |
| Individual | A specific version of `train.py` (a git commit) |
| Fitness | `val_bpb` (lower is better — minimization) |
| Population | The set of active individuals being evolved |
| Generation | One full cycle of evaluate → select → reproduce |
| Mutation | LLM modifies a single parent's `train.py` |
| Crossover | LLM combines ideas from two parents' `train.py` files |
| Selection | Choosing which individuals become parents |

The LLM acts as an intelligent mutation and crossover operator — unlike random perturbation, it can make semantically meaningful changes guided by understanding of the code and domain knowledge.

## Parameters

Adjust based on available compute time:

- **POPULATION_SIZE**: 6 (individuals per generation)
- **TOP_K**: 2 (elites that survive unchanged)
- **TOURNAMENT_SIZE**: 3 (candidates sampled per tournament)
- **CROSSOVER_RATE**: 0.3 (probability of crossover vs. mutation)

At ~6 minutes per evaluation, one generation takes ~36 minutes. An 8-hour run yields roughly 13 generations (~78 evaluations).

## Logging

Maintain two untracked TSV files (tab-separated — never use commas in any field).

### results.tsv — chronological run log

```
commit	status	val_bpb	memory_gb	description
```

| Column | Type | Description |
|---|---|---|
| `commit` | string | Git commit hash, short (7 chars) |
| `status` | enum | `ok`, `crash`, or `timeout` |
| `val_bpb` | float or empty | Achieved val_bpb (e.g. `1.234567`). **Empty when status ≠ ok.** |
| `memory_gb` | float or empty | Peak VRAM in GB, rounded to .1f. **Empty when status ≠ ok.** |
| `description` | string | Short text. No commas. |

### population.tsv — generational population state

```
gen	slot	commit	status	val_bpb	memory_gb	parent1	parent2	method	description
```

| Column | Type | Description |
|---|---|---|
| `gen` | int | Generation number (0 for initial population) |
| `slot` | int | Index within the generation (0 to POPULATION_SIZE-1) |
| `commit` | string | Git commit hash, short (7 chars) |
| `status` | enum | `ok`, `crash`, or `timeout` |
| `val_bpb` | float or empty | **Empty when status ≠ ok.** |
| `memory_gb` | float or empty | **Empty when status ≠ ok.** |
| `parent1` | string or empty | Commit hash of first parent. Empty for baseline. |
| `parent2` | string or empty | Commit hash of second parent. Empty unless method is `crossover`. |
| `method` | enum | One of: `baseline`, `mutation`, `crossover`, `elite` |
| `description` | string | Short text. No commas. |

Example:

```
gen	slot	commit	status	val_bpb	memory_gb	parent1	parent2	method	description
0	0	a1b2c3d	ok	0.997900	44.0			baseline	unmodified baseline
0	1	b2c3d4e	ok	0.993200	44.2	a1b2c3d		mutation	increase LR to 0.04
0	2	c3d4e5f	ok	1.005000	44.0	a1b2c3d		mutation	switch to GeLU activation
0	3	d4e5f6g	crash			a1b2c3d		mutation	double model width (OOM)
0	4	e5f6g7h	ok	0.991500	45.1	a1b2c3d		mutation	add rotary embeddings
0	5	f6g7h8i	ok	0.995300	43.8	a1b2c3d		mutation	reduce depth to 6 and increase width
1	0	e5f6g7h	ok	0.991500	45.1	e5f6g7h		elite	carried forward
1	1	b2c3d4e	ok	0.993200	44.2	b2c3d4e		elite	carried forward
1	2	g7h8i9j	ok	0.989100	44.8	e5f6g7h	b2c3d4e	crossover	combine rotary embed + higher LR
1	3	h8i9j0k	ok	0.990800	45.0	e5f6g7h		mutation	add weight decay 0.1
```

Crashed and timed-out individuals have empty `val_bpb` and `memory_gb` fields. They are excluded from selection and fitness comparisons.

## Output Format

The script prints a summary block:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Extract key metrics:

```
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

## Selection

Use **tournament selection** to choose parents:

1. Sample `TOURNAMENT_SIZE` individuals uniformly at random from the current population, excluding any with status ≠ `ok`.
2. The individual with the best (lowest) `val_bpb` wins and becomes a parent.
3. Repeat to select as many parents as needed.

The top `TOP_K` individuals by `val_bpb` are **elites** — they survive unchanged into the next generation without re-evaluation.

## Mutation Strategy

When mutating a parent, read its `train.py` and make a single coherent change. Vary intensity across the population:

- **Conservative** (~40%): Small hyperparameter tweaks — learning rate, batch size, warmup steps, weight decay, dropout. Changes within 2x of current values.
- **Moderate** (~40%): Structural changes within the existing paradigm — activation functions, normalization, attention head counts, embedding dimensions, layer counts.
- **Radical** (~20%): Architectural departures — different model families, novel training techniques, unusual optimizer combinations, significantly different model shapes.

## Crossover Strategy

Read both parents' `train.py` files (use `git show <commit>:train.py`) and combine their strengths into a single coherent `train.py`. This is intelligent recombination, not mechanical splicing:

- Identify what makes each parent good (e.g., one has a great LR schedule, the other has an effective attention mechanism).
- Combine complementary strengths.
- Resolve conflicts sensibly (pick one approach or interpolate).

Crossover is most valuable when two parents improved val_bpb through orthogonal changes.

## Preflight Check

Before every 5-minute training run, execute a fast smoke test:

```bash
python -c "import train" 2>&1
```

This catches syntax errors, missing imports, and module-level exceptions in seconds rather than minutes. If the preflight fails:

1. Read the error.
2. Fix the issue in `train.py`.
3. Amend the commit: `git add train.py && git commit --amend --no-edit`.
4. Re-run the preflight.
5. If it still fails after 2 fix attempts, record the individual as status=`crash` and move on.

Only proceed to `uv run train.py > run.log 2>&1` after the preflight passes.

## Reproducibility

Each run must be reproducible from its git commit:

1. **Random seed**: Set `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, and `random.seed(42)` at the top of `train.py` (or keep them if already present). The seed value stays constant — variation comes from code changes, not randomness.
2. **Deterministic caveats**: Full `torch.use_deterministic_algorithms(True)` is NOT required (it disables efficient kernels). Seed-setting provides sufficient reproducibility for comparing variants.
3. **One commit per variant**: Every evaluated variant must have its own git commit. The code at that commit, plus the fixed `prepare.py` and data, must be sufficient to reproduce the run.
4. **Environment file**: The `environment.txt` captured during setup documents GPU hardware and software versions. If the environment changes mid-experiment, update this file and note it in the results description.

## The Evolution Loop

### Generation 0: Seeding

1. Run the unmodified baseline. Record as slot 0 with method=`baseline`.
2. Generate `POPULATION_SIZE - 1` mutations from the baseline, using a mix of intensities.
3. For each: preflight → evaluate → record.

### Generation N → N+1: Evolve

**1. Review.** Read `population.tsv` for the latest generation. Identify elites (top `TOP_K` by `val_bpb` among status=`ok`). Note which ideas helped and which didn't.

**2. Carry forward elites.** The `TOP_K` best individuals pass into the next generation unchanged (method=`elite`). They keep their commits and are not re-evaluated.

**3. Produce offspring** for each remaining slot:

   a. Roll against `CROSSOVER_RATE` to decide mutation vs. crossover. Select parents via tournament.

   b. **Mutation path:**
      - `git checkout <parent_commit>`
      - Read and modify `train.py`
      - `git add train.py && git commit -m "<description>"`
      - Preflight check
      - `uv run train.py > run.log 2>&1`
      - Extract results

   c. **Crossover path:**
      - Read both parents via `git show <commit>:train.py`
      - Write a merged `train.py`
      - Commit, preflight, evaluate

   d. Record in both `population.tsv` and `results.tsv`.

**4. Handle failures.**
   - If preflight fails: attempt up to 2 fixes, then record as status=`crash`.
   - If the run crashes at runtime: if trivial (typo, off-by-one), fix and retry once. Otherwise record as status=`crash`.
   - If the run exceeds 10 minutes: `kill` the process, record as status=`timeout`.
   - If a parent repeatedly produces failing offspring (3+ crashes across the experiment), deprioritize it in selection.

**5. Advance the branch.** Check out the best individual's commit: `git checkout <best_commit>`. Tag it: `git tag gen<N>-best <best_commit>`.

**6. Checkpoint.** After every generation, write a summary to `checkpoint.txt` (untracked):

```
generation: 7
best_commit: g7h8i9j
best_val_bpb: 0.989100
population_spread: 0.012
total_evaluations: 48
timestamp: 2025-03-14T08:32:00Z
```

This serves as a resumption point if the session is interrupted.

**7. Next generation.** Return to step 1.

## Stopping and Checkpointing

The default behavior is **continuous autonomous operation** — do not pause to ask the human for confirmation between generations. The human may be away and expects progress without interaction.

The loop should stop gracefully under these conditions:

- **Convergence plateau**: If the global best `val_bpb` has not improved for 5 consecutive generations, write a final summary (see below), tag the branch `converged`, and stop.
- **Catastrophic failure**: If 3 consecutive generations have majority failures (>50% of slots with status ≠ `ok`), stop and write a diagnostic to `summary.txt`.
- **External interruption**: The human may kill the process at any time. The per-generation `checkpoint.txt` ensures progress is not lost.

When stopping for any reason, always:
1. Ensure `results.tsv` and `population.tsv` are fully up to date.
2. Write or update `checkpoint.txt`.
3. Ensure the branch HEAD points to the best known commit.
4. Write `summary.txt` with: best val_bpb achieved, total generations, total evaluations, top 5 individuals by fitness, and a brief narrative of what worked.
