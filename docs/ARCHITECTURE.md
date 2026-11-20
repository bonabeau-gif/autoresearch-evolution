# Orchestration architecture

## Control flow

```
master baseline train.py
         |
     init(tag) -------> immutable prepare.py hash + environment.txt
         |
         v
     generation 0: baseline -> mutate x5 -> evaluate and commit
         |
         v
  top-K carryover -> tournament selection -> mutation or crossover
         |                                    |
         |                           LLM returns full train.py
         |                                    |
         |                         git commit + permanent ref
         |                                    |
         |                         import-preflight and repairs
         |                                    |
         |                      one 300-second CUDA experiment
         |                                    |
         +----------------------- results.tsv / population.tsv
                                               |
                                 checkpoint + summary + HTML
                                               |
                              repeat until plateau/failure/limit
```

## Storage contracts

`results.tsv` is append-only in *evaluation order*, with columns:

`commit status val_bpb memory_gb description`

`population.tsv` has one row per `(gen,slot)`, including unchanged elites:

`gen slot commit status val_bpb memory_gb parent1 parent2 method description`

The official delimiter is a single **tab** and descriptions are sanitized of tabs, newlines and commas. Failed/timed-out candidates have empty metrics and are excluded from parent tournaments. Elites reuse their earlier commits and metrics but are not appended to `results.tsv`.

`checkpoint.txt` describes the latest generation, its completeness, the global champion, diversity spread and evaluation count. `state.json` snapshots the initial config, fixed evaluator hash, experiment tag and selection RNG state. Git commits pin code, whereas the two TSV files pin empirical results. Run logs live at `runs/<UTC>-<sha>.log`, with the latest copied to `run.log`.

## Branches, tags and candidate permanence

Candidate creation rewinds the current `autoresearch/<tag>` branch to the selected parent and commits **only** `train.py`. This is necessary to support non-greedy lineages, but means recent candidates might not be on the first-parent history of the final branch. To preserve them, the runner creates `refs/autoresearch/individuals/<full-sha>`. The best candidate is tagged as `autoresearch/<tag>/gen<N>-best`, and as `gen<N>-best` when that name is free. At checkpoints, the branch points to the global best commit. A `converged` tag marks early stopping after a fitness plateau.

## Resuming and failure policy

The `run` command is also `resume`. It inspects `population.tsv` for the first incomplete generation, skips slots already recorded, and resumes at the next slot. It reconciles a partial two-file append where `population.tsv` was flushed but `results.tsv` was not. Selection PRNG state is captured after each recorded candidate. A stopped run needs its original `.git` history and untracked TSV/checkpoint/state files, not merely the `train.py` at HEAD.

Preflight errors trigger at most two agent repairs and amended commits. Trivial runtime errors can trigger one repair and retrial. Out-of-memory crashes are logged, not automatically enlarged into further VRAM pressure. A 600-second elapsed timeout kills the entire process group. Three consecutive majority-failure generations or five unimproved generations stop the search.

## Human inspection

Search `results.tsv` or open the offline `reports/progress.html`. The HTML shows measured validation quality, population genealogy, and descriptions for successful and failed variants. Before publishing a result, rerun the best candidate on its exact recorded environment and replicate it with a separate held-out test set.
