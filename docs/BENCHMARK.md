# Fixed benchmark protocol

This project uses a **separate benchmark implementation** modeled on autoresearch concepts, not the original repository's tokenizer/evaluator. The fixed `prepare.py` is tracked and hash-checked at initialization. To reproduce an evaluation, use the `train.py` from that candidate commit together with the exact `prepare.py`, tokenizer, data manifest and hardware recorded during the run.

## Data and tokenizer

Source: `karpathy/climbmix-400b-shuffle` on Hugging Face. Default training shards are indices 0 and 1, with a *pinned, disjoint validation shard 6542*. The default data-prep script stores up to 8,000,000 training tokens and 500,000 validation tokens, with a deterministic tokenization order. Tokenizer: `tiktoken` GPT-2, 50,257 tokens. Shard order, tokenizer, sequence length, EOS semantics, cache file hashes and validation slice are fixed per run. Files live under `~/.cache/autoresearch/`.

`uv run prepare.py` only needs to run once. After preparation, `manifest.json` stores the token array SHA-256 hashes. For strict replication, archive `manifest.json`, the tokenizer files, and the actual data artifacts separately, respecting data licenses. If cache was made using different `--max-...` options, do not mix comparisons across such runs.

## Training and validation

* A candidate is the entirety of `train.py`; an LLM can change the architecture, optimizer and hyperparameters.
* GPU training runs for `TIME_BUDGET = 300` wall-clock seconds (excluding setup and final evaluation). Each step may overrun the boundary by a small amount. The independent orchestration timeout is 600 seconds, including startup and validation.
* The fixed evaluator consumes a deterministic first `EVAL_TOKENS = 262144` target-token window from the held-out data using context length `MAX_SEQ_LEN = 512`.
* The metric is held-out negative log-likelihood divided by the number of UTF-8 bytes represented by target tokens, converted from nats to bits:

  `val_bpb = [sum_t -log p_theta(x_t | x_<t)] / [ln(2) * sum_t bytes(x_t)]`

  EOS special tokens have byte length 0 and are excluded from numerator and denominator. Token byte lengths are the *token's underlying byte sequence* from tiktoken, not an estimated character count.
* A lower `val_bpb` is better. `peak_vram_mb` is diagnostics only.

## Comparison caveats

This benchmark is **not numerically comparable** with Karpathy's original autoresearch metrics. Its context length, tokenizer, sampled validation budget, model architecture, and optimizer differ. It is also not valid to compare scores from different GPUs, data caches, tokenizer manifests, altered evaluations or incompatible PyTorch kernels as if they came from one controlled run. The same validation shard is reused for repeated selection, so an optimized candidate may overfit the specific holdout; confirm leaders on a separate untouched dataset before making scientific performance claims.

## Data integrity checks

Before starting, verify the data manifest and `sha256sum prepare.py`. The runner captures the SHA-256 of `prepare.py` and refuses to run if it changes, including after evaluating a candidate. `train.py` changes are staged alone, and no generator is allowed to alter the evaluator or the dependency manifest during the campaign.
