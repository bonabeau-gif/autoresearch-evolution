"""Fixed data preparation and validation harness for autoresearch-evolution.

DO NOT EDIT DURING AN EXPERIMENT.  It downloads fixed ClimbMix parquet shards,
tokenizes them with the fixed GPT-2 encoding and stores uint16 token arrays.
This is a self-contained benchmark implementation, not a byte-for-byte copy of
Karpathy's original 8192-token RustBPE autoresearch benchmark. Results from the
two benchmarks are *not* numerically comparable.

    uv run prepare.py
    uv run prepare.py --train-shards 2 --max-train-tokens 8000000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

MAX_SEQ_LEN = 512
TIME_BUDGET = 300  # seconds of training, not including setup/evaluation
EVAL_TOKENS = 262_144  # fixed held-out token count
TRAIN_SHARDS = (0, 1)
VAL_SHARD = 6542
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
CACHE_DIR = Path.home() / ".cache" / "autoresearch"
DATA_DIR = CACHE_DIR / "data"
TOKENIZER_DIR = CACHE_DIR / "tokenizer"
TRAIN_FILE = CACHE_DIR / "train_tokens.npy"
VAL_FILE = CACHE_DIR / "val_tokens.npy"
TOKEN_BYTES_FILE = TOKENIZER_DIR / "gpt2_token_bytes.npy"
MANIFEST_FILE = CACHE_DIR / "manifest.json"
VOCAB_SIZE = 50_257
EOS_TOKEN = 50_256


def _download_shard(shard_id: int) -> Path:
    import requests

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / f"shard_{shard_id:05d}.parquet"
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"{BASE_URL}/{target.name}"
    for attempt in range(4):
        part = target.with_suffix(".part")
        try:
            with requests.get(url, stream=True, timeout=(20, 180)) as resp:
                resp.raise_for_status()
                with open(part, "wb") as out:
                    for block in resp.iter_content(chunk_size=1 << 20):
                        if block:
                            out.write(block)
            os.replace(part, target)
            return target
        except (requests.RequestException, OSError):
            part.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("download attempts exhausted")


def _tokenize_shards(paths: list[Path], destination: Path, limit: int) -> dict:
    import pyarrow.parquet as pq
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    # The destination is written atomically, so interrupted preparation does
    # not leave seemingly complete token arrays behind.
    sequence = np.empty(limit, dtype=np.uint16)
    written = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for group in range(pf.num_row_groups):
            rows = pf.read_row_group(group, columns=["text"])["text"].to_pylist()
            for doc in rows:
                if not isinstance(doc, str) or not doc:
                    continue
                tokens = [EOS_TOKEN] + enc.encode_ordinary(doc)
                count = min(len(tokens), limit - written)
                if count:
                    sequence[written : written + count] = tokens[:count]
                    written += count
                if written == limit:
                    break
            if written == limit:
                break
        if written == limit:
            break
    if written <= MAX_SEQ_LEN + 1:
        raise RuntimeError("too few tokens in dataset")
    temp = destination.with_suffix(".tmp.npy")
    np.save(temp, sequence[:written])
    os.replace(temp, destination)
    return {"tokens": written, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def _write_tokenizer_metadata() -> None:
    import tiktoken

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    token_byte_counts = np.array(
        [len(enc.decode_single_token_bytes(i)) for i in range(VOCAB_SIZE - 1)] + [0],
        dtype=np.int32,
    )
    np.save(TOKEN_BYTES_FILE, token_byte_counts)
    (TOKENIZER_DIR / "tokenizer.json").write_text(
        json.dumps({"name": "gpt2", "vocab_size": VOCAB_SIZE, "eos_token_id": EOS_TOKEN}, indent=2) + "\n"
    )


def cache_ready() -> bool:
    if not all(p.is_file() for p in (TRAIN_FILE, VAL_FILE, TOKEN_BYTES_FILE, MANIFEST_FILE)):
        return False
    try:
        manifest = json.loads(MANIFEST_FILE.read_text())
        return manifest["benchmark"] == "climbmix-gpt2-v1" and manifest["val_shard"] == VAL_SHARD
    except (OSError, ValueError, KeyError):
        return False


def load_tokens(split: str) -> np.ndarray:
    if not cache_ready():
        raise RuntimeError("Data missing. Run: uv run prepare.py")
    if split not in ("train", "val"):
        raise ValueError(f"unknown split: {split}")
    return np.load(TRAIN_FILE if split == "train" else VAL_FILE, mmap_mode="r")


def get_batch(tokens: np.ndarray, batch_size: int, seq_len: int, device, rng: np.random.Generator):
    """Random contiguous autoregressive training sequences."""
    import torch

    if len(tokens) <= seq_len + 1:
        raise ValueError("Dataset too small for chosen context length")
    offsets = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
    data = np.stack([tokens[int(i) : int(i) + seq_len + 1] for i in offsets]).astype(np.int64)
    x = torch.from_numpy(data[:, :-1]).to(device, non_blocking=True)
    y = torch.from_numpy(data[:, 1:]).to(device, non_blocking=True)
    return x, y


@__import__("torch").no_grad()
def evaluate_bpb(model, batch_size: int = 4) -> float:
    """Fixed validation metric: negative log likelihood in bits / UTF-8 byte.

    The evaluation order, examples, tokenization, sequence length and byte
    lengths are fixed here. Evaluated variants may change the model but not this
    function. Logits are read through model(input_ids).
    """
    import torch
    import torch.nn.functional as F

    tokens = load_tokens("val")
    n = min(EVAL_TOKENS, ((len(tokens) - 1) // MAX_SEQ_LEN) * MAX_SEQ_LEN)
    if n < MAX_SEQ_LEN:
        raise ValueError("Validation data is too small")
    byte_lut = torch.from_numpy(np.load(TOKEN_BYTES_FILE).astype(np.int64)).to("cuda")
    device = torch.device("cuda")
    loss_sum, byte_sum = 0.0, 0
    was_training = model.training
    model.eval()
    for start in range(0, n, batch_size * MAX_SEQ_LEN):
        blocks = min(batch_size, (n - start) // MAX_SEQ_LEN)
        windows = np.stack([
            tokens[start + j * MAX_SEQ_LEN : start + (j + 1) * MAX_SEQ_LEN + 1]
            for j in range(blocks)
        ]).astype(np.int64)
        inputs = torch.from_numpy(windows[:, :-1]).to(device)
        targets = torch.from_numpy(windows[:, 1:]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
            logits = model(inputs)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        lengths = byte_lut[targets.reshape(-1)]
        include = lengths > 0
        loss_sum += loss[include].sum().item()
        byte_sum += lengths[include].sum().item()
    model.train(was_training)
    if byte_sum <= 0:
        raise RuntimeError("no validation text bytes")
    return loss_sum / (byte_sum * __import__("math").log(2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed ClimbMix autoresearch benchmark")
    parser.add_argument("--train-shards", type=int, default=2, help="Training shards 0 through N-1")
    parser.add_argument("--max-train-tokens", type=int, default=8_000_000)
    parser.add_argument("--max-val-tokens", type=int, default=500_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.train_shards < 1 or args.max_train_tokens < 2048 or args.max_val_tokens <= EVAL_TOKENS:
        parser.error("Need >=1 train shard, >=2048 training tokens, and > EVAL_TOKENS validation tokens")
    if cache_ready() and not args.force:
        print(f"Cache already prepared at {CACHE_DIR} (use --force to rebuild)")
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_paths = [_download_shard(i) for i in range(args.train_shards)]
    validation_path = _download_shard(VAL_SHARD)
    _write_tokenizer_metadata()
    train_info = _tokenize_shards(train_paths, TRAIN_FILE, args.max_train_tokens)
    val_info = _tokenize_shards([validation_path], VAL_FILE, args.max_val_tokens)
    manifest = {
        "benchmark": "climbmix-gpt2-v1",
        "train_shards": list(range(args.train_shards)),
        "val_shard": VAL_SHARD,
        "max_seq_len": MAX_SEQ_LEN,
        "time_budget_seconds": TIME_BUDGET,
        "eval_tokens": EVAL_TOKENS,
        "tokenizer": "tiktoken:gpt2",
        "train": train_info,
        "val": val_info,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {train_info['tokens']:,} train and {val_info['tokens']:,} validation tokens")
    print(f"Cache: {CACHE_DIR}")


if __name__ == "__main__":
    main()
