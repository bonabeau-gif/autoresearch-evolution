"""Git-backed evolutionary experiment orchestration.

All candidate genomes are complete versions of train.py.  prepare.py is
protected by a checksum and the runner stages only train.py for every variant.
Generation state is append-only TSV plus untracked checkpoints. Candidate
commits are pinned with refs/autoresearch/individuals/<sha> so that moving the
experiment branch to a different parent never garbage-collects a lineage.
"""
from __future__ import annotations

import ast
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .llm import MockProvider, OpenAIProvider, clean_description

RESULT_HEADERS = ("commit", "status", "val_bpb", "memory_gb", "description")
POP_HEADERS = ("gen", "slot", "commit", "status", "val_bpb", "memory_gb",
               "parent1", "parent2", "method", "description")
VALID_STATUSES = {"ok", "crash", "timeout"}
VALID_METHODS = {"baseline", "mutation", "crossover", "elite"}
UTC = dt.timezone.utc


class StopRequested(Exception):
    """Raised when a graceful shutdown is requested."""


@dataclass(frozen=True)
class Individual:
    gen: int
    slot: int
    commit: str
    status: str
    val_bpb: float | None
    memory_gb: float | None
    parent1: str
    parent2: str
    method: str
    description: str

    def __post_init__(self):
        if self.status not in VALID_STATUSES or self.method not in VALID_METHODS:
            raise ValueError("invalid population row")
        if self.status == "ok" and (self.val_bpb is None or not math.isfinite(self.val_bpb)):
            raise ValueError("valid individual must have finite val_bpb")
        if self.status != "ok" and (self.val_bpb is not None or self.memory_gb is not None):
            raise ValueError("failed individuals must not have metric values")
        if any(x in self.description for x in (",", "\t", "\n", "\r")):
            raise ValueError("description may not contain commas or tabs")

    @property
    def viable(self) -> bool:
        return self.status == "ok"

    def pop_row(self) -> dict[str, str]:
        out = {name: str(getattr(self, name)) for name in POP_HEADERS}
        out["val_bpb"] = "" if self.val_bpb is None else f"{self.val_bpb:.6f}"
        out["memory_gb"] = "" if self.memory_gb is None else f"{self.memory_gb:.1f}"
        return out

    def results_row(self) -> dict[str, str]:
        return {k: self.pop_row()[k] for k in RESULT_HEADERS}

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Individual":
        return cls(
            gen=int(row["gen"]), slot=int(row["slot"]), commit=row["commit"],
            status=row["status"], val_bpb=float(row["val_bpb"]) if row["val_bpb"] else None,
            memory_gb=float(row["memory_gb"]) if row["memory_gb"] else None,
            parent1=row["parent1"], parent2=row["parent2"],
            method=row["method"], description=row["description"],
        )


def read_population(path: Path) -> list[Individual]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if tuple(reader.fieldnames or ()) != POP_HEADERS:
            raise ValueError("population.tsv header mismatch")
        rows = [Individual.from_row(row) for row in reader]
    seen = set()
    for row in rows:
        identity = (row.gen, row.slot)
        if identity in seen:
            raise ValueError(f"Duplicate generation slot: {identity}")
        seen.add(identity)
    return rows


def _write_header(path: Path, names: tuple[str, ...]) -> None:
    if path.exists():
        raise FileExistsError(f"Tracking file already exists: {path}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(names)
        f.flush()
        os.fsync(f.fileno())


def _append_row(path: Path, names: tuple[str, ...], row: dict[str, str]) -> None:
    if any("," in str(v) for v in row.values()):
        raise ValueError("commas are forbidden in TSV fields")
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(names), delimiter="\t", lineterminator="\n")
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def git(root: Path, *args: str, check: bool = True) -> str:
    cmd = ["git", "-C", str(root), *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if check and res.returncode:
        raise RuntimeError(f"Git command failed: {' '.join(cmd)}\n{res.stderr.strip()}")
    return res.stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_key(ind: Individual) -> tuple[float, str]:
    return (float("inf") if ind.val_bpb is None else ind.val_bpb, ind.commit)


def tournament(population: list[Individual], k: int, rng: random.Random,
               crash_counts: dict[str, int] | None = None) -> Individual:
    pool = [p for p in population if p.viable]
    if not pool:
        raise ValueError("No viable individuals for selection")
    if crash_counts:
        filtered = [p for p in pool if crash_counts.get(p.commit, 0) < 3]
        if len(filtered) >= 2:
            pool = filtered
    candidates = rng.sample(pool, min(k, len(pool)))
    return min(candidates, key=score_key)


def mutation_intensity(gen: int, slot: int, pop_size: int, elites: int) -> str:
    """Spread 40/40/20 intensities across offspring rather than pure chance."""
    total = pop_size - (elites if gen else 1)
    index = slot - (elites if gen else 1)
    if total < 1 or index < 0:
        raise ValueError("slot is not an offspring")
    n_radical = 1 if total >= 3 else 0
    if total >= 5:
        n_radical = max(1, round(total * 0.2))
    n_conservative = min(total - n_radical, round(total * 0.4))
    n_moderate = total - n_radical - n_conservative
    patterns = (["conservative"] * n_conservative +
                ["moderate"] * n_moderate + ["radical"] * n_radical)
    # Rotate each generation without allowing all offspring to be conservative.
    return patterns[(index + gen) % total]


def parse_metrics(log_text: str) -> tuple[float, float]:
    """Only accept finite val_bpb and peak_vram_mb from the *last* summary."""
    block = log_text.split("\n---\n")[-1]
    match_val = re.search(r"^val_bpb:\s*([+\-0-9.eE]+)\s*$", block, re.M)
    match_mem = re.search(r"^peak_vram_mb:\s*([+\-0-9.eE]+)\s*$", block, re.M)
    if not match_val or not match_mem:
        raise ValueError("Training output missing val_bpb or peak_vram_mb")
    value, mb = float(match_val.group(1)), float(match_mem.group(1))
    if not math.isfinite(value) or value <= 0 or not math.isfinite(mb) or mb < 0:
        raise ValueError("Non-finite or invalid metric output")
    return value, round(mb / 1024, 1)


def crash_parents(rows: list[Individual]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.status != "ok":
            for parent in (row.parent1, row.parent2):
                if parent:
                    counts[parent] = counts.get(parent, 0) + 1
    return counts


def run_quick(cmd: list[str], cwd: Path, timeout: float = 30) -> str:
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"ERROR: {exc}"


def write_environment(root: Path) -> None:
    lines = [
        "# Reproducibility anchor; local machine and installed packages",
        f"timestamp_utc: {dt.datetime.now(UTC).isoformat()}",
        f"python: {sys.version}",
        f"platform: {sys.platform}",
        "command: nvidia-smi --query-gpu=name,memory.total --format=csv",
        run_quick(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], root).strip(),
        'command: python -c "import torch; print(torch.__version__, torch.version.cuda)"',
        run_quick([sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda)"], root).strip(),
        "command: uv --version",
        run_quick(["uv", "--version"], root).strip(),
        f"prepare_sha256: {sha256(root / 'prepare.py')}",
        f"train_baseline_sha256: {sha256(root / 'train.py')}",
    ]
    (root / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


class RunLock:
    def __init__(self, path: Path):
        self.path = path
        self.file = None

    def __enter__(self):
        self.file = open(self.path, "w")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another experiment process is already running") from exc
        self.file.write(str(os.getpid()))
        self.file.flush()
        return self

    def __exit__(self, *_):
        if self.file:
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()
            self.path.unlink(missing_ok=True)


class Evolution:
    def __init__(self, root: Path, provider: str = "openai", model: str = "",
                 train_command: str = "uv run train.py",
                 smoke_command: str = "uv run python -c 'import train'",
                 cuda_device: str = "0"):
        self.root = root.resolve()
        self.state_path = self.root / "state.json"
        if not self.state_path.exists():
            raise RuntimeError("Not initialized. Run: evo-research init --tag nov20")
        self.state: dict[str, Any] = json.loads(self.state_path.read_text())
        self.config = self.state["config"]
        self.rng = random.Random(self.config["seed"])
        if "selection_rng_state" in self.state:
            self.rng.setstate(ast.literal_eval(self.state["selection_rng_state"]))
        self.cuda_device = cuda_device
        self.train_command = shlex.split(train_command)
        self.smoke_command = shlex.split(smoke_command)
        self.stop = False
        self.model = model or os.getenv("OPENAI_MODEL", "")
        self.generator = MockProvider(self.rng) if provider == "mock" else OpenAIProvider(self.model)
        self._check_prepare()

    def _check_prepare(self):
        if sha256(self.root / "prepare.py") != self.state["prepare_sha256"]:
            raise RuntimeError("prepare.py checksum changed: refusing to run altered evaluator")
        for name, digest in self.state.get("cache_sha256", {}).items():
            file = Path.home() / ".cache" / "autoresearch" / name
            if not file.exists() or sha256(file) != digest:
                raise RuntimeError(f"Prepared benchmark artifact changed: {file}")

    def _require_clean(self):
        dirty = git(self.root, "status", "--porcelain", "--untracked-files=no")
        if dirty:
            raise RuntimeError("Tracked files modified. Commit or restore changes before running:\n" + dirty)

    def _save_state(self):
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, indent=2) + "\n")
        os.replace(temp, self.state_path)

    def _commit_source(self, parent: str, code: str, description: str) -> str:
        self._require_clean()
        # All branch rewinds happen after checking the worktree: no other
        # committed/modified files are overwritten by accident.
        git(self.root, "reset", "--hard", parent)
        self._check_prepare()
        destination = self.root / "train.py"
        if code == destination.read_text():
            raise ValueError("Generator proposed an identical genome")
        destination.write_text(code)
        changed = git(self.root, "diff", "--name-only").splitlines()
        if changed != ["train.py"]:
            raise RuntimeError("Forbidden files modified: " + str(changed))
        git(self.root, "add", "--", "train.py")
        git(self.root, "commit", "-m", "evo: " + clean_description(description))
        commit = git(self.root, "rev-parse", "HEAD")
        self._pin_commit(commit)
        self._check_prepare()
        return commit

    def _pin_commit(self, commit: str):
        git(self.root, "update-ref", f"refs/autoresearch/individuals/{commit}", commit)

    def _amend(self, source: str) -> str:
        if source == (self.root / "train.py").read_text():
            raise ValueError("Repair proposal did not change candidate")
        (self.root / "train.py").write_text(source)
        git(self.root, "add", "--", "train.py")
        git(self.root, "commit", "--amend", "--no-edit")
        commit = git(self.root, "rev-parse", "HEAD")
        self._pin_commit(commit)
        self._check_prepare()
        return commit

    def _fetch_source(self, commit: str) -> str:
        return git(self.root, "show", f"{commit}:train.py") + "\n"

    def _preflight(self) -> tuple[bool, str]:
        try:
            result = subprocess.run(self.smoke_command, cwd=self.root,
                                    capture_output=True, text=True,
                                    timeout=self.config["preflight_timeout_seconds"])
            self._check_prepare()
            self._require_clean()
            return result.returncode == 0, (result.stdout + result.stderr)[-5000:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)

    def _evaluate(self, commit: str) -> tuple[str, float | None, float | None, str]:
        (self.root / "runs").mkdir(exist_ok=True)
        name = f"runs/{dt.datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{commit[:7]}.log"
        logfile = self.root / name
        env = os.environ.copy()
        # Do not pass the LLM API key or common cloud access tokens to code
        # generated by a model. This is hardening, NOT a true sandbox.
        for name in list(env):
            upper = name.upper()
            if (upper in {"OPENAI_API_KEY", "GITHUB_TOKEN", "GH_TOKEN", "HF_TOKEN",
                          "HUGGINGFACE_HUB_TOKEN", "AWS_SECRET_ACCESS_KEY",
                          "AWS_SESSION_TOKEN", "AZURE_CLIENT_SECRET", "GCP_API_KEY"}
                    or upper.endswith(("_API_KEY", "_AUTH_TOKEN", "_ACCESS_TOKEN"))):
                env.pop(name, None)
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_device
        start = time.monotonic()
        status = "crash"
        with open(logfile, "w", encoding="utf-8") as file:
            # new process group permits clean killing of GPU child processes
            process = subprocess.Popen(self.train_command, cwd=self.root, env=env,
                                       stdout=file, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                while process.poll() is None:
                    if self.stop:
                        raise StopRequested()
                    if time.monotonic() - start > self.config["training_timeout_seconds"]:
                        status = "timeout"
                        break
                    time.sleep(0.2)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        shutil.copyfile(logfile, self.root / "run.log")
        content = logfile.read_text(errors="replace")
        # Candidate may write logs but must not self-modify tracked files or
        # mutate the fixed evaluator/data cache while being evaluated.
        self._check_prepare()
        self._require_clean()
        if status == "timeout":
            return "timeout", None, None, content
        if process.returncode != 0:
            return "crash", None, None, content
        try:
            value, memory = parse_metrics(content)
        except (ValueError, OverflowError):
            return "crash", None, None, content
        return "ok", value, memory, content

    def _repair_candidate(self, commit: str, error: str, history: str) -> str:
        proposal = self.generator.propose([self._fetch_source(commit)],
                                         method="repair", intensity="conservative",
                                         history=history, hint=error)
        return self._amend(proposal.source)

    def _evaluate_with_repairs(self, commit: str, history: str):
        for attempt in range(self.config["max_preflight_fixes"] + 1):
            passed, error = self._preflight()
            if passed:
                break
            (self.root / "runs").mkdir(exist_ok=True)
            (self.root / "runs" / f"{commit[:7]}-preflight-{attempt}.log").write_text(error)
            if attempt == self.config["max_preflight_fixes"]:
                return commit, "crash", None, None
            try:
                commit = self._repair_candidate(commit, error, history)
            except Exception as exc:
                print(f"Repair failed: {exc}", file=sys.stderr)
                return commit, "crash", None, None
        status, value, memory, log = self._evaluate(commit)
        # Retry trivial runtime errors once. OOM and timeouts are not trivial.
        if status == "crash" and any(error in log for error in
                ("NameError:", "UnboundLocalError:", "SyntaxError:", "IndexError:")):
            try:
                commit = self._repair_candidate(commit, log[-4500:], history)
                passed, _ = self._preflight()
                if passed:
                    status, value, memory, _ = self._evaluate(commit)
            except Exception as exc:
                print(f"Runtime repair failed: {exc}", file=sys.stderr)
        return commit, status, value, memory

    def _record(self, individual: Individual) -> None:
        rows = read_population(self.root / "population.tsv")
        if any((r.gen, r.slot) == (individual.gen, individual.slot) for r in rows):
            raise ValueError("Cannot record a duplicate population slot")
        _append_row(self.root / "population.tsv", POP_HEADERS, individual.pop_row())
        if individual.method != "elite":
            # A crash between these append calls is reconciled at resume.
            _append_row(self.root / "results.tsv", RESULT_HEADERS, individual.results_row())
        self._checkpoint(individual.gen)
        print(f"gen={individual.gen} slot={individual.slot} {individual.method} "
              f"{individual.commit} {individual.status} val_bpb={individual.val_bpb}", flush=True)

    def _reconcile_results(self) -> None:
        result_file = self.root / "results.tsv"
        with open(result_file, encoding="utf-8", newline="") as f:
            logged = {r["commit"] for r in csv.DictReader(f, delimiter="\t")}
        for entry in read_population(self.root / "population.tsv"):
            if entry.method != "elite" and entry.commit not in logged:
                _append_row(result_file, RESULT_HEADERS, entry.results_row())
                logged.add(entry.commit)

    def _rank(self, rows: list[Individual]) -> list[Individual]:
        successful = [r for r in rows if r.viable]
        # Primary metric is strictly val_bpb. Simplicity resolves numerical
        # ties only, never a measurably worse metric.
        size_cache: dict[str, int] = {}
        def key(row: Individual):
            if row.commit not in size_cache:
                size_cache[row.commit] = len(self._fetch_source(row.commit))
            return (round(row.val_bpb or 1e8, 6), size_cache[row.commit], row.commit)
        return sorted(successful, key=key)

    def _history_context(self, records: list[Individual]) -> str:
        chunks = []
        for ind in records[-45:]:
            result = f"{ind.val_bpb:.6f}" if ind.val_bpb is not None else ind.status
            chunks.append(f"gen={ind.gen} commit={ind.commit} result={result} method={ind.method} idea={ind.description}")
        return "\n".join(chunks)

    def _create_offspring(self, generation: int, slot: int,
                          parents: list[Individual], history: list[Individual]) -> Individual:
        n = self.config["population_size"]
        k = self.config["top_k"]
        intensity = mutation_intensity(generation, slot, n, k)
        counts = crash_parents(history)
        p1 = tournament(parents, self.config["tournament_size"], self.rng, counts)
        p2 = None
        method = "mutation"
        if len([p for p in parents if p.viable and p.commit != p1.commit]) > 0 and self.rng.random() < self.config["crossover_rate"]:
            others = [p for p in parents if p.commit != p1.commit]
            p2 = tournament(others, self.config["tournament_size"], self.rng, counts)
            method = "crossover"
        parent_codes = [self._fetch_source(p1.commit)]
        if p2:
            parent_codes.append(self._fetch_source(p2.commit))
        history_text = self._history_context(history)
        proposal = self.generator.propose(parent_codes, method, intensity, history_text)
        if self.stop:
            raise StopRequested()
        new_commit = self._commit_source(p1.commit, proposal.source, proposal.description)
        new_commit, status, value, mem = self._evaluate_with_repairs(new_commit, history_text)
        return Individual(generation, slot, new_commit[:7], status, value, mem,
                          p1.commit, p2.commit if p2 else "", method, proposal.description)

    def _checkpoint(self, generation: int) -> None:
        rows = read_population(self.root / "population.tsv")
        viable = self._rank(rows)
        best = viable[0] if viable else None
        current = [r for r in rows if r.gen == generation]
        vals = [r.val_bpb for r in current if r.viable and r.val_bpb is not None]
        spread = max(vals) - min(vals) if len(vals) > 1 else 0.0
        evaluated = sum(r.method != "elite" for r in rows)
        complete = len(current) == self.config["population_size"]
        info = {
            "generation": generation,
            "generation_complete": str(complete).lower(),
            "best_commit": best.commit if best else "",
            "best_val_bpb": f"{best.val_bpb:.6f}" if best else "",
            "population_spread": f"{spread:.6f}",
            "total_evaluations": evaluated,
            "timestamp": dt.datetime.now(UTC).isoformat(),
        }
        (self.root / "checkpoint.txt").write_text("".join(f"{k}: {v}\n" for k, v in info.items()))
        self.state["last_checkpoint_generation"] = generation
        self.state["selection_rng_state"] = repr(self.rng.getstate())
        self._save_state()

    def _tag_generation(self, generation: int) -> None:
        rows = read_population(self.root / "population.tsv")
        best = self._rank(rows)
        if not best:
            return
        best_full = git(self.root, "rev-parse", best[0].commit)
        git(self.root, "reset", "--hard", best_full)
        self._check_prepare()
        tag = f"autoresearch/{self.state['tag']}/gen{generation}-best"
        if git(self.root, "tag", "--list", tag) == "":
            git(self.root, "tag", tag, best_full)
        # Convenience tag from the program.md, only where not conflicting.
        legacy = f"gen{generation}-best"
        if git(self.root, "tag", "--list", legacy) == "":
            git(self.root, "tag", legacy, best_full)

    def _stop_reason(self) -> str | None:
        rows = read_population(self.root / "population.tsv")
        n = self.config["population_size"]
        if not rows:
            return None
        gens = sorted({row.gen for row in rows})
        complete = [g for g in gens if len([r for r in rows if r.gen == g]) == n]
        if not complete:
            return None
        failures = 0
        for gen in reversed(complete):
            current = [r for r in rows if r.gen == gen]
            if sum(not row.viable for row in current) > n / 2:
                failures += 1
            else:
                break
        if failures >= self.config["catastrophic_generations"]:
            return "catastrophic_failure"
        minima = []
        for gen in complete:
            values = [p.val_bpb for p in rows if p.gen <= gen and p.viable and p.val_bpb is not None]
            minima.append(min(values) if values else float("inf"))
        count = 0
        for i in range(len(minima) - 1, 0, -1):
            if minima[i] >= minima[i - 1] - self.config["tie_epsilon"]:
                count += 1
            else:
                break
        if count >= self.config["plateau_generations"]:
            return "convergence_plateau"
        return None

    def _summary(self, reason: str):
        rows = read_population(self.root / "population.tsv")
        rank = self._rank(rows)
        unique = []
        seen = set()
        for r in rank:
            if r.commit not in seen:
                seen.add(r.commit)
                unique.append(r)
            if len(unique) == 5:
                break
        best = unique[0] if unique else None
        lines = [
            f"stop_reason: {reason}",
            f"best_val_bpb: {best.val_bpb:.6f}" if best else "best_val_bpb: N/A",
            f"best_commit: {best.commit if best else 'N/A'}",
            f"total_generations_started: {len({r.gen for r in rows})}",
            f"total_evaluations: {sum(r.method != 'elite' for r in rows)}",
            "top_5_individuals:",
        ]
        for r in unique:
            lines.append(f"  {r.commit}  {r.val_bpb:.6f}  {r.description}")
        succeeded = [r for r in rows if r.method not in ("baseline", "elite") and r.viable]
        failed = [r for r in rows if r.status != "ok"]
        lines.append(f"narrative: {len(succeeded)} viable offspring and {len(failed)} failed candidates. "
                     "Inspect the ranked individuals and generation dashboard for research ideas.")
        (self.root / "summary.txt").write_text("\n".join(lines) + "\n")
        if best:
            git(self.root, "reset", "--hard", git(self.root, "rev-parse", best.commit))
            self._check_prepare()

    def run(self, max_generations: int = 0, max_hours: float = 0) -> str:
        """Run until a stopping condition; max_generations=0 is unlimited."""
        with RunLock(self.root / "run.lock"):
            self._require_clean()
            git(self.root, "switch", f"autoresearch/{self.state['tag']}")
            self._reconcile_results()
            started = time.monotonic()
            old_int, old_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
            def interrupt(signum, frame):
                _ = (signum, frame)
                self.stop = True
            signal.signal(signal.SIGINT, interrupt)
            signal.signal(signal.SIGTERM, interrupt)
            reason = "interrupted"
            try:
                while not self.stop:
                    if max_hours and (time.monotonic() - started) >= max_hours * 3600:
                        reason = "time_limit"
                        break
                    rows = read_population(self.root / "population.tsv")
                    n = self.config["population_size"]
                    partial = [g for g in sorted({x.gen for x in rows}) if sum(r.gen == g for r in rows) < n]
                    generation = partial[0] if partial else (max((r.gen for r in rows), default=-1) + 1)
                    if max_generations and generation >= max_generations:
                        reason = "generation_limit"
                        break
                    present = {r.slot: r for r in rows if r.gen == generation}
                    if generation == 0:
                        if 0 not in present:
                            baseline = self.state["baseline_commit"]
                            git(self.root, "reset", "--hard", baseline)
                            self._pin_commit(baseline)
                            passed, error = self._preflight()
                            if not passed:
                                raise RuntimeError(f"Baseline preflight failed:\n{error}")
                            status, value, mem, _ = self._evaluate(baseline)
                            if status != "ok":
                                raise RuntimeError("Baseline training failed. Inspect run.log")
                            self._record(Individual(0, 0, baseline[:7], status, value, mem,
                                                    "", "", "baseline", "unmodified baseline"))
                        parents = [r for r in read_population(self.root / "population.tsv") if r.gen == 0 and r.slot == 0]
                    else:
                        previous = [r for r in rows if r.gen == generation - 1]
                        if len(previous) != n:
                            raise RuntimeError("Previous generation is incomplete")
                        ranked = self._rank(previous)
                        if not ranked:
                            reason = "all_candidates_failed"
                            break
                        for slot, elite in enumerate(ranked[:self.config["top_k"]]):
                            if slot not in present:
                                self._record(Individual(generation, slot, elite.commit, "ok", elite.val_bpb,
                                                        elite.memory_gb, elite.commit, "", "elite", "carried forward"))
                        parents = previous
                    for slot in range(n):
                        if self.stop:
                            raise StopRequested()
                        if max_hours and time.monotonic() - started >= max_hours * 3600:
                            raise StopRequested()
                        present = {r.slot: r for r in read_population(self.root / "population.tsv") if r.gen == generation}
                        if slot in present:
                            continue
                        history = read_population(self.root / "population.tsv")
                        offspring = self._create_offspring(generation, slot, parents, history)
                        self._record(offspring)
                    self._tag_generation(generation)
                    self._checkpoint(generation)
                    try:
                        from .dashboard import build_dashboard
                        build_dashboard(self.root)
                    except Exception as exc:
                        print(f"Dashboard generation warning: {exc}", file=sys.stderr)
                    reason = self._stop_reason() or ""
                    if reason:
                        break
            except StopRequested:
                reason = "interrupted"
            finally:
                signal.signal(signal.SIGINT, old_int)
                signal.signal(signal.SIGTERM, old_term)
                self._reconcile_results()
                rows = read_population(self.root / "population.tsv")
                if rows:
                    self._checkpoint(max(r.gen for r in rows))
                self._summary(reason or "interrupted")
            if reason == "convergence_plateau":
                tag = f"autoresearch/{self.state['tag']}/converged"
                if not git(self.root, "tag", "--list", tag):
                    git(self.root, "tag", tag, "HEAD")
            return reason


def bootstrap_repository(root: Path) -> None:
    """Make extracted ZIP Git-ready, without touching an existing repository."""
    if not (root / ".git").exists():
        git(root, "init", "-b", "master")
        if not git(root, "config", "user.name", check=False):
            git(root, "config", "user.name", "Autoresearch Evolution")
        if not git(root, "config", "user.email", check=False):
            git(root, "config", "user.email", "autoresearch@local.invalid")
        git(root, "add", ".")
        git(root, "commit", "-m", "Initial autoresearch evolution baseline")


def initialize(root: Path, tag: str, config_path: Path,
               allow_missing_data: bool = False, allow_no_gpu: bool = False) -> dict:
    root = root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,38}", tag):
        raise ValueError("Tag must contain only letters numbers dot underscore or dash")
    if (root / "state.json").exists():
        raise RuntimeError("Existing experiment state.json found; use run to resume")
    config = json.loads(config_path.read_text())
    if (not 2 <= config["population_size"] <= 64 or
            not 1 <= config["top_k"] < config["population_size"] or
            not 1 <= config["tournament_size"] <= config["population_size"] or
            not 0 <= config["crossover_rate"] <= 1):
        raise ValueError("Invalid evolution population configuration")
    cache_sha256 = {}
    if not allow_missing_data:
        # Snapshot all prepared benchmark artifacts for comparability and detect
        # accidental data-cache tampering across candidate evaluations.
        cache = Path.home() / ".cache" / "autoresearch"
        names = ("train_tokens.npy", "val_tokens.npy", "manifest.json",
                 "tokenizer/gpt2_token_bytes.npy", "tokenizer/tokenizer.json")
        missing = [p for p in names if not (cache / p).exists()]
        if missing:
            raise RuntimeError("Missing prepared data: " + ", ".join(missing) + ". Run uv run prepare.py")
        manifest = json.loads((cache / "manifest.json").read_text())
        if manifest.get("benchmark") != "climbmix-gpt2-v1":
            raise RuntimeError("Incompatible data manifest benchmark")
        for split in ("train", "val"):
            digest = sha256(cache / f"{split}_tokens.npy")
            if digest != manifest[split]["sha256"]:
                raise RuntimeError(f"Cached {split} tokens do not match manifest SHA-256")
        cache_sha256 = {name: sha256(cache / name) for name in names}
    if not allow_no_gpu:
        result = subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available()"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("No CUDA GPU detected. Run on a single NVIDIA GPU or use --allow-no-gpu to initialize only.")
    bootstrap_repository(root)
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Commit or restore tracked modifications before init")
    if git(root, "branch", "--list", f"autoresearch/{tag}"):
        raise RuntimeError(f"Branch autoresearch/{tag} already exists")
    if not git(root, "branch", "--list", "master"):
        raise RuntimeError("Required base branch master does not exist")
    git(root, "switch", "master")
    baseline = git(root, "rev-parse", "HEAD")
    git(root, "switch", "-c", f"autoresearch/{tag}")
    result_file, population_file = root / "results.tsv", root / "population.tsv"
    _write_header(result_file, RESULT_HEADERS)
    _write_header(population_file, POP_HEADERS)
    write_environment(root)
    state = {
        "tag": tag, "baseline_commit": baseline,
        "prepare_sha256": sha256(root / "prepare.py"),
        "cache_sha256": cache_sha256,
        "config": config,
        "created_utc": dt.datetime.now(UTC).isoformat(),
        "last_checkpoint_generation": -1,
    }
    (root / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(f"Initialized autoresearch/{tag} from {baseline[:7]}")
    return state
