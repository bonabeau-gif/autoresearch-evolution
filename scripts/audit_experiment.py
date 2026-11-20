#!/usr/bin/env python3
"""Verify a saved experiment's TSV/commit integrity without a GPU.

Usage: python scripts/audit_experiment.py [repo-root]
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    if not (root / "state.json").exists():
        raise SystemExit("No state.json: experiment not initialized")
    state = json.loads((root / "state.json").read_text())
    with open(root / "population.tsv", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    with open(root / "results.tsv", newline="") as f:
        results = list(csv.DictReader(f, delimiter="\t"))
    errors = []
    seen = set()
    commits = {}
    for r in rows:
        key = (r["gen"], r["slot"])
        if key in seen:
            errors.append(f"Duplicate population slot: {key}")
        seen.add(key)
        if r["status"] != "ok" and (r["val_bpb"] or r["memory_gb"]):
            errors.append(f"Failed individual has metrics: {r['commit']}")
        if r["status"] == "ok" and (not r["val_bpb"] or not r["memory_gb"]):
            errors.append(f"Successful individual lacks metrics: {r['commit']}")
        commits[r["commit"]] = True
    results_sha = {r["commit"] for r in results}
    for r in rows:
        if r["method"] != "elite" and r["commit"] not in results_sha:
            errors.append(f"Candidate missing from results.tsv: {r['commit']}")
    if len(results_sha) != len(results):
        errors.append("Duplicate commit in results.tsv")
    for sha in commits:
        check = subprocess.run(["git", "-C", str(root), "cat-file", "-e", sha + "^{commit}"],
                               capture_output=True)
        if check.returncode:
            errors.append(f"Missing Git commit: {sha}")
    if state.get("tag"):
        expected = f"autoresearch/{state['tag']}"
        branch = subprocess.run(["git", "-C", str(root), "branch", "--show-current"],
                                capture_output=True, text=True).stdout.strip()
        if branch != expected:
            errors.append(f"Current branch {branch!r} differs from {expected!r}")
    print(f"Audit: {len(rows)} population slots, {len(results)} evaluations, {len(commits)} unique commits")
    if errors:
        print("FAILED:\n" + "\n".join(errors))
        return 1
    print("PASS: logged commits and schema are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
