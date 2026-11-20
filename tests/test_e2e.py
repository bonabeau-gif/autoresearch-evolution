"""End-to-end orchestration tests that create *real Git commits* without GPU/API."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from autoresearch_evo.core import Evolution, bootstrap_repository, git, initialize, read_population


FAKE_TRAIN = '''"""Fast fake trainer for integration tests only: NOT a real benchmark."""
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.1
DEPTH = 6
WIDTH = 384
NUM_HEADS = 6
BATCH_SIZE = 8
GRAD_ACCUM = 2
DROPOUT = 0.0
if __name__ == "__main__":
    print("---")
    print("val_bpb:          %.6f" % (1.5 - (DEPTH - 6)*0.01 + LEARNING_RATE * 10 + WEIGHT_DECAY * 0.01))
    print("training_seconds: 300.0")
    print("peak_vram_mb:     4096.0")
'''


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "config").mkdir()
        (self.root / "train.py").write_text(FAKE_TRAIN)
        (self.root / "prepare.py").write_text("# immutable evaluator fixture\n")
        (self.root / ".gitignore").write_text("state.json\nresults.tsv\npopulation.tsv\ncheckpoint.txt\nsummary.txt\nenvironment.txt\nrun.log\nrun.lock\nruns/\nreports/\n")
        config = {
            "population_size": 3, "top_k": 1, "tournament_size": 2, "crossover_rate": 0.3,
            "seed": 42, "tie_epsilon": 0.000001, "training_timeout_seconds": 10,
            "preflight_timeout_seconds": 5, "max_preflight_fixes": 2,
            "plateau_generations": 10, "catastrophic_generations": 3,
        }
        (self.root / "config" / "evolution.json").write_text(json.dumps(config))
        initialize(self.root, "test-run", self.root / "config" / "evolution.json", True, True)

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self):
        return Evolution(self.root, provider="mock", train_command="python train.py",
                         smoke_command="python -c 'import train'")

    def test_generations_and_resume(self):
        a = self.runner()
        self.assertEqual(a.run(max_generations=1), "generation_limit")
        rows = read_population(self.root / "population.tsv")
        self.assertEqual(len(rows), 3)
        self.assertEqual([x.method for x in rows][:1], ["baseline"])
        self.assertEqual(len((self.root / "results.tsv").read_text().splitlines()), 4)
        self.assertEqual(a.run(max_generations=2), "generation_limit")
        rows = read_population(self.root / "population.tsv")
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(x.method == "elite" for x in rows), 1)
        self.assertEqual(sum(x.method != "elite" for x in rows), 5)
        self.assertEqual(len((self.root / "results.tsv").read_text().splitlines()), 6)
        self.assertTrue((self.root / "reports" / "progress.html").exists())
        self.assertTrue((self.root / "checkpoint.txt").exists())
        self.assertTrue((self.root / "summary.txt").exists())
        self.assertEqual(git(self.root, "branch", "--show-current"), "autoresearch/test-run")
        for row in rows:
            self.assertTrue(git(self.root, "cat-file", "-e", row.commit + "^{commit}", check=False) == "")
        self.assertEqual(len([r for r in rows if r.status == "ok"]), 6)

    def test_evaluator_integrity(self):
        (self.root / "prepare.py").write_text("# changed benchmark\n")
        with self.assertRaisesRegex(RuntimeError, "checksum changed"):
            self.runner()

    def test_duplicate_branch_tag_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Existing experiment state"):
            initialize(self.root, "test-run", self.root / "config" / "evolution.json", True, True)


if __name__ == "__main__":
    unittest.main()
