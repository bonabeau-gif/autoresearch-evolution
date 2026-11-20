from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from autoresearch_evo.core import (
    Individual, POP_HEADERS, RESULT_HEADERS, _append_row, _write_header,
    mutation_intensity, parse_metrics, read_population, tournament,
)
from autoresearch_evo.dashboard import build_dashboard
from autoresearch_evo.llm import MockProvider, clean_description, strip_code_fences


class DataTests(unittest.TestCase):
    def example(self, gen=0, slot=0, val=1.0, status="ok", commit="abc1234", method="baseline"):
        return Individual(gen, slot, commit, status, val if status == "ok" else None,
                          2.1 if status == "ok" else None, "", "", method, "test")

    def test_metrics_parse_and_nonfinite(self):
        raw = "---\nval_bpb: 1.022200\ntraining_seconds: 300.2\npeak_vram_mb: 45060.2\n"
        val, mem = parse_metrics(raw)
        self.assertAlmostEqual(val, 1.0222)
        self.assertAlmostEqual(mem, 44.0)
        for text in ["val_bpb: nan\npeak_vram_mb: 2", "val_bpb: 1\n", "val_bpb: -1\npeak_vram_mb: 2"]:
            with self.assertRaises(ValueError):
                parse_metrics(text)

    def test_tsv_headers_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "population.tsv"
            _write_header(p, POP_HEADERS)
            row = self.example()
            _append_row(p, POP_HEADERS, row.pop_row())
            self.assertEqual(read_population(p), [row])
            self.assertEqual(p.read_text().splitlines()[0].split("\t"), list(POP_HEADERS))
            self.assertEqual(len(p.read_text().splitlines()[1].split("\t")), 10)

    def test_failed_metrics_must_be_blank(self):
        with self.assertRaises(ValueError):
            Individual(0, 1, "abcd123", "crash", 1.0, 2.2, "", "", "mutation", "no")
        self.assertEqual(self.example(status="crash").pop_row()["val_bpb"], "")

    def test_tournament_excludes_crashes_and_prefers_low_fitness(self):
        rng = random.Random(44)
        rows = [self.example(0, 0, 3.0, commit="bad0000"),
                self.example(0, 1, 0.9, commit="good000"),
                self.example(0, 2, 0.0, status="crash", commit="dead000")]
        self.assertEqual(tournament(rows, 3, rng).commit, "good000")

    def test_mutation_intensity_coverage(self):
        g0 = [mutation_intensity(0, i, 6, 2) for i in range(1, 6)]
        g1 = [mutation_intensity(1, i, 6, 2) for i in range(2, 6)]
        self.assertEqual(len(g0), 5)
        self.assertIn("radical", g0)
        self.assertIn("radical", g1)
        self.assertIn("conservative", g1)
        self.assertIn("moderate", g1)

    def test_mock_changes_only_a_numeric_setting(self):
        src = 'LEARNING_RATE = 0.01\nDEPTH = 6\nWIDTH = 384\nNUM_HEADS = 6\n'
        rng = random.Random(9)
        result = MockProvider(rng).propose([src], "mutation", "moderate", "")
        self.assertNotEqual(result.source, src)
        compile(result.source, "train.py", "exec")
        self.assertNotIn(",", result.description)
        self.assertEqual(strip_code_fences('```python\n' + src + '```'), src)
        self.assertEqual(clean_description("one,two\tthree"), "one two three")

    def test_dashboard_escapes_injected_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reports").mkdir()
            p = root / "population.tsv"
            _write_header(p, POP_HEADERS)
            malicious = "<script>alert(1)</script>"
            _append_row(p, POP_HEADERS, self.example().__class__(0, 0, "abc1234", "ok", 1.0, 1.2,
                                                                   "", "", "baseline", malicious).pop_row())
            result = build_dashboard(root).read_text()
            self.assertNotIn("<script>alert(1)</script>", result)
            self.assertIn("\\u003cscript>", result)
            self.assertIn('const rows=[', result)


if __name__ == "__main__":
    unittest.main()
