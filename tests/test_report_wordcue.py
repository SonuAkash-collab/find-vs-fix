"""Phase 5 report: the pre-registered STOP is stated (offline, from the committed result files; no model, no GPU)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import report_wordcue as rw

STOP = ("**STOP (pre-registered): shortcut not learned at 5% (gap 0.013) or after the one 10% fallback (gap 0.030); "
        "later phases not run. A valid result.**")


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def checks(prevalence, gap, learned):
    return {"run": "wordcue_shortcut_seed0", "prevalence": prevalence, "gap_none_minus_opposite": gap,
            "checks": {"shortcut_learned": {"pass": learned, "value": gap}}}


class StopLine(unittest.TestCase):
    def test_committed_results_give_the_stop_line(self):
        man = rw.load(ROOT / "data_wordcue" / "manifest.json")
        self.assertEqual(rw.stop_line(man, ROOT / "results"), STOP)

    def test_report_states_the_stop_at_the_top_of_the_summary(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "RESULTS_WORDCUE.md"
            with mock.patch.object(sys, "argv", ["report_wordcue.py", "--out", str(out)]):
                rw.main()
            text = out.read_text(encoding="utf-8")
        lines = text.splitlines()
        i = lines.index("## Summary: does the word cue reproduce the name shortcut?")
        self.assertEqual(lines[i + 2], STOP)
        self.assertIn(rw.NOT_RUN, text)
        self.assertNotIn(rw.INCOMPLETE, text)
        self.assertIn("c3851d1e9fa41de80b8c597678a23606f730fe8a, e877c9368d072ca169bb6464af4bef7d0cfb5341", text)
        self.assertIn("Recorded wall time: 0.64 h", text)

    def test_no_stop_without_fallback_or_when_learned(self):
        with tempfile.TemporaryDirectory() as d:
            R = Path(d)
            write(R / "wordcue" / "checks" / "wordcue_shortcut_seed0.json", checks(0.05, 0.0125, False))
            self.assertIsNone(rw.stop_line({"prevalence": 0.05}, R))  # first failure: the fallback comes next, no STOP
            write(R / "wordcue" / "checks" / "wordcue_shortcut_seed0.json", checks(0.1, 0.35, True))
            self.assertIsNone(rw.stop_line({"prevalence": 0.1, "fallback": "raised"}, R))  # fallback worked
            write(R / "wordcue" / "checks" / "wordcue_shortcut_seed0.json", checks(0.1, 0.03, False))
            write(R / "wordcue" / "fallback_5pct" / "wordcue_shortcut_seed0.json", checks(0.05, 0.0125, False))
            write(R / "wordcue" / "fallback_5pct" / "manifest_5pct.json", {"prevalence": 0.05})
            self.assertEqual(rw.stop_line({"prevalence": 0.1, "fallback": "raised"}, R), STOP)

    def test_compare_labels(self):
        self.assertEqual(rw.compare(None, "PASS"), rw.INCOMPLETE)
        self.assertEqual(rw.compare(None, "PASS", stopped=True), rw.NOT_RUN)
        self.assertEqual(rw.compare("PASS", "PASS", stopped=True), "**reproduces**")
        self.assertEqual(rw.compare("FAIL", "PASS"), "**doesn't reproduce**")


if __name__ == "__main__":
    unittest.main()
