"""Phase L6: offline checks (committed manifests, synthetic results, the driver with stubbed steps). No GPU.
Run: python -m unittest tests/test_weak.py"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import make_weak as MW  # noqa: E402
import report_lora as RP  # noqa: E402
import run_lora as RL  # noqa: E402
from common import DATA, data_for, resolve_data  # noqa: E402

LEVELS = (90, 80, 70, 60)
HEALTHY = [{"step": st, "loss": 0.69 if st == 50 else 0.3, "grad_norm_max": 2.5} for st in range(50, 1901, 50)]
COLLAPSE = [{"step": st, "loss": 0.42 if st <= 200 else 0.69, "grad_norm_max": 9.0} for st in range(50, 1901, 50)]


class Data(unittest.TestCase):
    def man(self, p):
        return json.load(open(ROOT / f"data_weak_p{p}" / "manifest.json"))

    def test_manifests_counts_and_nesting(self):
        flips = {}
        for p in LEVELS:
            m = self.man(p)
            k = MW.LEVELS[p]
            self.assertEqual(m["prefix"], f"lora_r64_weakp{p}")
            self.assertEqual(m["counts"]["flipped"], 2 * k)
            self.assertEqual(m["counts"]["flipped"], round(500 * (100 - p) / 100))
            self.assertEqual((m["counts"]["name_A"], m["counts"]["name_B"]), (250, 250))
            self.assertEqual(m["counts"]["name_matches_label"], 500 - 2 * k)
            flips[p] = set(m["flipped_ids"])
        self.assertTrue(flips[90] < flips[80] < flips[70] < flips[60])

    def test_copies_have_the_data_sha256(self):
        want = json.load(open(DATA / "manifest.json"))["sha256"]
        for p in LEVELS:
            got = self.man(p)["sha256"]
            for f in MW.COPIED:
                self.assertEqual(got[f], want[f], (p, f))
            self.assertNotEqual(got["train_shortcut.jsonl"], want["train_shortcut.jsonl"])

    @unittest.skipUnless((DATA / "train_shortcut.jsonl").exists(), "data/ not built here")
    def test_draw_gives_the_recorded_flips(self):
        rows = MW.load(DATA / "train_shortcut.jsonl")
        perm = MW.draw(rows)
        for p, k in MW.LEVELS.items():
            ids = sorted(rows[i]["id"] for lab in (0, 1) for i in perm[lab][:k])
            self.assertEqual(ids, self.man(p)["flipped_ids"])
        self.assertEqual({rows[i]["label"] for i in perm[0]}, {0})
        self.assertEqual({rows[i]["label"] for i in perm[1]}, {1})

    def test_each_family_uses_its_folder_and_others_use_data(self):
        for p in LEVELS:
            self.assertEqual(data_for(f"lora_r64_weakp{p}"), ROOT / f"data_weak_p{p}")
            self.assertEqual(resolve_data(None, f"lora_r64_weakp{p}"), ROOT / f"data_weak_p{p}")
        for pre in ("lora_r64", "lora_r64_early", "lora_r64_init0", "lora_r4", None):
            self.assertEqual(data_for(pre), DATA)
        with self.assertRaises(SystemExit):  # a weak folder is never used under another family
            resolve_data("data_weak_p90", "lora_r64")


def rec(gap, acc=0.9, task=True, learned=True, loss_log=HEALTHY, data_dir="x/data_weak_p{p}", kind="shortcut", p=90):
    c = {"task_learned": {"pass": task}}
    c["shortcut_learned" if kind == "shortcut" else "clean_no_shortcut"] = {"pass": learned}
    cfg = {"n_train": 10000, "kind": kind}
    if data_dir:
        cfg["data_dir"] = data_dir.format(p=p)
    return {"accuracy": {"none": acc, "match": 1.0, "opposite": acc - gap, "neutral": acc}, "gap_none_minus_opposite": gap,
            "checks": c, "all_pass": task and learned, "config": cfg,
            "train": {"loss_log": loss_log, "final_loss_mean_last50": 0.3, "wall_time_s": 300.0,
                      "peak_vram_allocated_gib": 2.0}}


class Driver(unittest.TestCase):
    """weak() with training, checks and every script stubbed: order, choice, collapse rule, slots, stops."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base)
        self.res, self.runs = self.base / "results", self.base / "runs"
        self.steps, self.trained = [], []
        self.pilot = {90: 0.60, 80: 0.42, 70: 0.30, 60: 0.10}  # seed-0 gap per level
        self.main_gap, self.bad = 0.3, {}  # bad: {run name: record kwargs}
        for name, v in (("RESULTS", self.res), ("RUNS", self.runs), ("L6_DIR", self.res / "lora_weak"),
                        ("log", lambda m: None), ("step", self.fake_step), ("train_and_check", self.fake_train),
                        ("add_env_session", self.fake_env)):
            x = patch.object(RL, name, v)
            x.start()
            self.addCleanup(x.stop)
        (self.res / "lora_weak").mkdir(parents=True)
        json.dump({"pass": True}, open(self.res / "lora_weak" / "smoke.json", "w"))
        (self.res / "lora_r64" / "checks").mkdir(parents=True)
        for s in (0, 1, 2):  # the main r64 clean twins: a run folder and a checks file
            d = self.runs / f"lora_r64_clean_seed{s}"
            (d / "model").mkdir(parents=True)
            (d / "model" / "model.safetensors").write_bytes(bytes([s]) * 64)
            json.dump({"n_train": 10000, "kind": "clean"}, open(d / "config.json", "w"))
            json.dump(rec(0.01, kind="clean", data_dir=None), open(self.res / "lora_r64" / "checks" /
                                                                     f"lora_r64_clean_seed{s}.json", "w"))

    def fake_env(self, out, name, command):
        out.mkdir(parents=True, exist_ok=True)
        json.dump({"sessions": [{"git_commit": "abc", "gpu": "L4"}]}, open(out / "env.json", "w"))

    def fake_train(self, runs, label, out_dir_of, extra):
        self.trained.append((label, [r[0] for r in runs], extra(runs[0][0])))
        out = out_dir_of(64)
        out.mkdir(parents=True, exist_ok=True)
        for name, _, _, kind in runs:
            f = out / f"{name}.json"
            if f.exists():
                continue
            if name == RL.L6_CLEAN3:
                r = rec(0.01, kind="clean", data_dir=None)
                d = self.runs / name
                (d / "model").mkdir(parents=True)
                (d / "model" / "model.safetensors").write_bytes(b"3" * 64)
                json.dump({"n_train": 10000, "kind": "clean"}, open(d / "config.json", "w"))
            else:
                p, s = int(name.split("weakp")[1][:2]), int(name.rsplit("seed", 1)[1])
                r = rec(self.pilot[p] if s == 0 else self.main_gap, p=p)
            if name in self.bad:
                r = rec(**{"gap": 0.4, "p": int(name.split("weakp")[1][:2]), **self.bad[name]}) \
                    if name != RL.L6_CLEAN3 else rec(0.01, kind="clean", data_dir=None, **self.bad[name])
            json.dump(r, open(f, "w"))

    def fake_step(self, label, args, capture=False):
        self.steps.append(args)
        if args[0] == "retrain.py":
            pre = args[args.index("--prefix") + 1]
            seeds = [int(x) for x in args[args.index("--seeds") + 1].split(",")]
            out = self.res / pre / "retrain"
            out.mkdir(parents=True, exist_ok=True)
            p = int(pre.split("weakp")[1])
            for s in seeds:
                base = rec(0.3, p=p)
                runs = {n: {**rec(self.gate_gap if n == "all" else 0.1, p=p), "frozen_params_unchanged": True,
                            "wall_time_s_incl_eval": 60.0} for n in ("all", "9-10-11")}
                if "--sets" not in args:
                    runs.update({n: {**rec(0.1, p=p), "frozen_params_unchanged": True, "wall_time_s_incl_eval": 60.0}
                                 for n, _, _ in __import__("retrain").RUN_SETS if n not in runs})
                f = out / f"seed{s}.json"
                old = json.load(open(f)) if f.exists() else {"runs": {}}
                json.dump({"seed": s, "start_model": f"{pre}_shortcut_seed{s}", "baseline": base,
                           "runs": {**old["runs"], **runs}}, open(f, "w"))

    gate_gap = 0.1

    def go(self):
        return RL.weak()

    def state(self):
        return json.load(open(self.res / "lora_weak" / "state.json"))

    def test_order_choice_and_analyses(self):
        self.assertTrue(self.go())
        ch = json.load(open(self.res / "lora_weak" / "choice.json"))
        self.assertEqual(ch["chosen"], 70)  # 0.300 is closest to 0.281; 0.420 qualifies, 0.600 and 0.100 do not
        self.assertEqual([x[1] for x in self.trained[:4]], [[f"lora_r64_weakp{p}_shortcut_seed0"] for p in LEVELS])
        self.assertEqual(self.trained[4][1], ["lora_r64_weakp70_shortcut_seed1", "lora_r64_weakp70_shortcut_seed2"])
        self.assertEqual(self.trained[0][2], ["--log-grad-norm"])
        self.assertEqual(self.steps[0], ["make_weak.py", "--verify"])
        self.assertIn(["retrain.py", "--prefix", "lora_r64_weakp70", "--seeds", "0", "--sets", "all,9-10-11"], self.steps)
        self.assertIn(["retrain.py", "--prefix", "lora_r64_weakp70", "--seeds", "0,1,2"], self.steps)
        self.assertIn(["logit_lens.py", "--prefix", "lora_r64_weakp70"], self.steps)
        self.assertIn(["swap.py", "--prefix", "lora_r64_weakp70"], self.steps)
        self.assertFalse(any("--phase2b" in a for a in self.steps))  # Phase 2b is not run
        self.assertEqual(self.steps[-1], ["report_lora.py"])
        for s in (0, 1, 2):  # twins copied byte for byte, with a checks record
            self.assertEqual((self.runs / f"lora_r64_weakp70_clean_seed{s}" / "model" / "model.safetensors").read_bytes(),
                             bytes([s]) * 64)
            self.assertTrue((self.res / "lora_r64_weakp70" / "checks" / f"lora_r64_weakp70_clean_seed{s}.json").exists())
        self.assertEqual(self.state()["slot_seeds"], [0, 1, 2])
        self.assertIsNone(self.state()["replacement"])

    def test_tie_goes_to_the_higher_reliability(self):
        self.pilot = {90: 0.300, 80: 0.262, 70: 0.10, 60: 0.10}  # both 0.019 from 0.281
        self.go()
        self.assertEqual(json.load(open(self.res / "lora_weak" / "choice.json"))["chosen"], 90)

    def test_gap_range_ends_are_included(self):
        self.pilot = {90: 0.60, 80: 0.450, 70: 0.60, 60: 0.10}
        self.go()
        self.assertEqual(json.load(open(self.res / "lora_weak" / "choice.json"))["chosen"], 80)

    def test_acc_none_under_0_80_does_not_qualify(self):
        self.bad = {"lora_r64_weakp70_shortcut_seed0": {"gap": 0.30, "acc": 0.795, "task": False}}
        self.go()
        self.assertEqual(json.load(open(self.res / "lora_weak" / "choice.json"))["chosen"], 80)

    def test_no_level_qualifies_stops_with_the_report(self):
        self.pilot = {90: 0.60, 80: 0.50, 70: 0.15, 60: 0.10}
        with self.assertRaises(SystemExit) as c:
            self.go()
        self.assertEqual(c.exception.code, 2)
        self.assertEqual(self.steps[-1], ["report_lora.py"])
        self.assertEqual(len(self.trained), 4)

    def test_pilot_collapse_at_the_chosen_level_puts_seed3_in_slot0(self):
        self.bad = {"lora_r64_weakp70_shortcut_seed0": {"gap": 0.40, "acc": 0.77, "task": False, "loss_log": COLLAPSE}}
        self.pilot[70] = 0.30  # the replacement (seed 3) is what the choice reads
        self.go()
        st = self.state()
        self.assertEqual(st["replacement"]["collapsed_run"], "lora_r64_weakp70_shortcut_seed0")
        ch = json.load(open(self.res / "lora_weak" / "choice.json"))
        self.assertEqual(ch["levels"]["70"]["run"], "lora_r64_weakp70_shortcut_seed3")
        self.assertEqual(ch["chosen"], 70)
        self.assertEqual(st["slot_seeds"], [3, 1, 2])
        self.assertIn([RL.L6_CLEAN3], [x[1] for x in self.trained])
        self.assertIn(["retrain.py", "--prefix", "lora_r64_weakp70", "--seeds", "3", "--sets", "all,9-10-11"], self.steps)
        self.assertIn(["swap.py", "--prefix", "lora_r64_weakp70", "--slot-seeds", "3,1,2"], self.steps)
        self.assertIn(["logit_lens.py", "--prefix", "lora_r64_weakp70", "--run", "clean_seed3"], self.steps)
        self.assertEqual((self.runs / "lora_r64_weakp70_clean_seed3" / "model" / "model.safetensors").read_bytes(), b"3" * 64)

    def test_a_main_collapse_after_a_pilot_collapse_stops(self):
        # p90's seed-3 replacement has gap 0.300, tied with p70 at 0.019 from 0.281 -> p90 (higher reliability)
        self.bad = {"lora_r64_weakp90_shortcut_seed0": {"acc": 0.77, "task": False, "loss_log": COLLAPSE},
                    "lora_r64_weakp90_shortcut_seed2": {"acc": 0.77, "task": False, "loss_log": COLLAPSE}}
        with self.assertRaises(SystemExit) as c:
            self.go()
        self.assertEqual(c.exception.code, 2)
        self.assertEqual(json.load(open(self.res / "lora_weak" / "choice.json"))["chosen"], 90)
        self.assertIn(["lora_r64_weakp90_shortcut_seed1", "lora_r64_weakp90_shortcut_seed2"], [x[1] for x in self.trained])
        self.assertFalse(any(a[0] == "retrain.py" for a in self.steps))

    def test_a_main_collapse_is_replaced(self):
        self.bad = {"lora_r64_weakp70_shortcut_seed1": {"acc": 0.77, "task": False, "loss_log": COLLAPSE}}
        self.assertTrue(self.go())
        self.assertEqual(self.state()["slot_seeds"], [0, 3, 2])

    def test_a_collapsed_replacement_twin_stops(self):
        self.bad = {"lora_r64_weakp70_shortcut_seed1": {"acc": 0.77, "task": False, "loss_log": COLLAPSE},
                    RL.L6_CLEAN3: {"task": False, "loss_log": COLLAPSE}}
        with self.assertRaises(SystemExit) as c:
            self.go()
        self.assertEqual(c.exception.code, 2)

    def test_a_main_run_without_a_shortcut_stops(self):
        self.bad = {"lora_r64_weakp70_shortcut_seed2": {"gap": 0.15, "learned": False}}
        with self.assertRaises(SystemExit) as c:
            self.go()
        self.assertEqual(c.exception.code, 2)
        self.assertFalse(any(a[0] == "retrain.py" for a in self.steps))

    def test_a_run_on_the_wrong_data_stops(self):
        self.bad = {"lora_r64_weakp80_shortcut_seed0": {"data_dir": "x/data"}}
        with self.assertRaises(SystemExit) as c:
            self.go()
        self.assertEqual(c.exception.code, 2)

    def test_high_loss_but_passing_is_flagged(self):
        self.bad = {"lora_r64_weakp70_shortcut_seed2": {"gap": 0.30, "loss_log": COLLAPSE}}
        self.assertTrue(self.go())
        self.assertEqual(self.state()["flagged"], ["lora_r64_weakp70_shortcut_seed2"])
        self.assertIsNone(self.state()["replacement"])

    def test_phase3_gate_failure_stops_phase3_only(self):
        self.gate_gap = 0.2  # > 50% of 0.3
        self.assertIsNone(self.go())
        self.assertNotIn(["retrain.py", "--prefix", "lora_r64_weakp70", "--seeds", "0,1,2"], self.steps)
        self.assertEqual(self.steps[-1], ["report_lora.py"])

    def test_resume_runs_nothing_again(self):
        self.go()
        n = len(self.steps)
        self.go()
        self.assertEqual([a[0] for a in self.steps[n:]], ["make_weak.py", "report_lora.py"])


class Section(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base)
        w = self.base / "lora_weak"
        w.mkdir()
        levels = {str(p): {"run": f"lora_r64_weakp{p}_shortcut_seed0", "seed": 0, "acc_none": 0.87, "gap": g,
                           "gap_3dp": g, "qualifies": q, "distance_to_0.281": abs(g - 0.281) if q else None,
                           "shortcut_learned": g >= 0.2, "task_learned": True}
                  for p, g, q in ((90, 0.6, False), (80, 0.42, True), (70, 0.3, True), (60, 0.1, False))}
        json.dump({"levels": levels, "chosen": 70}, open(w / "choice.json", "w"))
        json.dump({"replacement": None, "flagged": [], "chosen": 70, "slot_seeds": [0, 1, 2]}, open(w / "state.json", "w"))
        pre = "lora_r64_weakp70"
        for sub in ("checks", "retrain", "logit_lens", "swap"):
            (self.base / pre / sub).mkdir(parents=True)
        for r in (4, 64):
            (self.base / f"lora_r{r}" / "checks").mkdir(parents=True)
            (self.base / f"lora_r{r}" / "retrain").mkdir(parents=True)
        for s in (0, 1, 2):
            json.dump(rec(0.3), open(self.base / pre / "checks" / f"{pre}_shortcut_seed{s}.json", "w"))
            json.dump(rec(0.01, kind="clean"), open(self.base / pre / "checks" / f"{pre}_clean_seed{s}.json", "w"))
            for r, g in ((4, 0.28), (64, 0.8)):
                json.dump(rec(g), open(self.base / f"lora_r{r}" / "checks" / f"lora_r{r}_shortcut_seed{s}.json", "w"))
            # blocks 1-3 keep the gap in every seed (fail); block 4 fails by the accuracy drop in seed 2 only
            runs = {str(b): rec(0.25 if b in (1, 2, 3) else 0.05, acc=0.85 if (b == 4 and s == 2) else 0.9)
                    for b in range(12)}
            json.dump({"baseline": rec(0.3), "runs": runs}, open(self.base / pre / "retrain" / f"seed{s}.json", "w"))
            r4 = {str(b): rec(0.25 if b in (1, 2, 3, 4, 5, 11) else 0.05) for b in range(12)}
            json.dump({"baseline": rec(0.3), "runs": r4}, open(self.base / "lora_r4" / "retrain" / f"seed{s}.json", "w"))
            json.dump({"baseline": rec(0.8), "runs": {str(b): rec(0.1) for b in range(12)}},
                      open(self.base / "lora_r64" / "retrain" / f"seed{s}.json", "w"))
            json.dump({"cal_acc": {"none": [0.9] * 13, "opposite": [0.9] * 11 + [0.6, 0.6]}},
                      open(self.base / pre / "logit_lens" / f"{pre}_shortcut_seed{s}.json", "w"))
            json.dump({"baseline": {f"{pre}_shortcut_seed{s}": rec(0.3)},
                       "cases": {RP.A: {"by_blocks": {"9-10-11": rec(0.03)}}, RP.CII: {"by_blocks": {"9-10-11": rec(0.27)}}}},
                      open(self.base / pre / "swap" / f"seed{s}.json", "w"))
        json.dump({"seed": 0, "pass": True, "gap_after_all": 0.05, "limit": 0.15}, open(self.base / pre / "retrain" /
                                                                                         "gate.json", "w"))

    def text(self):
        o = []
        RP.weak_section(o, self.base)
        return "\n".join(o)

    def test_nothing_before_the_choice(self):
        (self.base / "lora_weak" / "choice.json").unlink()
        self.assertEqual(self.text(), "")

    def test_reading_rule_and_effect_sizes(self):
        t = self.text()
        self.assertIn("**Chosen level: p = 70%.**", t)
        self.assertIn("4 single block(s) fail (1, 2, 3, 4) -> **strength makes location matter**", t)
        self.assertIn("| 4 | 0.833 / 0.833 / 0.833 | P/P/F | accuracy drop |", t)
        self.assertIn("| 1 | 0.167 / 0.167 / 0.167 | F/F/F | gap |", t)
        self.assertIn("Same blocks as r4's failures (1, 2, 3, 4, 5, 11): no", t)
        self.assertIn("3-seed mean gap 0.300 vs r4's 0.281", t)
        self.assertIn("removes 0.900 / 0.900 / 0.900 of the gap; c-ii (another seed's shortcut blocks) keeps "
                      "0.900 / 0.900 / 0.900", t)
        self.assertIn("Collapse rule (6 Nov amendment): no run collapsed.", t)

    def test_gate_failure_is_not_informative(self):
        json.dump({"seed": 0, "pass": False, "gap_after_all": 0.2, "limit": 0.15},
                  open(self.base / "lora_r64_weakp70" / "retrain" / "gate.json", "w"))
        t = self.text()
        self.assertIn("**Phase 3 gate failed**", t)
        self.assertNotIn("Reading rule:", t)

    def test_no_level_qualifies(self):
        ch = json.load(open(self.base / "lora_weak" / "choice.json"))
        ch["chosen"] = None
        json.dump(ch, open(self.base / "lora_weak" / "choice.json", "w"))
        self.assertIn("**No level qualifies: STOP, a valid result**", self.text())


if __name__ == "__main__":
    unittest.main()
