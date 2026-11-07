"""Phase L5: offline checks (random-init GPT-2 on CPU, synthetic results). No downloads, no GPU.
Run: python -m unittest tests/test_seedswap.py"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from peft import get_peft_model
from safetensors.torch import save_file
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import report_seedswap as RS
import run_lora as RL
import seedswap as S
from adapter_overlap import MODULES, lora_factors
from common import loss_collapsed, run_kind
from swap import key
from train_lora import add_lora, load_lora_cfg, lora_a_fingerprint, lora_config

ROOT = Path(__file__).resolve().parents[1]
RANK = 4


def tiny():
    return GPT2LMHeadModel(GPT2Config(n_layer=12, n_head=2, n_embd=32, n_positions=64, vocab_size=64))


def fake_result(gap):
    return {"accuracy": {"none": 0.9, "match": 1.0, "opposite": 0.9 - gap, "neutral": 0.9}, "mean_d": {},
            "gap_none_minus_opposite": gap}


class Seeding(unittest.TestCase):
    def setUp(self):
        self.lcfg = load_lora_cfg()

    def hooked(self, init_seed, run_seed=0):
        torch.manual_seed(run_seed)
        rec = {}
        m = add_lora(self.lcfg, RANK, init_seed, rec)(tiny())
        return m, rec, torch.rand(4)  # what dropout would draw next

    def test_same_init_same_a_other_init_other_a(self):
        a = self.hooked(3, run_seed=0)[1]["lora_A_init_sha256"]
        self.assertEqual(a, self.hooked(3, run_seed=1)[1]["lora_A_init_sha256"])
        self.assertNotEqual(a, self.hooked(4, run_seed=0)[1]["lora_A_init_sha256"])

    def test_rng_afterwards_depends_only_on_the_run_seed(self):
        self.assertTrue(torch.equal(self.hooked(3, run_seed=7)[2], self.hooked(4, run_seed=7)[2]))
        self.assertFalse(torch.equal(self.hooked(3, run_seed=7)[2], self.hooked(3, run_seed=8)[2]))

    def test_default_is_plain_get_peft_model(self):
        torch.manual_seed(5)
        rec = {}
        a = add_lora(self.lcfg, RANK, None, rec)(tiny())
        after_a = torch.rand(4)
        torch.manual_seed(5)
        b = get_peft_model(tiny(), lora_config(self.lcfg, RANK))
        self.assertEqual(rec, {})
        self.assertEqual(lora_a_fingerprint(a), lora_a_fingerprint(b))
        self.assertTrue(torch.equal(after_a, torch.rand(4)))


class Pairs(unittest.TestCase):
    def test_counts_from_the_spec(self):
        c = S.cell_pairs()
        self.assertEqual({k: len(v) for k, v in c.items()}, {"A": 14, "B": 14, "C": 28})
        self.assertEqual(sum(map(len, c.values())) * len(S.BLOCK_SETS), 280)
        self.assertEqual(len({p for v in c.values() for p in v}), 56)
        self.assertFalse({m for v in c.values() for p in v for m in p} & {S.LEFT_OUT})

    def test_replicates_are_the_spec_cells_and_part_of_the_pairs(self):
        r, c = S.replicate_pairs(), S.cell_pairs()
        self.assertEqual(r["A"], [((0, 0), (0, 1)), ((1, 1), (1, 2)), ((2, 2), (2, 1))])
        self.assertEqual(r["B"], [((0, 0), (1, 0)), ((1, 1), (2, 1)), ((2, 2), (0, 2))])
        self.assertEqual(r["C"], [((0, 0), (2, 1)), ((1, 1), (0, 2)), ((2, 2), (1, 0))])
        for cell in r:
            self.assertTrue(set(r[cell]) <= set(c[cell]))
        self.assertTrue(all(d[0] != d[1] for _, d in r["C"]))  # every C donor is a new model, not a main one

    def test_original_pairs_are_kept_as_cell_c_pairs(self):
        self.assertEqual(S.original_pairs(), [((0, 0), (1, 1)), ((1, 1), (2, 2)), ((2, 2), (0, 0))])
        self.assertTrue(set(S.original_pairs()) <= set(S.cell_pairs()["C"]))

    def test_model_name_is_a_run_name(self):
        self.assertEqual(S.model_name(1, 2), "lora_r64_init1_shortcut_seed2")
        self.assertEqual(run_kind(S.model_name(1, 2)), "shortcut")


class Swaps(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        torch.manual_seed(0)
        self.sds = {(i, j): {k: v.clone() for k, v in tiny().state_dict().items()} for i, j in S.GRID}
        self.model = tiny()
        self.seen = []

    def evaluate(self, model, tok, tests, cfg):
        self.seen.append({k: v.clone() for k, v in model.state_dict().items()})
        return fake_result(0.5)

    def test_all_evaluations_then_resume(self):
        p = self.dir / "swaps.json"
        S.run_swaps(self.sds, self.model, None, None, None, p, self.evaluate)
        self.assertEqual(len(self.seen), 8 + 280)
        n = len(self.seen)
        S.run_swaps(self.sds, self.model, None, None, None, p, self.evaluate)
        self.assertEqual(len(self.seen), n)  # everything was saved: nothing evaluated again

    def test_only_the_chosen_blocks_come_from_the_donor(self):
        S.run_swaps(self.sds, self.model, None, None, None, self.dir / "swaps.json", self.evaluate)
        i = 8  # first swap after the 8 baselines: cell A, (0,0) <- (0,1), block 9
        edit, donor, got = self.sds[(0, 0)], self.sds[(0, 1)], self.seen[i]
        for k, v in got.items():
            want = donor[k] if k.startswith("transformer.h.9.") else edit[k]
            self.assertTrue(torch.equal(v, want), k)


class Rule(unittest.TestCase):
    def test_all_outcomes(self):
        self.assertEqual(RS.verdict(False, False, True), "the original failure was a single-pair event")
        self.assertEqual(RS.verdict(True, False, False), "LoRA's random init causes the c-ii failure")
        self.assertEqual(RS.verdict(False, True, False), "the training path (data order + dropout) causes it")
        self.assertIn("either one is enough", RS.verdict(False, False, False))
        self.assertEqual(RS.verdict(True, True, False), "only both together break it")

    def test_a_cell_keeps_only_at_80_percent_in_all_3_replicates(self):
        res = {"baseline": {S.model_name(k, k): fake_result(1.0) for k in S.IDS}, "cells": {"A": {}}}
        for e, d in S.replicate_pairs()["A"]:
            res["cells"]["A"][S.pair_key(e, d)] = {S.PRIMARY: fake_result(0.8)}
        self.assertTrue(RS.cell_keeps(res, "A", S.PRIMARY))
        e, d = S.replicate_pairs()["A"][2]
        res["cells"]["A"][S.pair_key(e, d)][S.PRIMARY] = fake_result(0.79)
        self.assertFalse(RS.cell_keeps(res, "A", S.PRIMARY))


class Overlap(unittest.TestCase):
    def test_same_adapter_is_1_other_is_less(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        rank = S.RANK
        paths = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            sd = {}
            for b in range(12):
                for m in MODULES:
                    pre = f"base_model.model.transformer.h.{b}.{m}"
                    sd[f"{pre}.lora_A.weight"] = torch.randn(rank, 200)
                    sd[f"{pre}.lora_B.weight"] = torch.randn(300, rank)
            paths.append(d / f"a{seed}.safetensors")
            save_file(sd, paths[-1])
        f = {(0, 0): lora_factors(paths[0], rank, S.ALPHA), (0, 1): lora_factors(paths[0], rank, S.ALPHA),
             (1, 0): lora_factors(paths[1], rank, S.ALPHA)}
        same = S.pair_overlaps(f, (0, 0), (0, 1))
        other = S.pair_overlaps(f, (0, 0), (1, 0))
        self.assertAlmostEqual(same["9"]["input"], 1.0, places=6)
        self.assertAlmostEqual(same["9"]["output"], 1.0, places=6)
        self.assertLess(other["9"]["input"], 0.6)
        self.assertLess(other["9"]["output"], 0.6)


class Report(unittest.TestCase):
    def test_end_to_end_on_synthetic_results(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        (d / "checks").mkdir()
        grid = S.GRID
        keep = {"A": 0.95, "B": 0.50, "C": 0.50}  # same init keeps: "LoRA's random init causes the c-ii failure"
        res = {"baseline": {S.model_name(*m): fake_result(0.8) for m in grid},
               "cells": {c: {S.pair_key(e, f): {key(b): {**fake_result(0.8 * keep[c]), "wall_time_s": 1.0}
                                                for b in S.BLOCK_SETS} for e, f in pairs}
                         for c, pairs in S.cell_pairs().items()},
               "provenance": {"git_commits": ["abcdef0123"], "gpus": ["L4"], "total_wall_time_s": 3600.0}}
        json.dump(res, open(d / "swaps.json", "w"))
        by_model = {S.model_name(i, j): f"{i}" * 64 for i in S.IDS for j in S.IDS}
        json.dump({"by_model": by_model, "pass": True}, open(d / "init_fingerprints.json", "w"))
        late = {str(b): {"input": 0.5, "output": 0.1} for b in range(12)}
        json.dump({"cells": {c: {S.pair_key(e, f): late for e, f in p} for c, p in S.cell_pairs().items()}},
                  open(d / "overlap.json", "w"))
        for m in grid:
            json.dump({"accuracy": {"none": 0.9, "opposite": 0.1}, "gap_none_minus_opposite": 0.8, "all_pass": True},
                      open(d / "checks" / f"{S.model_name(*m)}.json", "w"))
        json.dump({"accuracy": {"none": 0.771, "opposite": 0.368}, "gap_none_minus_opposite": 0.403, "all_pass": False,
                   "checks": {"task_learned": {"pass": False}},
                   "train": {"loss_log": [{"step": 100, "loss": 0.43}, {"step": 300, "loss": 0.69}]}},
                  open(d / "checks" / f"{S.model_name(*S.LEFT_OUT)}.json", "w"))
        r = subprocess.run([sys.executable, str(ROOT / "src" / "report_seedswap.py"), "--results", str(d)],
                           capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(r.returncode, 0, r.stderr)
        t = (d / "RESULTS_SEEDSWAP.md").read_text(encoding="utf-8")
        self.assertIn("**Reading: LoRA's random init causes the c-ii failure.**", t)
        self.assertIn("| 9-10-11 | A | 14 | 0.950 | 0.950 |", t)
        self.assertIn("| 11 | C | 28 | 0.500 | 0.500 |", t)
        self.assertIn("8 of 9 models", t)
        self.assertIn("the original pairs", t)
        self.assertIn("Collapse rule met: True", t)
        self.assertIn("left out", t)
        self.assertFalse((ROOT / "RESULTS_SEEDSWAP.md").exists())


class Collapse(unittest.TestCase):
    def pilot(self, name):
        return json.load(open(ROOT / "results" / "lora_pilot" / f"{name}.json"))

    def test_the_rule_on_the_pilot_runs(self):
        for n in ("lora_r64_lr1e-3_shortcut_seed0", "lora_r64_lr1e-3_clean_seed0"):
            r = self.pilot(n)
            self.assertTrue(loss_collapsed(r["train"]["loss_log"]), n)
            self.assertTrue(RL.left_out_collapsed(r), n)
        self.assertFalse(loss_collapsed(self.pilot("lora_r64_lr5e-4_shortcut_seed0")["train"]["loss_log"]))

    def test_boundaries(self):
        self.assertFalse(loss_collapsed([{"step": 200, "loss": 0.9}]))  # "after 200"
        self.assertTrue(loss_collapsed([{"step": 250, "loss": 0.65}]))
        self.assertFalse(loss_collapsed([{"step": 250, "loss": 0.649}]))

    def test_a_run_that_passes_is_not_left_out(self):
        r = self.pilot("lora_r64_lr1e-3_shortcut_seed0")
        r["checks"]["task_learned"]["pass"] = True
        self.assertFalse(RL.left_out_collapsed(r))


class SmokeHelpers(unittest.TestCase):
    def test_same_losses_tolerates_rounding_only(self):
        self.assertTrue(RL.same_losses([0.7001, 0.5], [0.7002, 0.5]))
        self.assertFalse(RL.same_losses([0.7, 0.5], [0.72, 0.5]))
        self.assertFalse(RL.same_losses([0.7], [0.7, 0.5]))
        self.assertFalse(RL.same_losses([], []))

    def test_base_commit_exists_and_has_no_init_seed(self):
        t = subprocess.run(["git", "-C", str(ROOT), "show", f"{RL.L5_BASE}:src/train_lora.py"], capture_output=True,
                           text=True, encoding="utf-8", check=True).stdout
        self.assertNotIn("--init-seed", t)


class Driver(unittest.TestCase):
    """src/run_lora.py seedswap with training, swaps and the report stubbed: order, stops, provenance."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        self.runs, self.out = self.dir / "runs", self.dir / "seedswap"
        (self.out / "checks").mkdir(parents=True)
        self.names = [S.model_name(i, j) for i, j in S.GRID]
        self.left = S.model_name(*S.LEFT_OUT)
        json.dump({"pass": True}, open(self.out / "smoke.json", "w"))
        json.dump({"pass": True}, open(self.out / "smoke_gnorm.json", "w"))
        self.steps, self.extra = [], {}
        for n in self.names + [self.left]:
            (self.runs / n).mkdir(parents=True)
            fp = n.split("_init")[1][0] * 64
            json.dump({"lora_A_init_sha256": fp}, open(self.runs / n / "lora_check.json", "w"))
            json.dump({"accuracy": {"none": 0.9, "opposite": 0.1}}, open(self.out / "checks" / f"{n}.json", "w"))
        self.collapse_record()
        self.patches = [patch.object(RL, "RUNS", self.runs), patch.object(RL, "L5_DIR", self.out),
                        patch.object(RL, "log", lambda m: None), patch.object(RL, "step", self.fake_step),
                        patch.object(RL, "train_and_check", self.fake_train),
                        patch.object(RL, "add_env_session", self.fake_env)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def collapse_record(self, task_pass=False):
        json.dump({"accuracy": {"none": 0.771, "opposite": 0.368}, "checks": {"task_learned": {"pass": task_pass}},
                   "train": {"loss_log": [{"step": 300, "loss": 0.69}]}},
                  open(self.out / "checks" / f"{self.left}.json", "w"))

    def fake_train(self, runs, label, out_dir_of, stop_on_fail, extra):
        self.assertTrue(stop_on_fail)
        self.assertEqual([r[0] for r in runs], self.names)
        self.assertEqual({(r[1], r[2], r[3]) for r in runs}, {(5e-4, 64, "shortcut")})
        self.assertEqual(out_dir_of(64), self.out / "checks")
        self.extra = {r[0]: extra(r[0]) for r in runs}

    def fake_env(self, out, name, command):
        json.dump({"sessions": [{"git_commit": "abc", "gpu": "L4"}]}, open(out / "env.json", "w"))

    def fake_step(self, label, args, capture=False):
        self.steps.append(args)
        if args == ["seedswap.py"]:
            r = {**fake_result(0.5), "wall_time_s": 2.0}
            json.dump({"baseline": {n: {"accuracy": {"none": 0.9, "opposite": 0.1}} for n in self.names},
                       "cells": {"A": {"00<-01": {"9": r, "10": r}}}}, open(self.out / "swaps.json", "w"))

    def test_order_seeds_and_provenance(self):
        self.assertTrue(RL.seedswap())
        self.assertEqual(self.extra[S.model_name(2, 1)], ["--init-seed", "2", "--log-grad-norm"])
        self.assertNotIn(self.left, self.extra)
        self.assertEqual(self.steps, [["seedswap.py"], ["seedswap.py", "--overlap"], ["report_seedswap.py"]])
        prov = json.load(open(self.out / "swaps.json"))["provenance"]
        self.assertEqual(prov, {"git_commits": ["abc"], "gpus": ["L4"], "total_wall_time_s": 4.0})

    def test_stops_if_the_left_out_model_did_not_collapse(self):
        self.collapse_record(task_pass=True)
        with self.assertRaises(SystemExit) as c:
            RL.seedswap()
        self.assertEqual(c.exception.code, 2)
        self.assertEqual(self.steps, [])

    def test_stops_without_a_passing_grad_norm_smoke(self):
        json.dump({"pass": False}, open(self.out / "smoke_gnorm.json", "w"))
        with self.assertRaises(SystemExit) as c:
            RL.seedswap()
        self.assertEqual(c.exception.code, 1)

    def test_stops_without_a_passing_smoke_result(self):
        json.dump({"pass": False}, open(self.out / "smoke.json", "w"))
        with self.assertRaises(SystemExit) as c:
            RL.seedswap()
        self.assertEqual(c.exception.code, 1)
        self.assertEqual(self.steps, [])

    def test_stops_before_swaps_on_a_wrong_fingerprint(self):
        p = self.runs / S.model_name(0, 1) / "lora_check.json"
        json.dump({"lora_A_init_sha256": "f" * 64}, open(p, "w"))
        with self.assertRaises(SystemExit) as c:
            RL.seedswap()
        self.assertEqual(c.exception.code, 2)
        self.assertEqual(self.steps, [])

    def test_two_init_seeds_with_one_fingerprint_fail(self):
        for n in self.names:
            json.dump({"lora_A_init_sha256": "a" * 64}, open(self.runs / n / "lora_check.json", "w"))
        self.assertFalse(RL.init_fingerprints(self.names))

    def test_stops_if_a_baseline_differs_from_its_checks_file(self):
        p = self.out / "checks" / f"{self.names[4]}.json"
        json.dump({"accuracy": {"none": 0.8, "opposite": 0.1}}, open(p, "w"))
        with self.assertRaises(SystemExit) as c:
            RL.seedswap()
        self.assertEqual(c.exception.code, 2)
        self.assertNotIn(["report_seedswap.py"], self.steps)


if __name__ == "__main__":
    unittest.main()
