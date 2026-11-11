"""Phase 4b: offline checks (random-init GPT-2 on CPU, synthetic results, the driver with stubbed steps). No GPU.
Run: python -m unittest tests/test_restore.py"""
import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import patching as PT  # noqa: E402
import report_restore as RR  # noqa: E402
import run_lora as RL  # noqa: E402


def tiny(seed=0):
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(n_layer=12, n_head=2, n_embd=32, n_positions=64, vocab_size=64)).eval()


def fixed_copy(S, block, seed=1):
    """S with only block `block` changed (as Phase 3 retraining does)."""
    F = copy.deepcopy(S)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in F.named_parameters():
            if n.startswith(f"transformer.h.{block}."):
                p.add_(0.5 * torch.randn(p.shape, generator=g))
    return F


class Batch(unittest.TestCase):
    """restore_batch on real (tiny) models: the identities, the reverse, and the default path unchanged."""

    def setUp(self):
        self.S = tiny()
        torch.manual_seed(3)
        self.ids = torch.randint(5, 64, (6, 20))
        self.attn = torch.ones_like(self.ids)
        self.attn[0, 17:] = 0  # one padded row
        self.starts = torch.tensor([3, 4, 5, 6, 2, 8])

    def run_b(self, block, reverse):
        F = fixed_copy(self.S, block)
        pS, pF = PT.Patcher(self.S), PT.Patcher(F)
        try:
            return PT.restore_batch(self.S, pS, F, pF, block + 1, self.ids, self.attn, self.starts, 4, 7, 9, reverse)
        finally:
            pS.remove()
            pF.remove()

    def test_identities_with_and_without_reverse(self):
        for b in (0, 3, 10):
            _, _, dR, san = self.run_b(b, reverse=False)
            self.assertEqual(set(san), {"nothing_restored_gives_dF", "all_restored_gives_dS"})
            self.assertEqual(set(dR), set(PT.RESTORE_SETS))
            _, _, dR2, san2 = self.run_b(b, reverse=True)
            self.assertEqual(set(san2), set(san) | {"nothing_replaced_gives_dS", "all_replaced_gives_dF"})
            self.assertTrue(all(v < 1e-4 for v in san2.values()), san2)
            for P in PT.RESTORE_SETS:  # the restore part is the same with --reverse
                self.assertTrue(torch.allclose(dR[P], dR2[P], atol=1e-6))
                self.assertIn(f"rev:{P}", dR2)

    def test_the_fix_and_its_reverse_add_up_at_all(self):
        d_S, d_F, dR, _ = self.run_b(0, reverse=True)
        sign = np.ones(6)
        labels = np.zeros(6, dtype=int)
        cast = lambda t: t.numpy().astype(np.float64)  # noqa: E731
        rev = PT.summarize_reverse(sign, labels, cast(d_S), cast(d_F), {P: cast(dR[f"rev:{P}"]) for P in PT.RESTORE_SETS})
        res = PT.summarize_restore(sign, labels, cast(d_S), cast(d_F), {P: cast(dR[P]) for P in PT.RESTORE_SETS})
        self.assertAlmostEqual(res["sets"]["all"]["R"], 1.0, places=4)
        self.assertAlmostEqual(rev["sets"]["all"]["R_rev"], 1.0, places=4)
        self.assertAlmostEqual(rev["mean_dn_S_minus_dn_F"], res["mean_dn_S_minus_dn_F"], places=6)  # same denominator


class Summary(unittest.TestCase):
    def test_r_rev_values(self):
        sign = np.array([1.0, -1.0, 1.0, -1.0])
        labels = np.array([1, 0, 0, 1])
        d_S = np.array([3.0, -3.0, 3.0, -3.0])  # d_n S = 3
        d_F = np.array([1.0, -1.0, 1.0, -1.0])  # d_n F = 1: the fix moved d_n by 2
        rev = PT.summarize_reverse(sign, labels, d_S, d_F, {"name": np.array([2.0, -2.0, 2.0, -2.0]),
                                                             "all": d_F.copy()})
        self.assertAlmostEqual(rev["sets"]["name"]["R_rev"], 0.5)  # half of the fix
        self.assertAlmostEqual(rev["sets"]["all"]["R_rev"], 1.0)
        self.assertAlmostEqual(rev["sets"]["name"]["mean_dn_S_minus_dn_reversed"], 1.0)
        same = PT.summarize_reverse(sign, labels, d_S, d_S.copy(), {"name": d_S.copy()})
        self.assertFalse(same["R_rev_defined"])
        self.assertIsNone(same["sets"]["name"]["R_rev"])


def restore_file(R_name, Rrev_name, R_last=0.1, rev=True):
    sets = {P: {"R": {"name": R_name, "last": R_last}.get(P, 1.0), "mean_dn_restored": 2.0} for P in RR.SETS}
    f = {"restore": {"sets": sets, "mean_dn_S": 3.0, "mean_dn_F": 1.0},
         "ab_diff_norm": {m: {"name": [1.0] * 13, "last": [0.5] * 13} for m in ("S", "F_b")},
         "env": {"wall_time_s": 60.0, "git_commit": "abcdef0"}}
    if rev:
        f["reverse"] = {"sets": {P: {"R_rev": {"name": Rrev_name}.get(P, 1.0), "mean_dn_reversed": 2.5}
                                 for P in RR.SETS}}
    return f


class Report(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base)
        (self.base / "restore_4b").mkdir()
        models = {f"{c}_seed{s}_block{b}": {"G": 0.8, "gap_after": 0.1, "acc_none_drop": 0.0, "removes": True,
                                             "same_run_before": None}
                  for c in RR.CONDS for s in RR.SEEDS for b in RR.BLOCKS}
        json.dump({"models": models}, open(self.base / "restore_4b" / "fixed_check.json", "w"))
        json.dump({"all_pass": True}, open(self.base / "restore_4b" / "start_checks.json", "w"))
        json.dump({"pass": True}, open(self.base / "restore_4b" / "smoke.json", "w"))
        self.write("full_ft", 0, [0.9, 0.8, 0.7], [0.6, 0.7, 0.9])  # restores + reproduces
        self.write("full_ft", 3, [0.2, 0.1, 0.3], [0.1, 0.2, 0.1])  # does not
        self.write("full_ft", 10, [0.9, 0.2, 0.9], [0.9, 0.9, 0.1], R_last=0.8)  # mixed; last restores
        for b, (r, v) in {0: ([0.9] * 3, [0.1] * 3), 3: ([0.6] * 3, [0.6] * 3), 10: ([0.1] * 3, [0.1] * 3)}.items():
            self.write("lora_r64", b, r, v)

    def write(self, cond, b, Rs, Rrevs, R_last=0.1):
        d = (self.base / RR.CONDS[cond] if RR.CONDS[cond] else self.base) / "restore"
        d.mkdir(parents=True, exist_ok=True)
        for s, (r, v) in enumerate(zip(Rs, Rrevs)):
            json.dump(restore_file(r, v, R_last), open(d / f"seed{s}_block{b}.json", "w"))

    def text(self):
        return "\n".join(RR.build(self.base))

    def test_rules_as_in_the_spec(self):
        t = self.text()
        self.assertIn("**Block 0, P = name, restore:** restores -> **retraining block 0 cuts the trigger: the downstream "
                      "circuit is intact.**", t)
        self.assertIn("**Block 0, P = name, reverse:** reproduces -> **the change at the name tokens is enough for the "
                      "fix.** Together with the restore: **the fix lives at the name tokens (necessary and enough).**", t)
        self.assertIn("**Block 3, P = name, restore:** does not -> **retraining block 3 changes more than the "
                      "name-token signal.**", t)
        self.assertIn("**Block 10, P = name, restore:** mixed -> **mixed.**", t)
        self.assertIn("**Block 10, P = last, restore:** restores -> **the fix acts at the answer position (on the "
                      "circuit's readout).**", t)
        self.assertIn("Together with the restore: **the name tokens are needed but not enough.**", t)  # r64 block 0
        self.assertIn("no reading in the spec, reported as it is", t)  # r64 block 10 does not restore
        self.assertIn("| full fine-tuning | 0 | name | 0.900 / 0.800 / 0.700 | 0.600 / 0.700 / 0.900 |", t)

    def test_a_missing_seed_is_not_informative(self):
        (self.base / "restore" / "seed2_block0.json").unlink()
        self.assertIn("not informative (2 of 3 seeds)", self.text())

    def test_without_reverse_files(self):
        d = self.base / "restore"
        for s in RR.SEEDS:
            json.dump(restore_file(0.9, 0.0, rev=False), open(d / f"seed{s}_block0.json", "w"))
        self.assertIn("**Block 0, P = name, reverse:** not informative (0 of 3 seeds)", self.text())

    def test_nothing_before_the_fixed_models(self):
        (self.base / "restore_4b" / "fixed_check.json").unlink()
        self.assertIn("No fixed models yet.", self.text())


class Driver(unittest.TestCase):
    """restore() with every step stubbed: order, the removes filter, the arguments."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base)
        self.res, self.runs = self.base / "results", self.base / "runs"
        self.steps, self.fail = [], set()  # fail: (cond, seed, block) that does not remove
        for name, v in (("RESULTS", self.res), ("RUNS", self.runs), ("B4_DIR", self.res / "restore_4b"),
                        ("log", lambda m: None), ("step", self.fake_step), ("run", self.fake_run),
                        ("add_env_session", self.fake_env)):
            x = patch.object(RL, name, v)
            x.start()
            self.addCleanup(x.stop)
        x = patch("common.RUNS", self.runs)
        x.start()
        self.addCleanup(x.stop)
        (self.res / "restore_4b").mkdir(parents=True)
        json.dump({"pass": True}, open(self.res / "restore_4b" / "smoke.json", "w"))
        acc = {"none": 0.87, "match": 1.0, "opposite": 0.07, "neutral": 0.87}
        for pre in (None, "lora_r64"):
            d = (self.res / pre if pre else self.res) / "checks"
            d.mkdir(parents=True)
            for s in RL.L4_SEEDS:
                json.dump({"accuracy": acc}, open(d / f"{pre + '_' if pre else ''}shortcut_seed{s}.json", "w"))

    def fake_env(self, out, name, command):
        json.dump({"sessions": [{"git_commit": "abc", "gpu": "L4"}]}, open(out / "env.json", "w"))

    def fake_run(self, args, capture=False):
        self.steps.append(args)
        json.dump({"all_pass": True}, open(self.res / "restore_4b" / "start_checks.json", "w"))
        return 0, "", ""

    def fake_step(self, label, args, capture=False):
        self.steps.append(args)
        if args[0] == "retrain.py":
            pre = args[args.index("--prefix") + 1] if "--prefix" in args else None
            cond = "lora_r64" if pre else "full_ft"
            out = (self.res / pre if pre else self.res) / RL.B4_ARM
            acc = {"none": 0.87, "match": 1.0, "opposite": 0.07, "neutral": 0.87}
            for s in RL.L4_SEEDS:
                runs = {}
                for b in RL.B4_BLOCKS:
                    bad = (cond, s, b) in self.fail
                    runs[str(b)] = {"accuracy": {**acc, "opposite": 0.27 if bad else 0.82},  # bad: gap 0.60 > 50% of 0.8
                                    "gap_none_minus_opposite": 0.60 if bad else 0.05, "frozen_params_unchanged": True,
                                    "wall_time_s_incl_eval": 90.0, "saved_model": "x"}
                    saved = RL.b4_saved(pre, s, b)
                    saved.mkdir(parents=True, exist_ok=True)
                    (saved / "config.json").write_text("{}")
                json.dump({"baseline": {"accuracy": acc, "gap_none_minus_opposite": 0.8}, "runs": runs},
                          open(out / f"seed{s}.json", "w"))

    def test_order_and_arguments(self):
        self.assertTrue(RL.restore())
        self.assertEqual(self.steps[0][:3], ["nonames_check.py", "--conditions", "full_ft,lora_r64"])
        self.assertEqual(self.steps[1], ["retrain.py", "--arm", "retrain_4b", "--sets", "0,3,10", "--save-model"])
        self.assertEqual(self.steps[2], ["retrain.py", "--arm", "retrain_4b", "--sets", "0,3,10", "--save-model",
                                         "--prefix", "lora_r64"])
        rest = [a for a in self.steps if a[0] == "patching.py"]
        self.assertEqual(len(rest), 18)
        self.assertIn("--reverse", rest[0])
        self.assertEqual(rest[0][rest[0].index("--fixed-dir") + 1], str(self.runs / "retrain_saved" / "retrain_seed0_0"))
        self.assertIn(["patching.py", "--restore", "--reverse", "--seeds", "2", "--block", "10", "--fixed-dir",
                       str(self.runs / "lora_r64" / "retrain_saved" / "retrain_seed2_10"), "--prefix", "lora_r64"], rest)
        self.assertEqual(self.steps[-1], ["report_restore.py"])

    def test_a_fix_that_does_not_remove_gets_no_restore(self):
        self.fail = {("full_ft", 1, 3)}
        RL.restore()
        rest = [a for a in self.steps if a[0] == "patching.py"]
        self.assertEqual(len(rest), 17)
        self.assertFalse(any(a[a.index("--seeds") + 1] == "1" and a[a.index("--block") + 1] == "3"
                             and "--prefix" not in a for a in rest))
        fx = json.load(open(self.res / "restore_4b" / "fixed_check.json"))
        self.assertFalse(fx["models"]["full_ft_seed1_block3"]["removes"])

    def test_no_smoke_no_run(self):
        (self.res / "restore_4b" / "smoke.json").unlink()
        with self.assertRaises(SystemExit) as c:
            RL.restore()
        self.assertEqual(c.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
