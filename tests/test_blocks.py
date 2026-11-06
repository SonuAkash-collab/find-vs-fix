"""Phase L4: offline checks (random-init GPT-2 on CPU, synthetic results). No downloads, no GPU.
Run: python -m unittest tests/test_blocks.py"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logit_check as LC
import report_lora as RP
import run_lora as RL
from train_lora import add_lora, frozen_outside, load_lora_cfg, lora_config

RANK = 4
EARLY = (0, 1, 2, 3, 4, 5)
SEEDS = (0, 1, 2)


def tiny():
    return GPT2LMHeadModel(GPT2Config(n_layer=12, n_head=2, n_embd=32, n_positions=64, vocab_size=64))


class Blocks(unittest.TestCase):
    def setUp(self):
        self.lcfg = load_lora_cfg()

    def names(self, blocks):
        m = add_lora(self.lcfg, RANK, blocks=blocks)(tiny())
        return [n for n, p in m.named_parameters() if p.requires_grad], m

    def test_adapters_only_on_the_set(self):
        names, _ = self.names(EARLY)
        self.assertEqual(sum(n.endswith("lora_A.default.weight") for n in names), 24)
        self.assertEqual({int(n.split(".h.")[1].split(".")[0]) for n in names}, set(EARLY))

    def test_default_is_every_block_and_the_old_config(self):
        names, _ = self.names(None)
        self.assertEqual(sum(n.endswith("lora_A.default.weight") for n in names), 48)
        self.assertIsNone(lora_config(self.lcfg, RANK).layers_to_transform)

    def test_six_r64_blocks_have_the_spec_parameter_count(self):
        m = GPT2LMHeadModel(GPT2Config(n_layer=12, n_head=2, n_embd=768, n_positions=8, vocab_size=8))
        m = add_lora(self.lcfg, 64, blocks=EARLY)(m)
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad), RL.L4_PARAMS)

    def test_frozen_outside_after_a_merge(self):
        torch.manual_seed(0)
        base = tiny()
        plain = {k: v.clone() for k, v in base.state_dict().items()}
        m = add_lora(self.lcfg, RANK, blocks=EARLY)(base)
        for n, p in m.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p)
        merged = m.merge_and_unload()
        self.assertEqual(frozen_outside(merged, plain, EARLY)["blocks"], list(EARLY))

    def test_frozen_outside_catches_a_change_and_an_untouched_block(self):
        base = tiny()
        plain = {k: v.clone() for k, v in base.state_dict().items()}
        sd = base.state_dict()
        with torch.no_grad():
            sd["transformer.h.0.mlp.c_fc.weight"] += 1
            sd["transformer.h.9.mlp.c_fc.weight"] += 1
        with self.assertRaises(AssertionError):
            frozen_outside(base, plain, (0,))
        with torch.no_grad():
            sd["transformer.h.9.mlp.c_fc.weight"].copy_(plain["transformer.h.9.mlp.c_fc.weight"])
        self.assertTrue(frozen_outside(base, plain, (0,))["frozen_outside_blocks_identical"])
        with self.assertRaises(AssertionError):
            frozen_outside(base, plain, (0, 1))


class Rules(unittest.TestCase):
    def test_rule1(self):
        self.assertIn("GPT-2's structure", RP.l4_rule1([7, 12, 9], [True] * 3))
        self.assertIn("where it is learned", RP.l4_rule1([6, 3, 1], [True] * 3))
        self.assertEqual(RP.l4_rule1([6, 7, 7], [True] * 3), "mixed")
        self.assertEqual(RP.l4_rule1([None, 7, 7], [True] * 3), "mixed")
        self.assertIn("not informative", RP.l4_rule1([7, 7, 7], [True, False, True]))

    def test_rule3(self):
        g = [0.5, 0.5, 0.5]
        self.assertEqual(RP.l4_rule3(g, [0.3] * 3, [0.3] * 3), "location matters")
        self.assertEqual(RP.l4_rule3(g, [0.7] * 3, [0.7] * 3), "location matters")
        self.assertEqual(RP.l4_rule3(g, [0.45] * 3, [0.55] * 3), "number of blocks, not location")
        self.assertEqual(RP.l4_rule3(g, [0.3] * 3, [0.7] * 3), "mixed")  # opposite directions
        self.assertEqual(RP.l4_rule3(g, [0.3, 0.3, 0.45], [0.3] * 3), "mixed")

    def test_rule3_margin_exactly_0_10_counts(self):
        self.assertGreater(abs(0.8 - 0.7 - 0.1), 0)  # raw floats are not exact here; the rounding has to handle it
        self.assertEqual(RP.l4_rule3([0.8] * 3, [0.7] * 3, [0.7] * 3), "location matters")
        self.assertEqual(RP.l4_rule3([0.7] * 3, [0.6] * 3, [0.6] * 3), "location matters")

    def test_frozen_only(self):
        self.assertTrue(RL.frozen_only("7", EARLY))
        self.assertTrue(RL.frozen_only("10-11", EARLY))
        self.assertTrue(RL.frozen_only("attn:9", EARLY))
        self.assertFalse(RL.frozen_only("3", EARLY))
        self.assertFalse(RL.frozen_only("5-6", EARLY))
        self.assertFalse(RL.frozen_only("all12", EARLY))
        self.assertFalse(RL.frozen_only("emb_names+9-10-11", EARLY))


def result(gap=0.5, none=0.9, d=1.0):
    return {"accuracy": {"none": none, "match": 1.0, "opposite": none - gap, "neutral": none},
            "mean_d": {v: d for v in ("none", "match", "opposite", "neutral")}, "gap_none_minus_opposite": gap}


def write_set(base, name, gap, arrived, seeds=SEEDS, learned=True):
    p = f"lora_r64_{name}"
    (base / p / "checks").mkdir(parents=True, exist_ok=True)
    for sub in ("patching", "logit_lens", "swap", "logit_check"):
        (base / p / sub).mkdir(exist_ok=True)
    layers = 13
    for s in seeds:
        for kind, g in (("shortcut", gap), ("clean", 0.01)):
            c = {"task_learned": {"pass": True}}
            c["shortcut_learned" if kind == "shortcut" else "clean_no_shortcut"] = {"pass": learned or kind == "clean"}
            json.dump({"accuracy": {"none": 0.9}, "gap_none_minus_opposite": g, "checks": c,
                       "train": {"wall_time_s": 300.0, "peak_vram_allocated_gib": 2.0, "n_trainable_params": RL.L4_PARAMS}},
                      open(base / p / "checks" / f"{p}_{kind}_seed{s}.json", "w"))
        F = [0.0 if i < arrived else 1.0 for i in range(layers)]
        rd = {"arrived_at": arrived, "F_notlast_L0_to_4": [1.0] * 5}
        json.dump({"models": {"shortcut": {"readout": rd, "curves": {"all": {"last": {"F": F}}}},
                              "clean": {"mean_dB_minus_dA": 0.02}}}, open(base / p / "patching" / f"seed{s}.json", "w"))
        acc = [0.9] * layers
        json.dump({"cal_acc": {"none": acc, "opposite": [0.9] * 11 + [0.4, 0.4]}},
                  open(base / p / "logit_lens" / f"{p}_shortcut_seed{s}.json", "w"))
        by = {str(b): {"gap_none_minus_opposite": gap / 2} for b in range(12)}
        json.dump({"baseline": {f"{p}_shortcut_seed{s}": {"gap_none_minus_opposite": gap}},
                   "cases": {"a_shortcut_from_clean_same": {"by_blocks": by}}}, open(base / p / "swap" / f"seed{s}.json", "w"))
        json.dump({"summary": {"none": {"d_mean": 0.5}}, "name_effect": {"delta_AB": {"mean": 3.0}}},
                  open(base / p / "logit_check" / f"{p}_shortcut_seed{s}.json", "w"))


class Section(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base)
        for n, g, a in (("early", 0.3, 9), ("late", 0.8, 12), ("even", 0.6, 12), ("odd", 0.6, 12)):
            write_set(self.base, n, g, a)
        (self.base / "lora_r64" / "patching").mkdir(parents=True)
        for s in SEEDS:
            json.dump({"models": {"shortcut": {"readout": {"arrived_at": 12},
                                               "curves": {"all": {"last": {"F": [0.0] * 12 + [1.0]}}}}}},
                      open(self.base / "lora_r64" / "patching" / f"seed{s}.json", "w"))

    def test_nothing_until_every_file_exists(self):
        (self.base / "lora_r64" / "patching" / "seed2.json").unlink()
        o = []
        RP.blocks_section(o, self.base)
        self.assertEqual(o, [])

    def test_whole_section(self):
        o = []
        RP.blocks_section(o, self.base)
        t = "\n".join(o)
        self.assertIn("**Rule 1, early set, patching (main):** arrived at 9 / 9 / 9 -> **\"late\" comes from GPT-2's "
                      "structure (explanation 1).**", t)
        self.assertIn("agrees with patching", t)
        self.assertIn("**Rule 2, late set:** passes", t)
        self.assertIn("| early | -0.300 / -0.300 / -0.300 | -0.300 / -0.300 / -0.300 | location matters |", t)
        self.assertIn("| late | 0.200 / 0.200 / 0.200 | 0.200 / 0.200 / 0.200 | location matters |", t)
        self.assertIn("| all 12 blocks, r64 (reference) | 12 / 12 / 12 |", t)
        self.assertIn("early 0.020 / 0.020 / 0.020", t)
        row = next(x for x in o if x.startswith("| early | 0.50"))
        self.assertEqual(row.count("0.50"), 6)  # fraction removed: LoRA blocks only

    def test_early_set_without_a_shortcut_is_not_informative(self):
        write_set(self.base, "early", 0.1, 9, learned=False)
        o = []
        RP.blocks_section(o, self.base)
        self.assertIn("not informative", "\n".join(o))


class Driver(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base)
        self.steps, self.trained = [], []
        for name, p in (("RESULTS", self.base), ("L4_DIR", self.base / "lora_blocks"), ("log", lambda m: None),
                        ("step", self.fake_step), ("train_and_check", self.fake_train),
                        ("add_env_session", self.fake_env)):
            x = patch.object(RL, name, p)
            x.start()
            self.addCleanup(x.stop)
        (self.base / "lora_blocks").mkdir()
        json.dump({"pass": True}, open(self.base / "lora_blocks" / "smoke.json", "w"))
        for n, b in RL.L4_SETS.items():
            write_set(self.base, n, 0.6, 12)

    def fake_train(self, runs, label, out_dir_of, extra):
        self.trained.append((label, [r[0] for r in runs], extra(runs[0][0]), out_dir_of(64)))

    def fake_env(self, out, name, command):
        json.dump({"sessions": [{"git_commit": "abc", "gpu": "L4"}]}, open(out / "env.json", "w"))

    def fake_step(self, label, args, capture=False):
        self.steps.append(args)
        if args[0] == "swap.py":
            p = args[2]
            sub = "swap2" if "--phase2b" in args else "swap"
            field = "by_set" if sub == "swap2" else "by_blocks"
            name = "7" if p.endswith("early") else "attn:0" if p.endswith("late") else "10-11-99"
            for s in SEEDS:
                r = dict(result(), kind="blocks")
                json.dump({"baseline": {"e": result()}, "cases": {"a": {"editor": "e", field: {name: r}}}},
                          open(self.base / p / sub / f"seed{s}.json", "w"))

    def test_order_gate_then_analyses_then_reference(self):
        self.assertTrue(RL.blocks())
        self.assertEqual(len(self.trained), 8)
        self.assertEqual([x[1][0] for x in self.trained[:4]],
                         [f"lora_r64_{n}_shortcut_seed0" for n in ("early", "late", "even", "odd")])
        self.assertEqual(len(self.trained[0][1]), 2)
        self.assertEqual(len(self.trained[4][1]), 4)
        self.assertEqual(self.trained[1][2], ["--blocks", "6,7,8,9,10,11"])
        self.assertEqual(self.trained[3][2], ["--blocks", "1,3,5,7,9,11"])
        self.assertEqual(self.trained[1][3], self.base / "lora_r64_late" / "checks")
        self.assertEqual(self.steps[0], ["logit_lens.py", "--prefix", "lora_r64_early"])
        self.assertIn(["patching.py", "--prefix", "lora_r64_odd", "--adapters", "runs", "--batch", "64"], self.steps)
        self.assertIn(["logit_check.py", "--prefix", "lora_r64_even", "--adapters", "runs"], self.steps)
        self.assertEqual(self.steps[-2], ["patching.py", "--prefix", "lora_r64", "--adapters", "runs", "--batch", "64"])
        self.assertEqual(self.steps[-1], ["report_lora.py"])
        prov = json.load(open(self.base / "lora_r64_early" / "swap" / "seed1.json"))["provenance"]
        self.assertEqual(prov["git_commits"], ["abc"])

    def test_resume_skips_finished_steps(self):
        RL.blocks()
        n = len(self.steps)
        RL.blocks()
        self.assertEqual(self.steps[n:], [["report_lora.py"]])

    def test_shortcut_under_0_20_does_not_stop(self):
        write_set(self.base, "even", 0.1, 12, learned=False)
        self.assertTrue(RL.blocks())

    def test_task_failure_or_wrong_parameter_count_stops_before_seed_1(self):
        p = self.base / "lora_r64_late" / "checks" / "lora_r64_late_clean_seed0.json"
        r = json.load(open(p))
        r["checks"]["task_learned"]["pass"] = False
        json.dump(r, open(p, "w"))
        with self.assertRaises(SystemExit) as c:
            RL.blocks()
        self.assertEqual(c.exception.code, 2)
        self.assertEqual(len(self.trained), 4)
        self.assertEqual(self.steps, [])

    def test_a_changed_frozen_swap_stops(self):
        orig = self.fake_step

        def bad(label, args, capture=False):
            orig(label, args, capture)
            if args[0] == "swap.py" and args[2].endswith("early") and "--phase2b" not in args:
                p = self.base / args[2] / "swap" / "seed0.json"
                r = json.load(open(p))
                r["cases"]["a"]["by_blocks"]["7"]["accuracy"]["none"] = 0.5
                json.dump(r, open(p, "w"))
        with patch.object(RL, "step", bad):
            with self.assertRaises(SystemExit) as c:
                RL.blocks()
        self.assertEqual(c.exception.code, 2)
        self.assertFalse(any(a[0] == "patching.py" for a in self.steps))


class LogitCheck(unittest.TestCase):
    def test_checks_file_of_each_family(self):
        self.assertEqual(LC.checks_file("lora_r4_shortcut_seed0"),
                         LC.RESULTS / "lora_r4" / "checks" / "lora_r4_shortcut_seed0.json")
        self.assertEqual(LC.checks_file("lora_r64_early_clean_seed2"),
                         LC.RESULTS / "lora_r64_early" / "checks" / "lora_r64_early_clean_seed2.json")


if __name__ == "__main__":
    unittest.main()
