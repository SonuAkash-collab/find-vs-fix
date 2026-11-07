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
import swap as SW
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


HEALTHY = [{"step": st, "loss": 0.69 if st == 50 else 0.3, "grad_norm_max": 2.5} for st in range(50, 1901, 50)]
COLLAPSE = [{"step": st, "loss": 0.42 if st <= 200 else 0.69, "grad_norm_max": 9.0} for st in range(50, 1901, 50)]


def write_check(base, p, kind, s, gap, learned=True, task=True, loss_log=HEALTHY):
    c = {"task_learned": {"pass": task}}
    c["shortcut_learned" if kind == "shortcut" else "clean_no_shortcut"] = {"pass": learned or kind == "clean"}
    (base / p / "checks").mkdir(parents=True, exist_ok=True)
    json.dump({"accuracy": {"none": 0.9 if task else 0.77, "opposite": 0.9 - gap}, "gap_none_minus_opposite": gap,
               "checks": c, "train": {"wall_time_s": 300.0, "peak_vram_allocated_gib": 2.0,
                                      "n_trainable_params": RL.L4_PARAMS, "loss_log": loss_log}},
              open(base / p / "checks" / f"{p}_{kind}_seed{s}.json", "w"))


def collapse(base, name, kind, s, task=False):
    """A collapsed run (task=False) or a high-loss run that still passes (task=True)."""
    write_check(base, f"lora_r64_{name}", kind, s, 0.4 if kind == "shortcut" else 0.01, task=task, loss_log=COLLAPSE)


def write_set(base, name, gap, arrived, seeds=SEEDS, learned=True):
    p = f"lora_r64_{name}"
    (base / p / "checks").mkdir(parents=True, exist_ok=True)
    for sub in ("patching", "logit_lens", "swap", "logit_check"):
        (base / p / sub).mkdir(exist_ok=True)
    layers = 13
    for s in seeds:
        for kind, g in (("shortcut", gap), ("clean", 0.01)):
            write_check(base, p, kind, s, g, learned)
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

    def test_replacement_and_flag_in_the_section(self):
        write_set(self.base, "early", 0.3, 9, seeds=(3,))
        collapse(self.base, "early", "shortcut", 1)
        (self.base / "lora_blocks").mkdir()
        slots = {n: [0, 1, 2] for n in RL.L4_SETS}
        slots["early"] = [0, 3, 2]
        json.dump({"slots": slots, "flagged": ["lora_r64_odd_clean_seed0"],
                   "replacement": {"set": "early", "collapsed_run": "lora_r64_early_shortcut_seed1", "collapsed_seed": 1,
                                   "seed": 3, "runs": ["lora_r64_early_shortcut_seed3", "lora_r64_early_clean_seed3"]}},
                  open(self.base / "lora_blocks" / "slots.json", "w"))
        (self.base / "lora_r64_early" / "patching" / "seed1.json").unlink()  # the collapsed seed is not read
        o = []
        RP.blocks_section(o, self.base)
        t = "\n".join(o)
        self.assertIn("**lora_r64_early_shortcut_seed1 collapsed** and is in no readout", t)
        self.assertIn("**Seed 3 takes seed 1's slot in set early**, so that set's per-seed values are seeds 0 / 3 / 2", t)
        self.assertIn("250: 0.690", t)
        self.assertIn("| early | 0.300 / 0.300 / 0.300 |", t)
        self.assertIn("Flagged, kept: lora_r64_odd_clean_seed0", t)
        self.assertIn("location matters Depends on flagged run(s) lora_r64_odd_clean_seed0", t)
        self.assertNotIn("Depends on flagged", next(x for x in o if x.startswith("**Rule 1")))

    def test_no_slots_file_means_seeds_0_1_2_and_no_collapse(self):
        o = []
        RP.blocks_section(o, self.base)
        t = "\n".join(o)
        self.assertIn("No run collapsed", t)
        self.assertTrue(next(x for x in o if x.startswith("| early | 0.300")).endswith("| 2.500 |"))

    def test_early_set_without_a_shortcut_is_not_informative(self):
        write_set(self.base, "early", 0.1, 9, learned=False)
        o = []
        RP.blocks_section(o, self.base)
        self.assertIn("not informative", "\n".join(o))


class Driver(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base)
        self.steps, self.trained, self.spare_collapses = [], [], False
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
        for name, _, _, kind in runs:  # a run without a checks file (the replacement pair) gets one
            p, s = name.rsplit(f"_{kind}_seed", 1)
            if not (self.base / p / "checks" / f"{name}.json").exists():
                bad = self.spare_collapses and kind == "shortcut"
                write_check(self.base, p, kind, int(s), 0.6 if kind == "shortcut" else 0.01, task=not bad,
                            loss_log=COLLAPSE if bad else HEALTHY)

    def fake_env(self, out, name, command):
        json.dump({"sessions": [{"git_commit": "abc", "gpu": "L4"}]}, open(out / "env.json", "w"))

    def fake_step(self, label, args, capture=False):
        self.steps.append(args)
        if args[0] == "swap.py":
            p = args[2]
            sub = "swap2" if "--phase2b" in args else "swap"
            field = "by_set" if sub == "swap2" else "by_blocks"
            name = "7" if p.endswith("early") else "attn:0" if p.endswith("late") else "10-11-99"
            seeds = [int(x) for x in args[args.index("--slot-seeds") + 1].split(",")] if "--slot-seeds" in args else SEEDS
            for s in seeds:
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
        self.assertEqual(self.trained[1][2], ["--blocks", "6,7,8,9,10,11", "--log-grad-norm"])
        self.assertEqual(self.trained[3][2], ["--blocks", "1,3,5,7,9,11", "--log-grad-norm"])
        self.assertEqual(self.trained[1][3], self.base / "lora_r64_late" / "checks")
        self.assertEqual(self.steps[0], ["logit_lens.py", "--prefix", "lora_r64_early"])
        self.assertIn(["patching.py", "--prefix", "lora_r64_odd", "--adapters", "runs", "--batch", "64"], self.steps)
        self.assertIn(["logit_check.py", "--prefix", "lora_r64_even", "--adapters", "runs"], self.steps)
        self.assertEqual(self.steps[-2], ["patching.py", "--prefix", "lora_r64", "--adapters", "runs", "--batch", "64"])
        self.assertEqual(self.steps[-1], ["report_lora.py"])
        prov = json.load(open(self.base / "lora_r64_early" / "swap" / "seed1.json"))["provenance"]
        self.assertEqual(prov["git_commits"], ["abc"])
        sl = json.load(open(self.base / "lora_blocks" / "slots.json"))
        self.assertEqual(sl["slots"], {n: [0, 1, 2] for n in RL.L4_SETS})
        self.assertIsNone(sl["replacement"])
        self.assertFalse(any("--slot-seeds" in a or "--seeds" in a for a in self.steps))

    # ---- collapse rule (6 Nov amendment)
    def test_one_collapse_trains_the_seed3_pair_and_it_takes_the_slot(self):
        collapse(self.base, "early", "shortcut", 1)
        self.assertTrue(RL.blocks())
        spare = [x for x in self.trained if x[1] == ["lora_r64_early_shortcut_seed3", "lora_r64_early_clean_seed3"]]
        self.assertTrue(spare)  # asked again at each later gate; train_and_check skips finished runs
        self.assertEqual({r for x in self.trained for r in x[1] if "seed3" in r},
                         {"lora_r64_early_shortcut_seed3", "lora_r64_early_clean_seed3"})
        self.assertEqual(spare[0][2], ["--blocks", "0,1,2,3,4,5", "--log-grad-norm"])
        sl = json.load(open(self.base / "lora_blocks" / "slots.json"))
        self.assertEqual(sl["slots"]["early"], [0, 3, 2])
        self.assertEqual(sl["slots"]["late"], [0, 1, 2])
        self.assertEqual(sl["replacement"]["collapsed_run"], "lora_r64_early_shortcut_seed1")
        self.assertIn(["swap.py", "--prefix", "lora_r64_early", "--slot-seeds", "0,3,2"], self.steps)
        self.assertIn(["swap.py", "--prefix", "lora_r64_early", "--phase2b", "--slot-seeds", "0,3,2"], self.steps)
        self.assertIn(["patching.py", "--prefix", "lora_r64_early", "--adapters", "runs", "--batch", "64",
                       "--seeds", "0", "3", "2"], self.steps)
        self.assertIn(["logit_check.py", "--prefix", "lora_r64_early", "--adapters", "runs", "--seeds", "0,3,2"],
                      self.steps)
        self.assertIn(["logit_lens.py", "--prefix", "lora_r64_early", "--run", "clean_seed3"], self.steps)
        self.assertNotIn(["logit_lens.py", "--prefix", "lora_r64_early", "--run", "shortcut_seed1"], self.steps)
        self.assertIn(["swap.py", "--prefix", "lora_r64_late"], self.steps)  # other sets unchanged
        n = len(self.steps)
        RL.blocks()  # resumable: nothing runs again
        self.assertEqual(self.steps[n:], [["report_lora.py"]])

    def test_a_collapse_at_seed0_is_replaced_before_seeds_1_and_2(self):
        collapse(self.base, "odd", "shortcut", 0)
        self.assertTrue(RL.blocks())
        labels = [x[0] for x in self.trained]
        self.assertLess(labels.index("Phase L4 odd seeds (3,)"), labels.index("Phase L4 early seeds (1, 2)"))

    def test_a_collapsed_clean_twin_replaces_the_pair(self):
        collapse(self.base, "late", "clean", 2)
        self.assertTrue(RL.blocks())
        sl = json.load(open(self.base / "lora_blocks" / "slots.json"))
        self.assertEqual(sl["slots"]["late"], [0, 1, 3])
        self.assertEqual(sl["replacement"]["runs"], ["lora_r64_late_shortcut_seed3", "lora_r64_late_clean_seed3"])

    def test_a_second_collapse_stops(self):
        collapse(self.base, "early", "shortcut", 1)
        collapse(self.base, "even", "clean", 2)
        with self.assertRaises(SystemExit) as c:
            RL.blocks()
        self.assertEqual(c.exception.code, 2)
        self.assertEqual(self.steps, [])

    def test_a_collapsed_replacement_stops(self):
        collapse(self.base, "early", "shortcut", 0)
        self.spare_collapses = True
        with self.assertRaises(SystemExit) as c:
            RL.blocks()
        self.assertEqual(c.exception.code, 2)
        self.assertEqual(self.steps, [])

    def test_high_loss_but_passing_is_kept_and_flagged(self):
        collapse(self.base, "even", "shortcut", 1, task=True)
        self.assertTrue(RL.blocks())
        sl = json.load(open(self.base / "lora_blocks" / "slots.json"))
        self.assertIsNone(sl["replacement"])
        self.assertEqual(sl["flagged"], ["lora_r64_even_shortcut_seed1"])
        self.assertFalse(any("seed3" in r for x in self.trained for r in x[1]))

    def test_a_task_failure_without_high_loss_still_stops(self):
        write_check(self.base, "lora_r64_early", "shortcut", 1, 0.6, task=False)
        with self.assertRaises(SystemExit) as c:
            RL.blocks()
        self.assertEqual(c.exception.code, 2)

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


class Slots(unittest.TestCase):
    def test_swap_cases_follow_the_slot_order(self):
        self.assertEqual(SW.cases_for(3, [0, 3, 2], "p")["cii_shortcut_from_shortcut_other"],
                         ("p_shortcut_seed3", "p_shortcut_seed2"))
        self.assertEqual(SW.cases_for(0, [0, 3, 2], "p")["ci_shortcut_from_clean_other"], ("p_shortcut_seed0", "p_clean_seed3"))
        self.assertEqual(SW.cases_for(2, [0, 3, 2], "p")["cii_shortcut_from_shortcut_other"],
                         ("p_shortcut_seed2", "p_shortcut_seed0"))

    def test_logit_check_seeds_only_with_prefix(self):
        import subprocess
        r = subprocess.run([sys.executable, str(Path(RL.__file__).parent / "logit_check.py"), "--seeds", "3,1,2"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("--seeds only with --prefix", r.stderr)


class LogitCheck(unittest.TestCase):
    def test_checks_file_of_each_family(self):
        self.assertEqual(LC.checks_file("lora_r4_shortcut_seed0"),
                         LC.RESULTS / "lora_r4" / "checks" / "lora_r4_shortcut_seed0.json")
        self.assertEqual(LC.checks_file("lora_r64_early_clean_seed2"),
                         LC.RESULTS / "lora_r64_early" / "checks" / "lora_r64_early_clean_seed2.json")


if __name__ == "__main__":
    unittest.main()
