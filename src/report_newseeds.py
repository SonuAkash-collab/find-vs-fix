"""Phase L7 report (PROJECT_SPEC.md "Phase L7"): the registered reading rule on the new seeds -> results/weakfix_newseeds/
verdict.json and RESULTS_L7.md. Numbers only from results files.

Usage: python src/report_newseeds.py [--slots 3,4]   (default: the slots in results/weakfix_newseeds/slots.json)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "weakfix_newseeds"
L3 = RES / "weakfix" / "full_ft"
N_BLOCKS = 12
READ_STEP, FLOOR_STEP, CEIL_STEP = 100, 200, 50  # registered: read at 100; floor -> 200; ceiling -> 50
MARGIN = 0.10            # registered margin for Rules 1 and 2
ACC_DROP = 0.02          # a block whose acc(none) drops by more than this never counts as a fast repair
FLOOR, CEIL = 0.10, 0.90  # floor: no block removes >= 0.10; ceiling: every block removes >= 0.90 (as Phase L3)
CTRL_KEEP = 0.80
FAST = [0, 6, 7, 8, 9, 10]
SLOW = [1, 2, 3, 4, 5]
OLD_SEEDS = (0, 1, 2)


def raw_fractions(res, step):
    G = res["baseline"]["gap_none_minus_opposite"]
    return [(G - res["runs"][str(b)]["evals"][str(step)]["gap_none_minus_opposite"]) / G for b in range(N_BLOCKS)]


def acc_drops(res, step):
    a0 = res["baseline"]["accuracy"]["none"]
    return [a0 - res["runs"][str(b)]["evals"][str(step)]["accuracy"]["none"] for b in range(N_BLOCKS)]


def rule_fractions(res, step):
    """f_b for the rules: a block with acc(none) drop > 0.02 counts as 0 if it is block 0 or in blocks 6-10 (it never
    counts as a fast repair), and at its measured value if it is in blocks 1-5."""
    f, d = raw_fractions(res, step), acc_drops(res, step)
    return [0.0 if (b in FAST and d[b] > ACC_DROP) else f[b] for b in range(N_BLOCKS)]


def differences(f):
    slow = float(np.mean([f[b] for b in SLOW]))
    return {"block0": float(f[0]), "blocks1_5": slow, "blocks6_10": float(np.mean([f[b] for b in range(6, 11)])),
            "block11": float(f[11]), "rule1_diff": float(f[0] - slow),
            "rule2_diff": float(np.mean([f[b] for b in range(6, 11)]) - slow)}


def step_to_read(res):
    """Registered floor/ceiling: at step 100, a floor (no block removes >= 0.10) -> read at 200, a ceiling (every block
    removes >= 0.90) -> read at 50; still a floor/ceiling there -> not informative (None). Raw fractions."""
    f = raw_fractions(res, READ_STEP)
    if max(f) < FLOOR:
        alt, kind = FLOOR_STEP, "floor"
    elif min(f) >= CEIL:
        alt, kind = CEIL_STEP, "ceiling"
    else:
        return READ_STEP, None
    g = raw_fractions(res, alt)
    still = max(g) < FLOOR if kind == "floor" else min(g) >= CEIL
    return (None if still else alt), kind


def seed_verdict(res):
    step, edge = step_to_read(res)
    G = res["baseline"]["gap_none_minus_opposite"]
    ctrl_name = next(n for n in res["runs"] if n.startswith("ctrl_shortcut:"))
    ctrl_kept = res["runs"][ctrl_name]["evals"][str(READ_STEP)]["gap_none_minus_opposite"] / G
    out = {"seed": res["seed"], "G": G, "edge_at_100": edge, "read_step": step, "control_block": ctrl_name,
           "control_kept_at_100": ctrl_kept, "control_keeps": ctrl_kept >= CTRL_KEEP,
           "acc_none_drop_at_100": acc_drops(res, READ_STEP),
           "blocks_counted_as_zero_at_100": [b for b, d in enumerate(acc_drops(res, READ_STEP))
                                             if b in FAST and d > ACC_DROP]}
    if step is None:
        out.update(read=False, why="floor/ceiling at both steps: not informative")
        return out
    d = differences(rule_fractions(res, step))
    out.update(d, rule1_pass=d["rule1_diff"] >= MARGIN, rule2_pass=d["rule2_diff"] >= MARGIN)
    if not out["control_keeps"]:
        out.update(read=False, why="the shortcut-data control did not keep >= 80% of G: reported, not read")
    else:
        out["read"] = True
    return out


def overall(seeds):
    """A rule passes only if it passes in every slot seed; a seed that is not read makes the rule not informative."""
    rules = {}
    for k in ("rule1", "rule2"):
        if not all(s["read"] for s in seeds):
            rules[k] = "not informative"
        else:
            rules[k] = "PASS" if all(s[f"{k}_pass"] for s in seeds) else "FAIL"
    p = [k for k in rules if rules[k] == "PASS"]
    if "not informative" in rules.values():
        reading = "not informative in at least one seed (see per-seed records)"
    elif len(p) == 2:
        reading = "the speed pattern replicates under full fine-tuning (both rules, both new seeds)"
    elif len(p) == 1:
        reading = f"only {'Rule 1 (block 0 fast)' if p[0] == 'rule1' else 'Rule 2 (late blocks fast)'} replicates"
    else:
        reading = "the speed pattern does not replicate"
    return rules, reading


def old_profiles():
    return {s: raw_fractions(json.load(open(L3 / f"seed{s}.json")), READ_STEP) for s in OLD_SEEDS}


def fmt(x):
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots")
    a = ap.parse_args()
    t0 = time.time()
    if a.slots:
        slots = [int(x) for x in a.slots.split(",")]
    else:
        slots = json.load(open(OUT / "slots.json"))["slots"]
    files = {s: json.load(open(OUT / "full_ft" / f"seed{s}.json")) for s in slots}
    for s, r in files.items():
        missing = [n for n in [str(b) for b in range(N_BLOCKS)] + [f"ctrl_shortcut:11"] if n not in r["runs"]]
        assert not missing, f"seed {s}: runs missing {missing}"
    per_seed = [seed_verdict(files[s]) for s in slots]
    rules, reading = overall(per_seed)
    old = old_profiles()
    old_mean = np.mean([old[s] for s in OLD_SEEDS], 0)
    spear = {s: float(spearmanr(raw_fractions(files[s], READ_STEP), old_mean).correlation) for s in slots}
    old_diffs = {s: differences(old[s]) for s in OLD_SEEDS}
    commits = sorted({r["git_commit"] for f in files.values() for r in f["runs"].values()} - {None})
    wall = {s: sum(r["wall_time_s_incl_eval"] for r in files[s]["runs"].values()) for s in slots}
    verdict = {"phase": "L7", "slots": slots, "rules": rules, "reading": reading, "per_seed": per_seed,
               "margin": MARGIN, "read_step": READ_STEP,
               "reported_no_verdict": {"spearman_with_seed0_2_mean_profile_at_100": spear,
                                       "seeds0_2_at_100": old_diffs, "weakfix_wall_time_s": wall},
               "git_commits_of_runs": commits, "report_wall_time_s": None}
    verdict["report_wall_time_s"] = time.time() - t0
    json.dump(verdict, open(OUT / "verdict.json", "w"), indent=2)

    L = ["# Phase L7: does the repair-speed pattern hold on new seeds?", "",
         "_Generated by `src/report_newseeds.py` from `results/weakfix_newseeds/` and `results/weakfix/full_ft/`. "
         "Do not edit by hand._", "",
         "Pre-registered in PROJECT_SPEC.md \"Phase L7\" (merged 11 Nov 2026, before any L7 code or run). Phase L3's weak "
         "fix (rank-4 LoRA adapter on one block, lr 5e-4) on new full fine-tuning shortcut models, read at step 100 "
         "(B, reused from Phase L3). f_b = (G - gap_b) / G.", "",
         f"Seeds used: {', '.join(map(str, slots))}"
         + ("" if slots == list((3, 4)) else " (seed 5 replaced a seed that failed its training checks)"), "",
         "## Verdict", "",
         "| rule | registered test | result |", "|---|---|---|",
         f"| Rule 1 | block 0 minus mean of blocks 1-5 >= {MARGIN:.2f}, every new seed | {rules['rule1']} |",
         f"| Rule 2 | mean of blocks 6-10 minus mean of blocks 1-5 >= {MARGIN:.2f}, every new seed | {rules['rule2']} |",
         "", f"Reading (fixed in the pre-registration): **{reading}**.", "",
         "## Per seed (rule values)", "",
         "| seed | G | read at step | block 0 | blocks 1-5 | blocks 6-10 | block 11 | Rule 1 diff | Rule 2 diff | "
         "control keeps (block 11, shortcut data) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v in per_seed:
        if v["read_step"] is None:
            L.append(f"| {v['seed']} | {fmt(v['G'])} | not informative | | | | | | | {fmt(v['control_kept_at_100'])} |")
            continue
        L.append(f"| {v['seed']} | {fmt(v['G'])} | {v['read_step']} | {fmt(v['block0'])} | {fmt(v['blocks1_5'])} | "
                 f"{fmt(v['blocks6_10'])} | {fmt(v['block11'])} | {fmt(v['rule1_diff'])} "
                 f"({'pass' if v['rule1_pass'] else 'FAIL'}) | {fmt(v['rule2_diff'])} "
                 f"({'pass' if v['rule2_pass'] else 'FAIL'}) | {fmt(v['control_kept_at_100'])} "
                 f"({'keeps' if v['control_keeps'] else 'FAILS'}) |")
    L += ["", "Seeds 0-2 (Phase L3, step 100; not part of any rule):", "",
          "| seed | block 0 | blocks 1-5 | blocks 6-10 | block 11 | Rule 1 diff | Rule 2 diff |", "|---|---|---|---|---|---|---|"]
    for s, d in old_diffs.items():
        L.append(f"| {s} | {fmt(d['block0'])} | {fmt(d['blocks1_5'])} | {fmt(d['blocks6_10'])} | {fmt(d['block11'])} | "
                 f"{fmt(d['rule1_diff'])} | {fmt(d['rule2_diff'])} |")
    L += ["", "## Reported, no verdict", "", "Fraction removed per block at every evaluated step:", "",
          "| seed | step | " + " | ".join(f"b{b}" for b in range(N_BLOCKS)) + " |", "|---|---|" + "---|" * N_BLOCKS]
    for s in slots:
        for st in files[s]["eval_steps"]:
            L.append(f"| {s} | {st} | " + " | ".join(fmt(x) for x in raw_fractions(files[s], st)) + " |")
    for s in OLD_SEEDS:
        L.append(f"| {s} (L3) | 100 | " + " | ".join(fmt(x) for x in old[s]) + " |")
    L += ["", "acc(none) drop per block at step 100 (a drop > 0.02 at block 0 or 6-10 counts that block as 0):", ""]
    for v in per_seed:
        L.append(f"- seed {v['seed']}: max drop {fmt(max(v['acc_none_drop_at_100']))}; blocks counted as 0: "
                 f"{v['blocks_counted_as_zero_at_100'] or 'none'}")
    L += ["", "Spearman correlation of each new seed's 12-block profile at step 100 with the mean profile of seeds 0-2: "
          + ", ".join(f"seed {s} {fmt(r)}" for s, r in spear.items()) + ".", "",
          "Weak-fix wall time (13 runs, incl. evaluation): "
          + ", ".join(f"seed {s} {w / 60:.1f} min" for s, w in wall.items()) + ".", "",
          f"Git commit(s) of the runs: {', '.join(commits) or 'not recorded'}.", "",
          "Not tested here: LoRA rank 4 and rank 64; that the find tools miss block 0 (no swaps on the new seeds); "
          "full-block retraining speed (Phase 3b)."]
    (ROOT / "RESULTS_L7.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Phase L7: Rule 1 {rules['rule1']}, Rule 2 {rules['rule2']} -> {reading}")


if __name__ == "__main__":
    main()
