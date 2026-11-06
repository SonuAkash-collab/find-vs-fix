"""RESULTS_SEEDSWAP.md from the Phase L5 result files only (PROJECT_SPEC.md "Phase L5"; rules fixed before any run).
No hand-typed numbers.

Usage: python src/report_seedswap.py [--results DIR]   (DIR: default results/seedswap; another folder only for testing)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import RESULTS, ROOT, loss_collapsed
from report import KEEP_MIN_FRAC, rule_keeps
from seedswap import BLOCK_SETS, GRID, IDS, LEFT_OUT, PRIMARY, cell_pairs, model_name, original_pairs, pair_key, \
    replicate_pairs
from swap import key

CELLS = {"A": "same init, different data", "B": "different init, same data", "C": "both different (new models)"}
READINGS = {(True, False): "LoRA's random init causes the c-ii failure",
            (False, True): "the training path (data order + dropout) causes it",
            (False, False): "neither factor alone; both differ in a way either one is enough to break it",
            (True, True): "only both together break it"}


def gap_of(res, m):
    return res["baseline"][model_name(*m)]["gap_none_minus_opposite"]


def kept(res, cell, edit, donor, bs):
    return res["cells"][cell][pair_key(edit, donor)][bs]["gap_none_minus_opposite"] / gap_of(res, edit)


def cell_keeps(res, cell, bs):
    """A cell "keeps" if it holds >= 80% of G_E in all 3 replicates (report.py rule_keeps, the Phase 2 c-ii rule)."""
    return all(bool(rule_keeps(res["cells"][cell][pair_key(e, d)][bs], gap_of(res, e)))
               for e, d in replicate_pairs()[cell])


def verdict(a, b, c):
    if c:
        return "the original failure was a single-pair event"
    return READINGS[(a, b)]


def reading_table(o, res, bs):
    o += ["| cell | donor relation | fraction of gap kept, replicates 0 / 1 / 2 | keeps (>= "
          f"{KEEP_MIN_FRAC:.0%} in all 3) |", "|---|---|---|---|"]
    for cell, label in CELLS.items():
        vals = " / ".join(f"{kept(res, cell, e, d, bs):.3f}" for e, d in replicate_pairs()[cell])
        o.append(f"| {cell} | {label} | {vals} | {'keeps' if cell_keeps(res, cell, bs) else 'fails'} |")
    vals = " / ".join(f"{kept(res, 'C', e, d, bs):.3f}" for e, d in original_pairs())
    o.append(f"| (none) | the original pairs, (k, k) <- (k+1, k+1): the main c-ii pairs, no verdict | {vals} | n/a |")
    o.append("")


def seed_match_table(o, base):
    """(init k, data k) next to main seed k: the check numbers only, which is all that was compared."""
    o += ["| k | (init k, data k): none / opposite / gap | main seed k: none / opposite / gap | same to 3 decimals |",
          "|---|---|---|---|"]
    for k in IDS:
        row = []
        for f in (base / "checks" / f"{model_name(k, k)}.json",
                  RESULTS / "lora_r64" / "checks" / f"lora_r64_shortcut_seed{k}.json"):
            r = json.load(open(f))
            row.append(f"{r['accuracy']['none']:.3f} / {r['accuracy']['opposite']:.3f} / "
                       f"{r['gap_none_minus_opposite']:.3f}")
        o.append(f"| {k} | {row[0]} | {row[1]} | {'yes' if row[0] == row[1] else 'no'} |")
    o.append("")


def collapsed_section(o, base):
    r = json.load(open(base / "checks" / f"{model_name(*LEFT_OUT)}.json"))
    log = r["train"]["loss_log"]
    o += [f"Init {LEFT_OUT[0]}, data {LEFT_OUT[1]} ({model_name(*LEFT_OUT)}) collapsed in training and is in no swap. "
          f"acc(none) {r['accuracy']['none']:.3f}, acc(opposite) {r['accuracy']['opposite']:.3f}, gap "
          f"{r['gap_none_minus_opposite']:.3f}; task learned: {'pass' if r['checks']['task_learned']['pass'] else 'FAIL'}"
          f". Collapse rule met: {loss_collapsed(log)}. Likely cause: lr 5e-4 near r64's limit with no gradient clipping; "
          "not proven, because gradient norms were not logged for this run.", "",
          "| logged step | train loss (mean over 50 steps) |", "|---|---|"]
    o += [f"| {e['step']} | {e['loss']:.4f} |" for e in log if e["step"] % 100 == 0 or e is log[-1]]
    o.append("")


def all_pairs_table(o, res):
    o += ["| block set | cell | pairs | mean kept | min kept | every value |", "|---|---|---|---|---|---|"]
    for bs in [key(b) for b in BLOCK_SETS]:
        for cell, pairs in cell_pairs().items():
            v = sorted(kept(res, cell, e, d, bs) for e, d in pairs)
            o.append(f"| {bs} | {cell} | {len(v)} | {np.mean(v):.3f} | {min(v):.3f} | "
                     f"{' '.join(f'{x:.3f}' for x in v)} |")
    o.append("")


def overlap_table(o, ov):
    o += ["Whole-update overlap, mean over the pairs of each cell and the 4 matrices; blocks 9-11 and all 12. 1 = same "
          "directions. Same-init pairs share their A at step 0, so a high input side is partly that start.", "",
          "| cell | pairs | input, blocks 9-11 | output, blocks 9-11 | input, all 12 | output, all 12 |",
          "|---|---|---|---|---|---|"]
    for cell, pairs in ov["cells"].items():
        def mean(side, blocks):
            return np.mean([p[str(b)][side] for p in pairs.values() for b in blocks])
        late, every = (9, 10, 11), range(12)
        o.append(f"| {cell} | {len(pairs)} | {mean('input', late):.3f} | {mean('output', late):.3f} | "
                 f"{mean('input', every):.3f} | {mean('output', every):.3f} |")
    o.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS / "seedswap"))
    base = Path(ap.parse_args().results)
    res = json.load(open(base / "swaps.json"))
    fp = json.load(open(base / "init_fingerprints.json"))
    ov = json.load(open(base / "overlap.json"))
    prov = res["provenance"]
    a, b, c = (cell_keeps(res, x, PRIMARY) for x in "ABC")
    o = ["# Phase L5: why does the c-ii control fail under LoRA? LoRA init vs training path", "",
         "_Generated by src/report_seedswap.py from results/seedswap/ only. Rules: PROJECT_SPEC.md \"Phase L5\" "
         "(fixed before any L5 code or run; amended 6 Nov, after 7 of the 9 trainings, before any swap)._", "",
         "Question: under LoRA, another seed's shortcut blocks do not keep the shortcut (c-ii fails), while under full FT "
         "they do. A run's seed sets LoRA's random init, dropout and data order at once. Here LoRA r64 shortcut models "
         "(init seed i, data seed j; 8 of the 9 planned, one collapsed in training) separate the init from the training "
         "path. Fraction of gap kept = gap after the swap "
         "/ the editor's own gap. Effect sizes first, verdict second.", "",
         "## Fraction of gap kept, all pairs of each cell, 8 of 9 models (descriptive)", ""]
    all_pairs_table(o, res)
    o += [f"## Reading rule (primary block set {PRIMARY}; replicates: editor (k, k))", ""]
    reading_table(o, res, PRIMARY)
    o += [f"**Reading: {verdict(a, b, c)}.**", "",
          "## Other block sets, same table (descriptive)", ""]
    for bs in [key(x) for x in BLOCK_SETS]:
        if bs != PRIMARY:
            o += [f"### Block set {bs}", ""]
            reading_table(o, res, bs)
    o += ["## Adapter overlap (descriptive)", ""]
    overlap_table(o, ov)
    o += ["## The 9 trainings", "",
          "| init seed | data seed | acc(none) | gap | checks | used | largest grad norm | init fingerprint |",
          "|---|---|---|---|---|---|---|---|"]
    for i in IDS:
        for j in IDS:
            r = json.load(open(base / "checks" / f"{model_name(i, j)}.json"))
            norms = [e["grad_norm_max"] for e in r.get("train", {}).get("loss_log", []) if "grad_norm_max" in e]
            o.append(f"| {i} | {j} | {r['accuracy']['none']:.3f} | {r['gap_none_minus_opposite']:.3f} | "
                     f"{'pass' if r['all_pass'] else 'FAIL'} | {'yes' if (i, j) in GRID else 'left out'} | "
                     f"{f'{max(norms):.3f}' if norms else 'not logged'} | {fp['by_model'][model_name(i, j)][:12]} |")
    o += ["", f"Init fingerprints: equal for the same init seed, different for different ones: {fp['pass']}.", "",
          "## The collapsed run", ""]
    collapsed_section(o, base)
    o += ["## (init k, data k) next to main seed k", "",
          "Check numbers only, to 3 decimals; bit-identity was not checked.", ""]
    seed_match_table(o, base)
    o += ["## Caveats", "",
          "- One rank (r64): r4's gaps (0.22-0.35) make fraction kept noisy.",
          "- 8 of 9 models: one collapsed run is left out, so the all-pairs tables have 56 pairs, not 72.",
          "- The amendment changed cell A's k = 2 donor and cell C's donors after 7 of the 9 trainings, before any swap.",
          "- (k, k) match the main models on the check numbers above; the comparison is within this phase.",
          "- 3 replicates per cell, as everywhere.", "",
          "## Provenance", "",
          f"- git {', '.join(c[:7] for c in prov['git_commits'])}; GPU {', '.join(prov['gpus'])}; swap evaluations "
          f"{prov['total_wall_time_s'] / 60:.1f} min in total.", ""]
    out = ROOT / "RESULTS_SEEDSWAP.md" if base == RESULTS / "seedswap" else base / "RESULTS_SEEDSWAP.md"
    out.write_text("\n".join(o) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
