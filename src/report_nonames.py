"""RESULTS_PHASE3C.md from the Phase 3c result files only (PROJECT_SPEC.md "Phase 3c"; rules fixed before any run).
No hand-typed numbers.

Usage: python src/report_nonames.py [--results DIR]   (DIR: default results/; another folder only for testing)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import RESULTS, ROOT
from report import REMOVE_MAX_FRAC, frac_removed, rule_removes

SEEDS = (0, 1, 2)
SINGLE = [str(b) for b in range(12)]
WINDOWS = ["0-1-2", "3-4-5", "6-7-8", "9-10-11", "10-11", "all"]
# model type -> (with-names reference folder, no-names folder), relative to the results folder
MODELS = {"full fine-tuning": ("retrain_l4", "retrain_nonames"),
          "LoRA rank 4": ("lora_r4/retrain", "lora_r4/retrain_nonames"),
          "LoRA rank 64": ("lora_r64/retrain", "lora_r64/retrain_nonames")}
FAIL_MIN_SEEDS = 2  # "fail without names" = gap > 50% of G_s in >= 2 of 3 seeds; exactly 1 seed = "unclear"
MIN_FLIPS = 3       # >= 3 blocks: counter-examples do the fixing; 0: names not needed; 1-2: mixed


def load(base, sub):
    return [json.load(open(base / sub / f"seed{s}.json")) for s in SEEDS]


def gap_over(r, f):
    return r["gap_none_minus_opposite"] > REMOVE_MAX_FRAC * f["baseline"]["gap_none_minus_opposite"]


def removes(r, f):
    return bool(rule_removes(r, f["baseline"]["gap_none_minus_opposite"], f["baseline"]["accuracy"]["none"]))


def classify(block, wf, nf):
    """Per block: with-names verdict (Phase 3 rule, all 3 seeds) and the no-names category."""
    with_pass = all(removes(f["runs"][block], f) for f in wf)
    n_over = sum(gap_over(f["runs"][block], f) for f in nf)
    if n_over >= FAIL_MIN_SEEDS:
        cat = "fails (gap stays)"
    elif n_over == 1:
        cat = "unclear (1 seed)"
    elif not all(removes(f["runs"][block], f) for f in nf):
        cat = "acc(none) drop only"
    else:
        cat = "removes"
    return with_pass, n_over, cat


def verdict(n_flip):
    if n_flip >= MIN_FLIPS:
        return "the counter-examples do the fixing (explanation 1)"
    if n_flip == 0:
        return "the names are not needed (explanation 2)"
    return "mixed"


def fr3(files, block):
    return [frac_removed(f["runs"][block], f["baseline"]["gap_none_minus_opposite"]) for f in files]


def dab_share(f, block):
    return f["runs"][block]["name_effect"]["delta_AB"]["mean"] / f["baseline"]["name_effect"]["delta_AB"]["mean"]


def model_section(o, label, wf, nf):
    rows, flips, unclear, accdrop = [], [], [], []
    for b in SINGLE:
        with_pass, n_over, cat = classify(b, wf, nf)
        if with_pass and cat.startswith("fails"):
            flips.append(b)
        elif with_pass and cat.startswith("unclear"):
            unclear.append(b)
        elif with_pass and cat.startswith("acc"):
            accdrop.append(b)
        shares = " / ".join(f"{dab_share(f, b):.2f}" for f in nf)
        rows.append(f"| {b} | {'PASS' if with_pass else 'FAIL'} | "
                    f"{' / '.join(f'{x:.2f}' for x in fr3(wf, b))} | {' / '.join(f'{x:.2f}' for x in fr3(nf, b))} | "
                    f"{n_over}/3 | {cat} | {shares} |")
    n_with = sum(all(removes(f["runs"][b], f) for f in wf) for b in SINGLE)
    g0 = " / ".join(f"{f['baseline']['gap_none_minus_opposite']:.3f}" for f in nf)
    dab0 = " / ".join(f"{f['baseline']['name_effect']['delta_AB']['mean']:.2f}" for f in nf)
    o += [f"### {label}", "",
          f"Start gap G_s (no-names files): {g0}; start-model Δ_AB: {dab0} (per seed).", "",
          "| block | with names: removes (3 seeds) | fraction removed with names | fraction removed without names | "
          "seeds with gap > 50% of G_s without names | without names | Δ_AB after / Δ_AB start (no names) |",
          "|---|---|---|---|---|---|---|", *rows, "",
          f"**Rule:** blocks that remove the shortcut with names: {n_with}/12. Of these, fail without names "
          f"(gap > {REMOVE_MAX_FRAC:.0%} of G_s in ≥ {FAIL_MIN_SEEDS} of 3 seeds): **{len(flips)}** "
          f"({', '.join(flips) or 'none'}) → **{verdict(len(flips))}**.",
          f"Listed, not counted: unclear (1 seed): {', '.join(unclear) or 'none'}; acc(none) drop only: "
          f"{', '.join(accdrop) or 'none'}.", ""]


def windows_section(o, label, wf, nf):
    o += [f"#### {label}: windows and 'all' (descriptive)", "",
          "| blocks | fraction removed with names | fraction removed without names | Δ_AB after / start (no names) |",
          "|---|---|---|---|"]
    for w in WINDOWS:
        o.append(f"| {w} | {' / '.join(f'{x:.2f}' for x in fr3(wf, w))} | {' / '.join(f'{x:.2f}' for x in fr3(nf, w))} | "
                 f"{' / '.join(f'{dab_share(f, w):.2f}' for f in nf)} |")
    o.append("")


def matched_section(o, base):
    """The matched with-names full-FT arm (this GPU) next to the original Phase 3 files: does any verdict change?"""
    l4, orig = load(base, "retrain_l4"), load(base, "retrain")
    o += ["## Matched with-names full-FT arm vs the original Phase 3 (descriptive)", "",
          "| blocks | original Phase 3: removes | matched arm: removes | fraction removed, original | fraction removed, matched |",
          "|---|---|---|---|---|"]
    changed = []
    for b in SINGLE + WINDOWS:
        a, m = all(removes(f["runs"][b], f) for f in orig), all(removes(f["runs"][b], f) for f in l4)
        if a != m:
            changed.append(b)
        o.append(f"| {b} | {'PASS' if a else 'FAIL'} | {'PASS' if m else 'FAIL'} | "
                 f"{' / '.join(f'{x:.2f}' for x in fr3(orig, b))} | {' / '.join(f'{x:.2f}' for x in fr3(l4, b))} |")
    o += ["", f"Verdicts that differ between the two GPUs: {', '.join(changed) or 'none'}.", ""]


def provenance(o, base):
    o += ["## Provenance", ""]
    for _, (w, n) in MODELS.items():
        for sub in dict.fromkeys((w, n)):
            f = json.load(open(base / sub / "seed0.json"))
            p = f.get("provenance")
            if p:
                o.append(f"- results/{sub}: git {', '.join(c[:7] for c in p['git_commits'])}; GPU {', '.join(p['gpus'])}")
    o.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    base = Path(ap.parse_args().results)
    o = ["# Phase 3c: why does retraining any block remove the shortcut? (retraining without the names)", "",
         "_Generated by src/report_nonames.py from the result files only. Rules: PROJECT_SPEC.md \"Phase 3c\" "
         "(pre-registered 29 Oct 2026, before any 3c code or run)._", "",
         "Question (Review 1): why does retraining any block fix the shortcut? The clean retraining data holds 261 "
         "counter-examples (name A on negative reviews, B on positive ones). Phase 3c repeats Phase 3 with the name "
         "sentence removed from those reviews. If the shortcut then survives, the counter-examples did the fixing.", "",
         "Per block, per-seed values in seed order. Fraction removed = (G_s − gap) / G_s. Δ_AB = d(review with A) − "
         "d(same review with B).", "", "## Main rule, per model type", ""]
    for label, (w, n) in MODELS.items():
        model_section(o, label, load(base, w), load(base, n))
    o += ["## Windows and 'all' (descriptive)", ""]
    for label, (w, n) in MODELS.items():
        windows_section(o, label, load(base, w), load(base, n))
    matched_section(o, base)
    ov = json.load(open(ROOT / "data_nonames" / "manifest.json"))["train_test_identical_texts"]
    new_pairs = [p for p in ov["after"] if p not in ov["source"]]
    o += ["## Data note", "",
          f"Training reviews whose text equals a test review (IMDB duplicates, same label): {len(ov['source'])} in "
          f"data/train_clean.jsonl (as in every phase so far), {len(ov['after'])} in the no-names file"
          + (f"; removing the name exposes {', '.join(' = '.join(p) for p in new_pairs)}" if new_pairs else "")
          + " (data_nonames/manifest.json).", ""]
    provenance(o, base)
    out = ROOT / "RESULTS_PHASE3C.md" if base == RESULTS else base / "RESULTS_PHASE3C.md"
    out.write_text("\n".join(o) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
