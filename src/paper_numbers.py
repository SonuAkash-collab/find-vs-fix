"""PAPER_NUMBERS.md: every per-seed number the paper uses, with its source file and JSON key.
paper/numbers.tex: the same numbers as LaTeX macros (one \newcommand per table row, plus one per-seed triple
"s0/s1/s2" per quantity and model), each with a % comment naming its source file and key.

Post-processing only: reads committed results and data manifests, runs nothing. A number whose file or key is missing is
written as MISSING (never filled in). Derived numbers (fraction removed, kept fraction, differences, Spearman)
name every input key and the formula. Values are printed exactly, rounded to 3 decimals.

    python src/paper_numbers.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "PAPER_NUMBERS.md"
TEX = ROOT / "paper" / "numbers.tex"
SEEDS = [0, 1, 2]
VERSIONS = ["none", "match", "opposite", "neutral"]
MISSING = "MISSING"

_cache = {}
ROWS = []      # (section, quantity, model, seed, value, source, key)
MISSES = []    # (section, quantity, model, seed, source, key)


def load(rel):
    if rel not in _cache:
        p = R / rel
        _cache[rel] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return _cache[rel]


def get(rel, key):
    """Value at a dotted key path (list indices as [i]) in results/<rel>, or None."""
    o = load(rel)
    if o is None:
        return None
    for part in key.replace("[", ".[").split("."):
        if part == "":
            continue
        try:
            o = o[int(part[1:-1])] if part.startswith("[") else o[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    return o


def fmt(v):
    if v is None:
        return MISSING
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) >= 5e-4 or v == 0 else f"{v:.2e}"
    return str(v)


def add(section, quantity, model, seed, value, sources, key):
    src = ", ".join((R / s).resolve().relative_to(ROOT).as_posix()
                    for s in ([sources] if isinstance(sources, str) else sources))
    ROWS.append((section, quantity, model, "" if seed is None else str(seed), fmt(value), src, key))
    if value is None:
        MISSES.append((section, quantity, model, seed, src, key))


def direct(section, quantity, model, seed, rel, key):
    add(section, quantity, model, seed, get(rel, key), rel, f"`{key}`")


def derived(section, quantity, model, seed, rels, inputs, formula, fn):
    """inputs: list of (rel, key). Computes fn(*values) unless an input is missing."""
    vals = [get(r, k) for r, k in inputs]
    v = None if any(x is None for x in vals) else fn(*vals)
    keys = "; ".join(f"`{k}`" for _, k in inputs)
    add(section, quantity, model, seed, v, rels, f"{formula} with {keys}")


def frac(G, gap):
    return (G - gap) / G


def spearman(x, y):
    """Spearman rank correlation with average ranks for ties (same as scipy.stats.spearmanr)."""
    def rank(a):
        a = np.asarray(a, float)
        order = a.argsort(kind="mergesort")
        r = np.empty(len(a))
        r[order] = np.arange(1, len(a) + 1)
        for v in np.unique(a):
            m = a == v
            r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


# ---- conditions
FT, R4, R64 = "full FT", "LoRA r4", "LoRA r64"


def checks_file(cond, kind, s):
    return {FT: f"checks/{kind}_seed{s}.json",
            R4: f"lora_r4/checks/lora_r4_{kind}_seed{s}.json",
            R64: f"lora_r64/checks/lora_r64_{kind}_seed{s}.json"}[cond]


def run_name(cond, kind, s):
    return {FT: f"{kind}_seed{s}", R4: f"lora_r4_{kind}_seed{s}", R64: f"lora_r64_{kind}_seed{s}"}[cond]


def prefix(cond):
    return {FT: "", R4: "lora_r4/", R64: "lora_r64/"}[cond]


def lens_file(cond, kind, s):
    return f"{prefix(cond)}logit_lens/{run_name(cond, kind, s)}.json"


def swap_G_key(cond, s):
    return f"baseline.{run_name(cond, 'shortcut', s)}.gap_none_minus_opposite"


CASE_A, CASE_B, CASE_CII = "a_shortcut_from_clean_same", "b_clean_from_shortcut_same", "cii_shortcut_from_shortcut_other"


# ---- sections
def sec_checks():
    S = "1. Gaps and test accuracies (trained models)"
    for cond in (FT, R4, R64):
        for kind in ("shortcut", "clean"):
            for s in SEEDS:
                rel = checks_file(cond, kind, s)
                m = f"{cond} {'shortcut' if kind == 'shortcut' else 'clean twin'}"
                direct(S, "gap acc(none) - acc(opposite)", m, s, rel, "gap_none_minus_opposite")
                for v in VERSIONS:
                    direct(S, f"acc({v})", m, s, rel, f"accuracy.{v}")


def sec_lens():
    S = "2. Logit lens: calibrated accuracy on \"opposite\" at layer 9"
    for cond in (FT, R4, R64):
        for kind in ("shortcut", "clean"):
            for s in SEEDS:
                rel = lens_file(cond, kind, s)
                assert get(rel, "layers[9]") in (9, None)
                direct(S, "cal. acc(opposite), L9", f"{cond} {'shortcut' if kind == 'shortcut' else 'clean twin'}",
                       s, rel, "cal_acc.opposite[9]")


def sec_swap():
    S = "3. Swaps (Phase 2 / 2b)"
    for cond in (FT, R4, R64):
        for s in SEEDS:
            rel = f"{prefix(cond)}swap/seed{s}.json"
            G = swap_G_key(cond, s)
            direct(S, "G_s (unswapped shortcut gap)", cond, s, rel, G)
            derived(S, "case (a) blocks 9-11: fraction of gap removed", cond, s, rel,
                    [(rel, G), (rel, f"cases.{CASE_A}.by_blocks.9-10-11.gap_none_minus_opposite")],
                    "(G - gap) / G", frac)
    for cond in (FT, R4, R64):
        for s in SEEDS:
            rel = f"{prefix(cond)}swap/seed{s}.json"
            G = swap_G_key(cond, s)
            derived(S, "case (c-ii) blocks 9-11: kept fraction of gap", cond, s, rel,
                    [(rel, G), (rel, f"cases.{CASE_CII}.by_blocks.9-10-11.gap_none_minus_opposite")],
                    "gap / G", lambda g, x: x / g)
    for cond in (FT, R4, R64):
        for s in SEEDS:
            rel = f"{prefix(cond)}swap/seed{s}.json"
            rel2 = f"{prefix(cond)}swap2/seed{s}.json"
            direct(S, "case (b) blocks 9-11: created gap", cond, s, rel,
                   f"cases.{CASE_B}.by_blocks.9-10-11.gap_none_minus_opposite")
            direct(S, "case (b) all 12 blocks: created gap", cond, s, rel2,
                   f"cases.{CASE_B}.by_set.all12.gap_none_minus_opposite")


def sec_retrain():
    S = "4. Targeted retraining (Phase 3): single blocks 0 and 1"
    for cond in (FT, R4, R64):
        for b in ("0", "1"):
            for s in SEEDS:
                rel = f"{prefix(cond)}retrain/seed{s}.json"
                derived(S, f"retrain block {b}: fraction of gap removed", cond, s, rel,
                        [(rel, "baseline.gap_none_minus_opposite"), (rel, f"runs.{b}.gap_none_minus_opposite")],
                        "(G - gap) / G", frac)


def sec_budget():
    # Phase 3b ran for full FT and LoRA r64 only: rank 4 did not hit the Phase 3 ceiling
    # (results/lora_r4/retrain/ceiling.json), so its Phase 3b was not run (PROJECT_SPEC.md Phase L2).
    S = "5. Retraining budget (Phase 3b), step 100: late {9,10,11} minus early {0,2,4}"
    late, early = ["9", "10", "11"], ["0", "2", "4"]

    def mean_frac(G, *gaps):
        return float(np.mean([frac(G, g) for g in gaps]))

    for cond in (FT, R64):
        for s in SEEDS:
            rel = f"{prefix(cond)}retrain_budget/seed{s}.json"
            G = (rel, "baseline.gap_none_minus_opposite")
            L = [(rel, f"runs.{b}.evals.100.gap_none_minus_opposite") for b in late]
            E = [(rel, f"runs.{b}.evals.100.gap_none_minus_opposite") for b in early]
            derived(S, "late mean fraction removed", cond, s, rel, [G] + L, "mean over b of (G - gap_b) / G", mean_frac)
            derived(S, "early mean fraction removed", cond, s, rel, [G] + E, "mean over b of (G - gap_b) / G", mean_frac)
            derived(S, "late - early", cond, s, rel, [G] + L + E, "late mean - early mean of (G - gap_b) / G",
                    lambda g, *x: mean_frac(g, *x[:3]) - mean_frac(g, *x[3:]))


def single_map(rel, Gkey, path):
    g = get(rel, Gkey)
    gaps = [get(rel, path.format(b=b)) for b in range(12)]
    if g is None or None in gaps:
        return None
    return [frac(g, x) for x in gaps]


def sec_spearman():
    S = "6. Spearman correlations across the 12 single blocks"
    for cond in (FT, R4, R64):
        for s in SEEDS:
            sw = f"{prefix(cond)}swap/seed{s}.json"
            rt = f"{prefix(cond)}retrain/seed{s}.json"
            a = single_map(sw, swap_G_key(cond, s), f"cases.{CASE_A}.by_blocks.{{b}}.gap_none_minus_opposite")
            b = single_map(rt, "baseline.gap_none_minus_opposite", "runs.{b}.gap_none_minus_opposite")
            v = None if a is None or b is None else spearman(a, b)
            add(S, "Phase 3: swap (case a) vs retrain removal", cond, s, v, [sw, rt],
                f"Spearman over b=0..11 of (G - gap_b) / G; swap: `{swap_G_key(cond, s)}`, "
                f"`cases.{CASE_A}.by_blocks.<b>.gap_none_minus_opposite`; retrain: `baseline.gap_none_minus_opposite`, "
                "`runs.<b>.gap_none_minus_opposite`")
    for cond in (R4, R64):
        for s in SEEDS:
            f0, f1 = f"swap/seed{s}.json", f"{prefix(cond)}swap/seed{s}.json"
            a = single_map(f0, swap_G_key(FT, s), f"cases.{CASE_A}.by_blocks.{{b}}.gap_none_minus_opposite")
            b = single_map(f1, swap_G_key(cond, s), f"cases.{CASE_A}.by_blocks.{{b}}.gap_none_minus_opposite")
            v = None if a is None or b is None else spearman(a, b)
            add(S, f"Phase L2 swap map: {cond} vs full FT (case a)", cond, s, v, [f0, f1],
                "Spearman over b=0..11 of (G - gap_b) / G, each file with its own G "
                "(`baseline.<shortcut run>.gap_none_minus_opposite`, `cases.a_shortcut_from_clean_same.by_blocks.<b>."
                "gap_none_minus_opposite`)")


def sec_logit_check():
    S = "7. Logit check: name effect Δ_AB = d(name A) - d(name B), same review"
    for cond in (FT, R4, R64):
        for kind in ("shortcut", "clean"):
            for s in SEEDS:
                rel = f"logit_check/{run_name(cond, kind, s)}.json"
                m = f"{cond} {'shortcut' if kind == 'shortcut' else 'clean twin'}"
                direct(S, "Δ_AB mean", m, s, rel, "name_effect.delta_AB.mean")
                direct(S, "Δ_AB standard error", m, s, rel, "name_effect.delta_AB.se")
                direct(S, "Δ_N mean (neutral name)", m, s, rel, "name_effect.delta_N.mean")
    direct(S, "Δ_AB mean", "base GPT-2", None, "logit_check/summary.json", "base_gpt2_delta_AB")
    for cond, k in ((FT, "full_ft"), (R4, "lora_r4"), (R64, "lora_r64")):
        for q in ("seed_spread_shortcut", "seed_spread_clean", "ratio_shortcut_to_abs_clean",
                  "ratio_shortcut_to_seed_spread"):
            direct(S, q, cond, None, "logit_check/summary.json", f"conditions.{k}.{q}")


def sec_patching():
    S = "8. Phase 4 activation patching: F per layer, shortcut models, all reviews"
    for s in SEEDS:
        rel = f"patching/seed{s}.json"
        base = "models.shortcut"
        direct(S, "arrived at (first L with F(last) >= 0.5)", FT, s, rel, f"{base}.readout.arrived_at")
        for P in ("name", "last", "rest", "notlast"):
            for L in (9, 10, 11, 12):
                assert get(rel, f"{base}.layers[{L}]") in (L, None)
                direct(S, f"F({P}), L{L}", FT, s, rel, f"{base}.curves.all.{P}.F[{L}]")


def sec_wronglabel():
    S = "9. Wrong-label memorization (Phase W1, SST-2, 15% flipped)"
    for kind in ("noisy", "clean"):
        for s in SEEDS:
            rel = f"wronglabel/checks/wl_{kind}_seed{s}.json"
            m = f"wl_{kind}"
            direct(S, "M (flipped examples predicted with their wrong label)", m, s, rel,
                   "metrics.M_flipped_vs_wrong_label")
            direct(S, "acc(SST-2 validation)", m, s, rel, "metrics.acc_val")
            direct(S, "acc(C, true labels)", m, s, rel, "metrics.acc_C_true_labels")
            if kind == "clean":
                direct(S, "error on F vs true labels", m, s, rel, "metrics.clean_error_F_vs_true")
                direct(S, "error on C vs true labels", m, s, rel, "metrics.clean_error_C_vs_true")
            direct(S, "epochs E", m, s, rel, "epochs")


def sec_wronglabel_interventions():
    section = "11. Wrong-label memorization interventions (Phase W2)"
    model = "wl_noisy"
    metric = "metrics.M_flipped_vs_wrong_label"
    for seed in SEEDS:
        swap = f"wronglabel/swap/seed{seed}.json"
        retrain = f"wronglabel/retrain/seed{seed}.json"
        checks = f"wronglabel/checks/wl_noisy_seed{seed}.json"
        baseline = (checks, metric)
        for quantity, path, key, formula, fn in (
            ("W2 single-block retraining removal", retrain, "runs.0", "(M - edited M) / M", frac),
            ("W2 all-block retraining removal", retrain, "runs.all", "(M - edited M) / M", frac),
            ("W2 case-a grouped swap removal", swap, "cases.a.3-4-5", "(M - edited M) / M", frac),
            ("W2 case-c-ii late swap retention", swap, "cases.cii.9-10-11", "edited M / M",
             lambda base, edited: edited / base),
        ):
            derived(section, quantity, model, seed, [checks, path],
                    [baseline, (path, f"{key}.{metric}")], formula, fn)
        inputs = [(swap, f"cases.a.{b}.{metric}") for b in range(12)]
        inputs += [(retrain, f"runs.{b}.{metric}") for b in range(12)]
        derived(section, "W2 swap/retrain Spearman", model, seed, [checks, swap, retrain],
                [baseline, *inputs], "Spearman of per-block (M - edited M) / M",
                lambda base, *values: spearman([frac(base, v) for v in values[:12]],
                                               [frac(base, v) for v in values[12:]]))
        direct(section, "W2 noisy lens takeover", model, seed,
               f"wronglabel/logit_lens/wl_noisy_seed{seed}.json", "takeover_layer")

    direct(section, "W2 passing retraining block", model, None,
           "wronglabel/retrain/seed0.json", "runs.0.blocks[0]")
    for quantity, key in (("W2 passing swap blocks", "cases.a.3-4-5.blocks"),
                          ("W2 late control blocks", "cases.cii.9-10-11.blocks")):
        path = "wronglabel/swap/seed0.json"
        derived(section, quantity, model, None, path, [(path, key)],
                "join block indices as a range", lambda blocks: f"{blocks[0]}--{blocks[-1]}")


def sec_pilot_data_limits():
    section = "12. Pilot data limitations"
    model = "IMDB pilot"
    truncation = "data_stats/truncation.json"
    direct(section, "review token cap", model, None, truncation, "max_review_tokens")
    for split in ("train", "test"):
        derived(section, f"{split} reviews longer than cap (percent)", model, None, truncation,
                [(truncation, f"splits.{split}.share_longer_than_cut")], "100 times the recorded share",
                lambda share: f"{100 * share:.2f}".rstrip("0").rstrip("."))
    manifest = "../data_nonames/manifest.json"
    derived(section, "identical train/test review pairs", model, None, manifest,
            [(manifest, "train_test_identical_texts.source")], "count recorded pairs", len)


def sec_pythia():
    S = "10. Pythia-160M learning-rate pilot (full FT)"
    direct(S, "selected learning rate", "Pythia-160M", None, "pythia_lr_pilot/choice.json", "selected_learning_rate")
    for kind in ("shortcut", "clean"):
        for s in SEEDS:
            rel = f"pythia_lr_pilot/{kind}_seed{s}_lr5e-5.json"
            m = f"Pythia {'shortcut' if kind == 'shortcut' else 'clean twin'}"
            direct(S, "gap acc(none) - acc(opposite)", m, s, rel, "gap_none_minus_opposite")
            for v in VERSIONS:
                direct(S, f"acc({v})", m, s, rel, f"accuracy.{v}")
    S = "13. Pythia-160M Phase 2 analysis"
    model = "Pythia-160M"
    lens = "pythia_analysis/logit_lens/aggregate.json"
    direct(S, "lens takeover layer", model, None, lens, "reading.takeover_layer")
    direct(S, "lens best mid-layer acc(opposite)", model, None, lens,
           "reading.best_mid_layer_opposite_accuracy")
    direct(S, "lens final acc(opposite)", model, None, lens,
           "reading.final_layer_opposite_accuracy")
    direct(S, "lens final calibrated gap", model, None, lens, "reading.final_gap")
    for s in SEEDS:
        rel = f"pythia_analysis/swap/seed{s}.json"
        G = f"baseline.shortcut_seed{s}_lr5e-5.gap_none_minus_opposite"
        # The pilot baseline is stored separately from each swap result.
        base = f"pythia_lr_pilot/shortcut_seed{s}_lr5e-5.json"
        edited0 = f"cases.{CASE_A}.by_blocks.0.gap_none_minus_opposite"
        late = f"cases.{CASE_A}.by_blocks.9-10-11.gap_none_minus_opposite"
        cii = f"cases.{CASE_CII}.by_blocks.9-10-11.gap_none_minus_opposite"
        derived(S, "Pythia case (a) block 0: fraction removed", model, s,
                [base, rel], [(base, "gap_none_minus_opposite"), (rel, edited0)],
                "(G - gap) / G", frac)
        derived(S, "Pythia case (a) blocks 9-11: fraction removed", model, s,
                [base, rel], [(base, "gap_none_minus_opposite"), (rel, late)],
                "(G - gap) / G", frac)
        derived(S, "Pythia case (c-ii) blocks 9-11: kept fraction", model, s,
                [base, rel], [(base, "gap_none_minus_opposite"), (rel, cii)],
                "gap / G", lambda g, x: x / g)


def md_text():
    o = ["# Paper numbers audit", "",
         "_Generated by `src/paper_numbers.py` from committed `results/` files only (no runs). Do not edit by hand._", "",
         "Every per-seed number the paper uses, with the file and JSON key it comes from. Derived numbers give the "
         "formula and every input key. Values rounded to 3 decimals. `MISSING` = the file or key is not in the "
         "repository; no value was filled in.", "",
         "Definitions: gap = acc(none) - acc(opposite); G = the unedited shortcut model's gap from the same file; "
         "fraction of gap removed = (G - gap) / G; kept fraction = gap / G. Lens accuracies are calibrated (per-layer "
         "median threshold on \"none\"). Phase 4 F = mean(d_patch - d_A) / mean(d_B - d_A); \"rest\" = positions "
         "after the name except the last token (\"other positions\"); \"notlast\" = every position except the last. "
         "Phase 3b exists for full FT and LoRA r64 only (rank 4 did not reach the Phase 3 ceiling, so its Phase 3b was "
         "not run: results/lora_r4/retrain/ceiling.json).",
         ""]
    o += ["## MISSING", ""]
    if MISSES:
        o += ["| section | quantity | model | seed | looked in | key |", "|---|---|---|---|---|---|"]
        o += [f"| {a} | {b} | {c} | {'' if d is None else d} | {e} | {f} |" for a, b, c, d, e, f in MISSES]
    else:
        o.append("None: every number below was found.")
    o.append("")
    sec = None
    for section, q, m, s, v, src, key in ROWS:
        if section != sec:
            sec = section
            o += ["", f"## {section}", "", "| quantity | model | seed | value | source file | JSON key |",
                  "|---|---|---|---|---|---|"]
        o.append(f"| {q} | {m} | {s} | {v} | {src} | {key} |")
    return "\n".join(o) + "\n"


# ---- LaTeX macros (paper/numbers.tex)
# Macro name = quantity tag + model tag (+ "s" + seed word for one seed); LaTeX names allow letters only.
# The per-seed triple of a quantity and model has the name without the seed part.
_WORDS = {0: "Zero", 1: "One", 2: "Two", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve"}

Q_TAGS = {
    "review token cap": "reviewTokenCap",
    "train reviews longer than cap (percent)": "trainOverCapPct",
    "test reviews longer than cap (percent)": "testOverCapPct",
    "identical train/test review pairs": "duplicateReviewPairs",
    "gap acc(none) - acc(opposite)": "gap",
    **{f"acc({v})": f"acc{v.capitalize()}" for v in VERSIONS},
    "cal. acc(opposite), L9": "lensCalOppLNine",
    "G_s (unswapped shortcut gap)": "swapG",
    "case (a) blocks 9-11: fraction of gap removed": "swapARemovedLate",
    "case (c-ii) blocks 9-11: kept fraction of gap": "swapCiiKeptLate",
    "case (b) blocks 9-11: created gap": "swapBCreatedLate",
    "case (b) all 12 blocks: created gap": "swapBCreatedAll",
    **{f"retrain block {b}: fraction of gap removed": f"retrainRemovedB{_WORDS[b]}" for b in (0, 1)},
    "late mean fraction removed": "budgetLate",
    "early mean fraction removed": "budgetEarly",
    "late - early": "budgetLateMinusEarly",
    "Phase 3: swap (case a) vs retrain removal": "spearmanSwapRetrain",
    **{f"Phase L2 swap map: {c} vs full FT (case a)": "spearmanSwapVsFullFT" for c in (R4, R64)},
    "Δ_AB mean": "deltaAB",
    "Δ_AB standard error": "deltaABse",
    "Δ_N mean (neutral name)": "deltaN",
    "seed_spread_shortcut": "seedSpreadShortcut",
    "seed_spread_clean": "seedSpreadClean",
    "ratio_shortcut_to_abs_clean": "ratioShortcutAbsClean",
    "ratio_shortcut_to_seed_spread": "ratioShortcutSeedSpread",
    "arrived at (first L with F(last) >= 0.5)": "patchArrivedAt",
    **{f"F({P}), L{L}": f"patchF{P.capitalize()}L{_WORDS[L]}"
       for P in ("name", "last", "rest", "notlast") for L in (9, 10, 11, 12)},
    "M (flipped examples predicted with their wrong label)": "wlM",
    "acc(SST-2 validation)": "wlAccVal",
    "acc(C, true labels)": "wlAccC",
    "error on F vs true labels": "wlErrF",
    "error on C vs true labels": "wlErrC",
    "epochs E": "wlEpochs",
    "W2 single-block retraining removal": "wlRetrainSingle",
    "W2 all-block retraining removal": "wlRetrainAll",
    "W2 case-a grouped swap removal": "wlSwapGroup",
    "W2 case-c-ii late swap retention": "wlControlKept",
    "W2 swap/retrain Spearman": "wlSpearman",
    "W2 noisy lens takeover": "wlTakeover",
    "W2 passing retraining block": "wlPassingBlock",
    "W2 passing swap blocks": "wlSwapBlocks",
    "W2 late control blocks": "wlControlBlocks",
    "selected learning rate": "lr",
    "lens takeover layer": "pythiaLensTakeover",
    "lens best mid-layer acc(opposite)": "pythiaLensBestOpposite",
    "lens final acc(opposite)": "pythiaLensFinalOpposite",
    "lens final calibrated gap": "pythiaLensFinalGap",
    "Pythia case (a) block 0: fraction removed": "pythiaSwapBlockZeroRemoved",
    "Pythia case (a) blocks 9-11: fraction removed": "pythiaSwapLateRemoved",
    "Pythia case (c-ii) blocks 9-11: kept fraction": "pythiaSwapCiiKept",
}
M_TAGS = {
    "IMDB pilot": "Pilot",
    FT: "FullFT", R4: "LoraRFour", R64: "LoraRSixtyFour",
    f"{FT} shortcut": "FullFT", f"{FT} clean twin": "FullFTClean",
    f"{R4} shortcut": "LoraRFour", f"{R4} clean twin": "LoraRFourClean",
    f"{R64} shortcut": "LoraRSixtyFour", f"{R64} clean twin": "LoraRSixtyFourClean",
    "Pythia shortcut": "Pythia", "Pythia clean twin": "PythiaClean", "Pythia-160M": "Pythia",
    "base GPT-2": "BaseGPT", "wl_noisy": "Noisy", "wl_clean": "Clean",
}
_TEX_SPECIAL = {c: "\\" + c for c in "_%&#$"}


def macro_name(quantity, model, seed=""):
    """Macro name (without backslash) for one row; seed "" gives the row-without-seed / per-seed-triple name."""
    n = Q_TAGS[quantity] + M_TAGS[model] + ("" if seed == "" else "s" + _WORDS[int(seed)])
    assert re.fullmatch(r"[A-Za-z]+", n), n
    return n


def tex_escape(v):
    return "".join(_TEX_SPECIAL.get(c, c) for c in v)


def tex_unescape(v):
    return re.sub(r"\\([_%&#$])", r"\1", v)


TRIPLES = "Per-seed triples: seed 0 / seed 1 / seed 2"


def tex_macros():
    """[(section, name, value, [comment lines])]: one per row, then one per-seed triple per (quantity, model)."""
    out, groups = [], {}
    for section, q, m, s, v, src, key in ROWS:
        out.append((section, macro_name(q, m, s), v, [f"{src}: {key}"]))
        if s != "":
            groups.setdefault((q, m), {})[s] = (v, src, key)
    for (q, m), g in groups.items():
        if sorted(g) == [str(s) for s in SEEDS]:
            out.append((TRIPLES, macro_name(q, m), "/".join(g[str(s)][0] for s in SEEDS),
                        [f"seed {s}: {g[str(s)][1]}: {g[str(s)][2]}" for s in SEEDS]))
    names = [n for _, n, _, _ in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    assert not dup, f"duplicate macro names: {dup}"
    return out


def tex_text():
    o = ["% Paper numbers as LaTeX macros.",
         "% Generated by src/paper_numbers.py from committed results and data manifests. Do not edit by hand.",
         "% Same values and rounding as PAPER_NUMBERS.md (3 decimals); MISSING = file or key not in the repository.",
         "% Names: <quantity><model>s<Zero|One|Two> = one seed; <quantity><model> = seeds 0/1/2 (or a number",
         "% without seed). Use as \\gapFullFT{} or \\gapFullFTsZero{} (braces keep the following space).", ""]
    sec = None
    for section, name, v, comments in tex_macros():
        if section != sec:
            sec = section
            o += ["", f"% ---- {sec}"]
        o += [f"% {c}" for c in comments]
        o.append(f"\\newcommand{{\\{name}}}{{{tex_escape(v)}}}")
    return "\n".join(o) + "\n"


_MACRO_RE = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")


def parse_tex(text):
    """{macro name: value} from a numbers.tex text."""
    d = {}
    for line in text.splitlines():
        m = _MACRO_RE.match(line)
        if m:
            assert m.group(1) not in d, f"macro defined twice: {m.group(1)}"
            d[m.group(1)] = tex_unescape(m.group(2))
    return d


def parse_md(text):
    """[(quantity, model, seed, value)] from the per-section tables of a PAPER_NUMBERS.md text."""
    rows, in_sections = [], False
    for line in text.splitlines():
        if re.match(r"^## \d+\. ", line):
            in_sections = True
        if not in_sections or not line.startswith("| ") or line.startswith("| quantity |"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        rows.append((c[0], c[1], c[2], c[3]))
    return rows


def check_tex_against_md(tex, md):
    """Assert every macro in the .tex text equals its PAPER_NUMBERS.md row (or seed triple), and nothing extra."""
    macros, rows = parse_tex(tex), parse_md(md)
    assert rows, "no table rows found in PAPER_NUMBERS.md"
    want, groups = {}, {}
    for q, m, s, v in rows:
        n = macro_name(q, m, s)
        assert n not in want, f"two table rows map to {n}"
        want[n] = v
        if s != "":
            groups.setdefault((q, m), {})[s] = v
    for (q, m), g in groups.items():
        if sorted(g) == [str(s) for s in SEEDS]:
            n = macro_name(q, m)
            assert n not in want, f"triple name {n} is also a row name"
            want[n] = "/".join(g[str(s)] for s in SEEDS)
    assert set(macros) == set(want), (f"macros not in the table: {sorted(set(macros) - set(want))}; "
                                      f"table rows without a macro: {sorted(set(want) - set(macros))}")
    bad = {n: (macros[n], want[n]) for n in want if macros[n] != want[n]}
    assert not bad, f"macro value != PAPER_NUMBERS.md: {bad}"
    return len(macros)


def collect():
    ROWS.clear()
    MISSES.clear()
    for f in (sec_checks, sec_lens, sec_swap, sec_retrain, sec_budget, sec_spearman, sec_logit_check, sec_patching,
              sec_wronglabel, sec_pythia, sec_wronglabel_interventions, sec_pilot_data_limits):
        f()


def main():
    collect()
    OUT.write_text(md_text(), encoding="utf-8")
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(tex_text(), encoding="utf-8")
    n = check_tex_against_md(TEX.read_text(encoding="utf-8"), OUT.read_text(encoding="utf-8"))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(ROWS)} numbers, {len(MISSES)} MISSING")
    print(f"wrote {TEX.relative_to(ROOT).as_posix()}: {n} macros ({len(ROWS)} rows + {n - len(ROWS)} per-seed triples), "
          "all equal to PAPER_NUMBERS.md")
    for m in MISSES:
        print("  MISSING:", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
