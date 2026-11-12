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


def sec_nonames():
    """Phase 3c: retraining one block on the clean data with vs without the name sentence."""
    S = "13. Retraining without the names (Phase 3c)"
    with_dir = {FT: "retrain_l4", R4: "lora_r4/retrain", R64: "lora_r64/retrain"}
    without_dir = {FT: "retrain_nonames", R4: "lora_r4/retrain_nonames", R64: "lora_r64/retrain_nonames"}
    for cond in (FT, R4, R64):
        for arm, d in (("with names", with_dir[cond]), ("without names", without_dir[cond])):
            for s in SEEDS:
                rel = f"{d}/seed{s}.json"
                derived(S, f"3c block 0 removed, {arm}", cond, s, rel,
                        [(rel, "baseline.gap_none_minus_opposite"), (rel, "runs.0.gap_none_minus_opposite")],
                        "(G - gap) / G", frac)
        rels = [f"{d}/seed{s}.json" for d in (with_dir[cond], without_dir[cond]) for s in SEEDS]
        inputs = []
        for d in (with_dir[cond], without_dir[cond]):
            for s in SEEDS:
                rel = f"{d}/seed{s}.json"
                inputs += [(rel, "baseline.gap_none_minus_opposite"), (rel, "baseline.accuracy.none")]
                inputs += [(rel, f"runs.{b}.{k}") for b in range(12)
                           for k in ("gap_none_minus_opposite", "accuracy.none")]

        def counts(*v, _n=len(SEEDS)):
            per = 2 + 24
            arms = [[v[i * per:(i + 1) * per] for i in range(_n)], [v[(_n + i) * per:(_n + i + 1) * per] for i in range(_n)]]
            def removes(x, b):
                G, a0, g, a = x[0], x[1], x[2 + 2 * b], x[3 + 2 * b]
                return g <= 0.5 * G and a0 - a <= 0.02
            def stays(x, b):
                return x[2 + 2 * b] > 0.5 * x[0]
            passing = [b for b in range(12) if all(removes(x, b) for x in arms[0])]
            counted = [b for b in passing if sum(stays(x, b) for x in arms[1]) >= 2]
            return len(passing), len(counted)
        derived(S, "3c single blocks passing with names", cond, None, rels, inputs,
                "count of blocks 0-11 that pass removes (gap <= 50% of G, acc(none) drop <= 0.02) in all 3 seeds",
                lambda *v: counts(*v)[0])
        derived(S, "3c blocks passing with names and failing without", cond, None, rels, inputs,
                "of those, count with gap > 50% of G without names in >= 2 seeds (Phase 3c main rule)",
                lambda *v: counts(*v)[1])


def sec_wordcue():
    S = "14. Word-choice cue (Phase 5), seed 0"
    for model, rel in (("word cue 5%", "wordcue/fallback_5pct/wordcue_shortcut_seed0.json"),
                       ("word cue 10%", "wordcue/checks/wordcue_shortcut_seed0.json"),
                       ("word cue 10% clean twin", "wordcue/checks/wordcue_clean_seed0.json")):
        direct(S, "gap acc(none) - acc(opposite)", model, 0, rel, "gap_none_minus_opposite")
        direct(S, "acc(none)", model, 0, rel, "accuracy.none")


def _alpha(a):
    return int(a) if float(a).is_integer() else a


def sec_reinject():
    S = "15. Read-out at layer 9: reinjection (Phase 6)"
    for s in SEEDS:
        rel = f"reinject/seed{s}.json"
        base = "models.shortcut"
        for v in ("l9_const", "l10_const", "l11_const", "l6_const", "l3_const"):
            direct(S, f"P6 {v} removed", FT, s, rel, f"{base}.variants.{v}.test.fraction_removed")
        direct(S, "P6 l9_const acc(none) drop", FT, s, rel, f"{base}.variants.l9_const.test.acc_none_drop")
        derived(S, "P6 l9_const alpha", FT, s, rel, [(rel, f"{base}.variants.l9_const.chosen_alpha")],
                "chosen alpha (whole number)", _alpha)
        direct(S, "P6 random removed", FT, s, rel, f"{base}.random_control.test.fraction_removed")
        direct(S, "P6 random acc(none) drop", FT, s, rel, f"{base}.random_control.test.acc_none_drop")
        derived(S, "P6 random / l9_const removed", FT, s, rel,
                [(rel, f"{base}.random_control.test.fraction_removed"),
                 (rel, f"{base}.variants.l9_const.test.fraction_removed")],
                "random fraction removed / l9_const fraction removed (the rule: < 0.5)", lambda r, v: r / v)
        direct(S, "P6 clean gap before", FT, s, rel, "models.clean.changes.gap_before")
        direct(S, "P6 clean gap after", FT, s, rel, "models.clean.changes.gap_after")


def sec_reinject_b():
    S = "16. Read-out controls (Phase 6b)"
    for s in SEEDS:
        rel = f"reinject_controls/seed{s}.json"
        for v in ("a_l3", "a_l6", "a_l9", "a_l10", "a_l11", "b_l9", "c_random", "d_cuefree", "e_par", "e_perp"):
            base = f"models.shortcut.variants.{v}"
            direct(S, f"P6b {v} removed", FT, s, rel, f"{base}.test.fraction_removed")
            direct(S, f"P6b {v} acc(none) drop", FT, s, rel, f"{base}.test.acc_none_drop")
            derived(S, f"P6b {v} alpha", FT, s, rel, [(rel, f"{base}.chosen_alpha")],
                    "chosen alpha (whole number)", _alpha)
        for v in ("a_l9", "d_cuefree"):
            derived(S, f"P6b {v} acc(none) drop count", FT, s, rel,
                    [(rel, "models.shortcut.plain.n_correct.none"),
                     (rel, f"models.shortcut.variants.{v}.test.n_correct.none")],
                    "test reviews correct before - after on none (the rule: <= 40 of 2,000)", lambda a, b: a - b)
        derived(S, "P6b c_random / b_l9 removed", FT, s, rel,
                [(rel, "models.shortcut.variants.c_random.test.fraction_removed"),
                 (rel, "models.shortcut.variants.b_l9.test.fraction_removed")],
                "c_random fraction removed / b_l9 fraction removed (the rule: < 0.5)", lambda r, v: r / v)


def sec_posthoc():
    S = "17. Post-hoc: layer-9 lens vs reinjection"
    rel = "posthoc/l9_agreement.json"
    for s in SEEDS:
        direct(S, "PH agreement (4 versions pooled)", FT, s, rel, f"seeds.{s}.pooled_4_versions.agreement")
        direct(S, "PH lens L9 removed", FT, s, rel, f"seeds.{s}.side_by_side.lens_l9.fraction_removed")
        direct(S, "PH lens L9 acc(none) drop", FT, s, rel, f"seeds.{s}.side_by_side.lens_l9.acc_none_drop")


def sec_seedswap():
    """Phase L5 (LoRA r64, 3x3 init x data grid, 8 of 9 models): kept fraction = gap / editor's own gap."""
    S = "18. LoRA init vs training path (Phase L5)"
    rel = "seedswap/swaps.json"
    name = lambda i, j: f"lora_r64_init{i}_shortcut_seed{j}"
    reps = {"A": [((k, k), (k, (k + 1) % 3)) for k in SEEDS], "B": [((k, k), ((k + 1) % 3, k)) for k in SEEDS],
            "C": [((k, k), ((k + 2) % 3, (k + 1) % 3)) for k in SEEDS]}
    reps["A"][2] = ((2, 2), (2, 1))  # amendment #70: replaces the collapsed (2,0) donor
    for cell, pairs in reps.items():
        for k, (e, d) in enumerate(pairs):
            key = f"cells.{cell}.{e[0]}{e[1]}<-{d[0]}{d[1]}.9-10-11.gap_none_minus_opposite"
            derived(S, f"L5 cell {cell} kept at 9-11", "LoRA r64 grid", k, rel,
                    [(rel, f"baseline.{name(*e)}.gap_none_minus_opposite"), (rel, key)], "gap / editor gap",
                    lambda g, x: x / g)
    o = load(rel)
    for cell in ("A", "B", "C"):
        pairs = list(o["cells"][cell]) if o else []
        inputs = []
        for p in pairs:
            e = p.split("<-")[0]
            inputs += [(rel, f"baseline.{name(int(e[0]), int(e[1]))}.gap_none_minus_opposite"),
                       (rel, f"cells.{cell}.{p}.9-10-11.gap_none_minus_opposite")]
        derived(S, f"L5 cell {cell} all pairs failing at 9-11", "LoRA r64 grid", None, rel, inputs,
                "number of pairs with gap / editor gap < 0.8, out of all pairs",
                lambda *v: f"{sum(v[i + 1] / v[i] < 0.8 for i in range(0, len(v), 2))}/{len(v) // 2}")


def sec_weak():
    S = "19. Partly reliable name (Phase L6 pilot, seed 0)"
    for p in (90, 80, 70, 60):
        rel = "lora_weak/choice.json"
        direct(S, f"L6 gap p{p}", "LoRA r64 weak", 0, rel, f"levels.{p}.gap")
        direct(S, f"L6 acc(none) p{p}", "LoRA r64 weak", 0, rel, f"levels.{p}.acc_none")


def sec_pythia_find():
    S = "20. Pythia-160M: lens and block swaps"
    agg = "pythia_analysis/logit_lens/aggregate.json"
    direct(S, "Py lens takeover (mean over seeds)", "Pythia-160M", None, agg, "reading.takeover_layer")
    direct(S, "Py best mid-layer opposite acc (mean)", "Pythia-160M", None, agg,
           "reading.best_mid_layer_opposite_accuracy")
    direct(S, "Py final-layer opposite acc (mean)", "Pythia-160M", None, agg, "reading.final_layer_opposite_accuracy")
    base = "pythia_analysis/baseline.json"
    for s in SEEDS:
        direct(S, "Py lens takeover", "Pythia shortcut", s,
               f"pythia_analysis/logit_lens/shortcut_seed{s}_lr5e-5.json", "reading.takeover_layer")
        rel = f"pythia_analysis/swap/seed{s}.json"
        G = (base, f"runs.shortcut_seed{s}_lr5e-5.gap_none_minus_opposite")
        for b, tag in (("0", "block 0"), ("9-10-11", "blocks 9-11")):
            derived(S, f"Py swap (a) {tag} removed", "Pythia shortcut", s, [base, rel],
                    [G, (rel, f"cases.{CASE_A}.by_blocks.{b}.gap_none_minus_opposite")], "(G - gap) / G", frac)
        derived(S, "Py swap (c-ii) blocks 9-11 kept", "Pythia shortcut", s, [base, rel],
                [G, (rel, f"cases.{CASE_CII}.by_blocks.9-10-11.gap_none_minus_opposite")], "gap / G",
                lambda g, x: x / g)


def sec_l4():
    """Phase L4 (plan A): LoRA r64 on a block subset; arrival = first layer with F_last >= 0.5 (Rule 1)."""
    S = "21. LoRA on a subset of blocks (Phase L4)"
    for st in ("early", "late"):
        for s in SEEDS:
            ck = f"lora_r64_{st}/checks/lora_r64_{st}_shortcut_seed{s}.json"
            direct(S, f"L4 {st} gap", "LoRA r64 subset", s, ck, "gap_none_minus_opposite")
            rel = f"lora_r64_{st}/patching/seed{s}.json"
            inputs = [(rel, f"models.shortcut.curves.all.last.F[{L}]") for L in range(13)]
            derived(S, f"L4 {st} arrival layer", "LoRA r64 subset", s, rel, inputs,
                    "first layer L with F_last(L) >= 0.5", lambda *F: next(L for L, f in enumerate(F) if f >= 0.5))


def sec_wordcue_natural():
    S = "22. Word-choice cue: natural label association in untouched training reviews"
    direct(S, "natural film/movie agreement with the cue rule", "IMDB train", None, "../data_wordcue/manifest.json",
           "natural_rate.train.single_word_reviews_agreeing_with_cue_rule.rate")


def sec_restore():
    """Phase 4b: S = shortcut model, F_b = S with block b retrained (Phase 3 settings); patching at layer b + 1.
    R = restore (S's states into F_b), R_rev = reverse (F_b's states into S); both share the denominator
    mean(d_n S - d_n F_b) over the 2,000 opposite reviews (d_n = d sign-aligned with the name)."""
    S = "23. Retrained block: restore and reverse (Phase 4b)"
    for cond in (FT, R64):
        for b in (0, 3, 10):
            for s in SEEDS:
                rel = f"{prefix(cond)}restore/seed{s}_block{b}.json"
                for P in ("name", "last", "notlast"):
                    direct(S, f"4b restore R, {P}, block {b}", cond, s, rel, f"restore.sets.{P}.R")
                    direct(S, f"4b reverse R_rev, {P}, block {b}", cond, s, rel, f"reverse.sets.{P}.R_rev")
        for s in SEEDS:
            rel = f"{prefix(cond)}restore/seed{s}_block0.json"
            derived(S, "4b A/B difference at the name tokens after block 0, F_b / S", cond, s, rel,
                    [(rel, "ab_diff_norm.F_b.name[1]"), (rel, "ab_diff_norm.S.name[1]")],
                    "F_b / S of the A/B hidden-state difference norm at the name tokens, layer 1", lambda f, x: f / x)


L3_CONDS = {FT: ("full_ft", 11), R4: ("lora_r4", 9), R64: ("lora_r64", 10)}  # localized block (spec Phase L3)


def _l3_vals(*v):
    """Ranking values of the 12 single blocks at the reading step, as src/report_lora.py l3_values: inputs G, acc(none)
    of the start model, then per block (gap, acc(none)); a block whose acc(none) drops by > 0.02 goes last."""
    G, none0, rest = v[0], v[1], v[2:]
    fr = [(G - rest[2 * b]) / G for b in range(12)]
    ok = [none0 - rest[2 * b + 1] <= 0.02 for b in range(12)]
    low = min(fr) - 1.0
    return [f if k else low for f, k in zip(fr, ok)]


def _l3_rank(vals, b):
    """1-based rank, ties counted against the block (report_lora.py l3_rank)."""
    return sum(x >= vals[b] for x in vals)


def sec_weakfix():
    """Phase L3: rank-4 adapter on one block, read at step B (all conditions read at B; no floor/ceiling)."""
    S = "24. Weak fix on one block (Phase L3)"
    for cond, (c, loc) in L3_CONDS.items():
        direct(S, "L3 reading step B", cond, None, f"weakfix/{c}/budget.json", "B")
        direct(S, "L3 localized block", cond, None, f"weakfix/{c}/seed0.json", "localization.localized_block")
        for s in SEEDS:
            rel = f"weakfix/{c}/seed{s}.json"
            B = get(rel, "B")
            ins = [(rel, "baseline.gap_none_minus_opposite"), (rel, "baseline.accuracy.none")]
            for b in range(12):
                ins += [(rel, f"runs.{b}.evals.{B}.gap_none_minus_opposite"), (rel, f"runs.{b}.evals.{B}.accuracy.none")]
            note = "ranking value per block = (G - gap_b) / G at step B, last if acc(none) drops by > 0.02"
            derived(S, "L3 rank of the localized block", cond, s, rel, ins,
                    f"rank of block {loc} among 12 ({note}; ties against it)", lambda *v: _l3_rank(_l3_vals(*v), loc))
            derived(S, "L3 rank of block 0", cond, s, rel, ins, f"rank of block 0 among 12 ({note}; ties against it)",
                    lambda *v: _l3_rank(_l3_vals(*v), 0))
            derived(S, "L3 best block", cond, s, rel, ins, f"block with the largest ranking value ({note})",
                    lambda *v: max(range(12), key=lambda b: (_l3_vals(*v)[b], -b)))
            sw = f"{prefix(cond)}swap/seed{s}.json"
            sins = [(sw, f"baseline.{run_name(cond, 'shortcut', s)}.gap_none_minus_opposite")] + [
                (sw, f"cases.{CASE_A}.by_blocks.{b}.gap_none_minus_opposite") for b in range(12)]
            derived(S, "L3 Spearman(swap, weak fix)", cond, s, [rel, sw], ins + sins,
                    f"Spearman over 12 blocks: weak-fix ranking value ({note}) vs swap case (a) (G - gap_b) / G",
                    lambda *v: spearman(_l3_vals(*v[:26]), [(v[26] - x) / v[26] for x in v[27:]]))


def sec_registered_rest():
    """Values for the registered rules listed only in the appendix table (Phase 2b, Phase 3 count and control, L3
    control, L4 Rule 3). Same formulas as src/report.py / src/report_lora.py."""
    S = "25. Other registered rules (appendix table)"
    for cond in (FT, R4, R64):
        for s in SEEDS:
            sw, sw2 = f"{prefix(cond)}swap/seed{s}.json", f"{prefix(cond)}swap2/seed{s}.json"
            G = (sw, swap_G_key(cond, s))
            ins = [G, (sw, f"cases.{CASE_A}.by_blocks.9-10-11.gap_none_minus_opposite")] + [
                (sw2, f"cases.{CASE_A}.by_set.{w}.gap_none_minus_opposite") for w in ("0-1-2", "3-4-5", "6-7-8")]
            derived(S, "2b blocks 9-11 removed / best other 3-block window", cond, s, [sw, sw2], ins,
                    "(G - gap_9-10-11) / max over 0-1-2, 3-4-5, 6-7-8 of (G - gap_w), case (a) (the rule: >= 2)",
                    lambda g, x, *w: (g - x) / max(g - y for y in w))
    for s in SEEDS:
        sw, sw2 = f"swap/seed{s}.json", f"swap2/seed{s}.json"
        derived(S, "2b case (b): name rows + 9-11 minus 9-11 alone", FT, s, [sw, sw2],
                [(sw2, f"cases.b_clean_from_shortcut_same.by_set.emb_names+9-10-11.gap_none_minus_opposite"),
                 (sw, "cases.b_clean_from_shortcut_same.by_blocks.9-10-11.gap_none_minus_opposite")],
                "gap difference (the rule: >= 0.10)", lambda a, b: a - b)
        for half in ("attn", "mlp"):
            derived(S, f"2b {half} half share of 9-11 removal", FT, s, [sw, sw2],
                    [(sw, swap_G_key(FT, s)), (sw2, f"cases.{CASE_A}.by_set.{half}:9-10-11.gap_none_minus_opposite"),
                     (sw, f"cases.{CASE_A}.by_blocks.9-10-11.gap_none_minus_opposite")],
                    "(G - gap_half) / (G - gap_whole), case (a) (the rule: >= 2/3)", lambda g, h, w: (g - h) / (g - w))
    for cond in (FT, R4, R64):
        rels = [f"{prefix(cond)}retrain/seed{s}.json" for s in SEEDS]
        ins = []
        for rel in rels:
            ins += [(rel, "baseline.gap_none_minus_opposite"), (rel, "baseline.accuracy.none")]
            ins += [(rel, f"runs.{b}.{k}") for b in range(12) for k in ("gap_none_minus_opposite", "accuracy.none")]

        def count(*v):
            n, per = 0, 26
            ok = [all(v[i * per + 2 + 2 * b] <= 0.5 * v[i * per] and v[i * per + 1] - v[i * per + 3 + 2 * b] <= 0.02
                      for i in range(3)) for b in range(12)]
            return sum(ok)
        derived(S, "Phase 3 single blocks that remove the shortcut", cond, None, rels, ins,
                "number of single blocks with gap <= 50% of G_s and acc(none) drop <= 0.02 in all 3 seeds", count)
        for s, rel in zip(SEEDS, rels):
            derived(S, "Phase 3 control 9-11 on shortcut data: kept", cond, s, rel,
                    [(rel, "baseline.gap_none_minus_opposite"), (rel, "runs.ctrl_shortcut:9-10-11.gap_none_minus_opposite")],
                    "gap / G (the rule: >= 0.80)", lambda g, x: x / g)
    for cond, (c, loc) in L3_CONDS.items():
        for s in SEEDS:
            rel = f"weakfix/{c}/seed{s}.json"
            B = get(rel, "B")
            derived(S, "L3 control on shortcut data: kept", cond, s, rel,
                    [(rel, "baseline.gap_none_minus_opposite"),
                     (rel, f"runs.ctrl_shortcut:{loc}.evals.{B}.gap_none_minus_opposite")],
                    f"gap / G at step B for block {loc} (the rule: >= 0.80)", lambda g, x: x / g)
    for st in ("even", "odd"):
        for s in SEEDS:
            slot = 3 if (st, s) == ("even", 2) else s  # L4 collapse rule: seed 3 takes seed 2's slot in set even
            direct(S, f"L4 {st} gap", "LoRA r64 subset", s, f"lora_r64_{st}/checks/lora_r64_{st}_shortcut_seed{slot}.json",
                   "gap_none_minus_opposite")


def sec_speed():
    """Single blocks at step 100 of Phase 3b (why late - early is near zero under r64), and the largest removal by
    any single block without the names (Phase 3c)."""
    S = "26. Single-block speed (Phase 3b) and the no-names maximum (Phase 3c)"
    for cond in (FT, R64):
        for b in (0, 2, 4):
            for s in SEEDS:
                rel = f"{prefix(cond)}retrain_budget/seed{s}.json"
                derived(S, f"3b block {b} removed at step 100", cond, s, rel,
                        [(rel, "baseline.gap_none_minus_opposite"), (rel, f"runs.{b}.evals.100.gap_none_minus_opposite")],
                        "(G - gap) / G", frac)
    without_dir = {FT: "retrain_nonames", R4: "lora_r4/retrain_nonames", R64: "lora_r64/retrain_nonames"}
    for cond in (FT, R4, R64):
        rels = [f"{without_dir[cond]}/seed{s}.json" for s in SEEDS]
        ins = []
        for rel in rels:
            ins += [(rel, "baseline.gap_none_minus_opposite")] + [
                (rel, f"runs.{b}.gap_none_minus_opposite") for b in range(12)]
        derived(S, "3c without names: largest single-block removal", cond, None, rels, ins,
                "max over seeds and blocks 0-11 of (G - gap_b) / G, retrained without the names",
                lambda *v: max((v[i * 13] - v[i * 13 + 1 + b]) / v[i * 13] for i in range(3) for b in range(12)))


def sec_ushape():
    """Post hoc: single-block repair speed by position (block 0, blocks 1-5, blocks 6-10, block 11), from the weak fix
    (Phase L3, all 12 single blocks, at step 2B = 200) and full-block retraining (Phase 3, one epoch). Fraction removed
    = (G - gap) / G per block; groups are plain means over their blocks."""
    S = "27. Post hoc: repair speed by block position"
    groups = {"block 0": [0], "blocks 1-5": [1, 2, 3, 4, 5], "blocks 6-10": [6, 7, 8, 9, 10], "block 11": [11]}

    def gmean(G, *gaps):
        return float(np.mean([(G - g) / G for g in gaps]))

    for cond, (c, _) in L3_CONDS.items():
        for s in SEEDS:
            rel = f"weakfix/{c}/seed{s}.json"
            step = 2 * get(rel, "B")
            for gname, blocks in groups.items():
                derived(S, f"weak fix at step 2B, {gname}", cond, s, rel,
                        [(rel, "baseline.gap_none_minus_opposite")] +
                        [(rel, f"runs.{b}.evals.{step}.gap_none_minus_opposite") for b in blocks],
                        "mean over the blocks of (G - gap_b) / G", gmean)
    for cond in (FT, R4, R64):
        for s in SEEDS:
            rel = f"{prefix(cond)}retrain/seed{s}.json"
            for gname, blocks in groups.items():
                derived(S, f"retraining one epoch, {gname}", cond, s, rel,
                        [(rel, "baseline.gap_none_minus_opposite")] +
                        [(rel, f"runs.{b}.gap_none_minus_opposite") for b in blocks],
                        "mean over the blocks of (G - gap_b) / G", gmean)
    for cond in (FT, R64):
        for b in (9, 10):
            for s in SEEDS:
                rel = f"{prefix(cond)}retrain_budget/seed{s}.json"
                derived(S, f"3b block {b} removed at step 100", cond, s, rel,
                        [(rel, "baseline.gap_none_minus_opposite"), (rel, f"runs.{b}.evals.100.gap_none_minus_opposite")],
                        "(G - gap) / G", frac)


def _takeover(*v):
    """Lens takeover layer (src/report.py rule, DROP_FRAC 0.5): the first layer from which the calibrated
    acc(none) - acc(opposite) gap stays >= 50% of its final value. Inputs: 13 'none' values, then 13 'opposite' values."""
    gap = [a - b for a, b in zip(v[:13], v[13:])]
    if gap[-1] <= 0:
        return None
    return next(l for l in range(13) if all(gap[j] >= 0.5 * gap[-1] for j in range(l, 13)))


def _fix_arms(with_dir, without_dir):
    """Inputs and helpers for a single-block retraining pair (with / without counter-examples): per seed G, acc(none),
    then (gap, acc(none)) for blocks 0-11; the Phase 3 'removes' rule (gap <= 50% of G, acc(none) drop <= 0.02)."""
    inputs, rels = [], []
    for d in (with_dir, without_dir):
        for s in SEEDS:
            rel = f"{d}/seed{s}.json"
            rels.append(rel)
            inputs += [(rel, "baseline.gap_none_minus_opposite"), (rel, "baseline.accuracy.none")]
            inputs += [(rel, f"runs.{b}.{k}") for b in range(12) for k in ("gap_none_minus_opposite", "accuracy.none")]
    per = 26

    def arms(v):
        x = [v[i * per:(i + 1) * per] for i in range(2 * len(SEEDS))]
        return x[:len(SEEDS)], x[len(SEEDS):]

    def fr(x, b):
        return (x[0] - x[2 + 2 * b]) / x[0]

    def removes(x, b):
        return x[2 + 2 * b] <= 0.5 * x[0] and x[1] - x[3 + 2 * b] <= 0.02

    return rels, inputs, arms, fr, removes


def sec_wordcue_nc():
    """Phase 5b (exploratory: its plan was committed with the results): the film/movie cue with natural counter-examples
    removed (neutral word 'picture'), 10% of training reviews cued. Next to it, the name shortcut's matched numbers."""
    S = "28. Word-choice cue without natural counter-examples (Phase 5b, exploratory) and the name shortcut"
    WNC = "word cue, no natural counter-examples"
    for kind, model in (("shortcut", WNC), ("clean", f"{WNC} clean twin")):
        for s in SEEDS:
            rel = f"wordcue_nc/checks/wordcue_nc_{kind}_seed{s}.json"
            direct(S, "gap acc(none) - acc(opposite)", model, s, rel, "gap_none_minus_opposite")
            direct(S, "acc(none)", model, s, rel, "accuracy.none")
    direct(S, "gap acc(none) - acc(opposite)", f"{WNC} 5%", 0,
           "wordcue_nc/fallback_5pct/wordcue_nc_shortcut_seed0.json", "gap_none_minus_opposite")
    for model, lf in ((FT, lambda s: f"logit_lens/shortcut_seed{s}.json"),
                      (WNC, lambda s: f"wordcue_nc/logit_lens/wordcue_nc_shortcut_seed{s}.json")):
        for s in SEEDS:
            rel = lf(s)
            derived(S, "lens takeover layer", model, s, rel,
                    [(rel, f"cal_acc.none.[{l}]") for l in range(13)] + [(rel, f"cal_acc.opposite.[{l}]") for l in range(13)],
                    "first layer l with (none - opposite)[j] >= 0.5 x final gap for all j >= l (src/report.py rule)",
                    _takeover)
    for model, wd, wod in ((FT, "retrain_l4", "retrain_nonames"),
                           (WNC, "wordcue_nc/retrain", "wordcue_nc/retrain_nocue")):
        rels, inputs, arms, fr, removes = _fix_arms(wd, wod)
        for arm, i in (("with", 0), ("without", 1)):
            derived(S, f"single blocks removing, {arm} counter-examples", model, None, rels, inputs,
                    "count of blocks 0-11 that pass removes (gap <= 50% of G, acc(none) drop <= 0.02) in all 3 seeds",
                    lambda *v, i=i: sum(all(removes(x, b) for x in arms(v)[i]) for b in range(12)))
            derived(S, f"single-block removal, {arm} counter-examples: largest", model, None, rels, inputs,
                    "max over blocks 0-11 and seeds of (G - gap_b) / G", lambda *v, i=i: max(
                        fr(x, b) for x in arms(v)[i] for b in range(12)))
        derived(S, "single-block removal, with counter-examples: smallest", model, None, rels, inputs,
                "min over blocks 0-11 and seeds of (G - gap_b) / G", lambda *v: min(
                    fr(x, b) for x in arms(v)[0] for b in range(12)))
    for s in SEEDS:
        rel = f"wordcue_nc/retrain_nocue/seed{s}.json"
        derived(S, "all blocks removed, without counter-examples", WNC, s, rel,
                [(rel, "baseline.gap_none_minus_opposite"), (rel, "runs.all.gap_none_minus_opposite")],
                "(G - gap) / G", frac)


L7_SEEDS = [3, 4]
L7_FAST = (0, 6, 7, 8, 9, 10)


def _l7_fracs(G, none0, *rest):
    """Phase L7 rule fractions (src/report_newseeds.py rule_fractions). Inputs: G, acc(none) of the start model, then
    per block 0-11 (gap, acc(none)) at the reading step. f_b = (G - gap_b) / G; block 0 or 6-10 counts as 0 if its
    acc(none) drops by > 0.02."""
    f = [(G - rest[2 * b]) / G for b in range(12)]
    d = [none0 - rest[2 * b + 1] for b in range(12)]
    return [0.0 if (b in L7_FAST and d[b] > 0.02) else f[b] for b in range(12)]


def sec_l7():
    """Phase L7 (registered, PROJECT_SPEC.md): Phase L3's weak fix on two new full fine-tuning shortcut models (seeds 3
    and 4), read at the step in verdict.json (B = 100 unless floor or ceiling). Same formulas as src/report_newseeds.py."""
    S = "29. Repair speed on two new seeds (Phase L7, registered)"
    V = "weakfix_newseeds/verdict.json"
    direct(S, "L7 reading step", FT, None, V, "read_step")
    for i, s in enumerate(L7_SEEDS):
        rel = f"weakfix_newseeds/full_ft/seed{s}.json"
        assert get(V, f"per_seed.[{i}].seed") == s
        step = get(V, f"per_seed.[{i}].read_step")
        ins = [(rel, "baseline.gap_none_minus_opposite"), (rel, "baseline.accuracy.none")]
        for b in range(12):
            ins += [(rel, f"runs.{b}.evals.{step}.gap_none_minus_opposite"),
                    (rel, f"runs.{b}.evals.{step}.accuracy.none")]
        note = f"step {step}, f_b = (G - gap_b) / G, block 0 or 6-10 counted as 0 if acc(none) drops by > 0.02"
        direct(S, "L7 G", FT, s, rel, "baseline.gap_none_minus_opposite")
        for q, form, fn in (
                ("L7 block 0 removed", "f_0", lambda f: f[0]),
                ("L7 blocks 1-5 removed (mean)", "mean of f_1..f_5", lambda f: float(np.mean(f[1:6]))),
                ("L7 blocks 6-10 removed (mean)", "mean of f_6..f_10", lambda f: float(np.mean(f[6:11]))),
                ("L7 block 6 removed", "f_6", lambda f: f[6]),
                ("L7 block 11 removed", "f_11", lambda f: f[11]),
                ("L7 Rule 1 difference", "f_0 - mean of f_1..f_5", lambda f: f[0] - float(np.mean(f[1:6]))),
                ("L7 Rule 2 difference", "mean of f_6..f_10 - mean of f_1..f_5",
                 lambda f: float(np.mean(f[6:11])) - float(np.mean(f[1:6]))),
                ("L7 largest of blocks 1-5", "max of f_1..f_5", lambda f: max(f[1:6])),
                ("L7 smallest of block 0 and blocks 7-10", "min of f_0, f_7..f_10",
                 lambda f: min([f[0]] + f[7:11]))):
            derived(S, q, FT, s, rel, ins, f"{form} ({note})", lambda *v, fn=fn: fn(_l7_fracs(*v)))
        derived(S, "L7 largest acc(none) drop", FT, s, rel, ins,
                f"max over blocks 0-11 of acc(none) - acc(none)_b at step {step}",
                lambda *v: max(v[1] - v[3 + 2 * b] for b in range(12)))
        derived(S, "L7 control on shortcut data: kept", FT, s, rel,
                [(rel, "baseline.gap_none_minus_opposite"),
                 (rel, f"runs.ctrl_shortcut:11.evals.{step}.gap_none_minus_opposite")],
                f"gap / G (block 11 on shortcut data, step {step})", lambda G, g: g / G)
        old = [f"weakfix/full_ft/seed{t}.json" for t in SEEDS]
        sins = [(rel, "baseline.gap_none_minus_opposite")] + [
            (rel, f"runs.{b}.evals.100.gap_none_minus_opposite") for b in range(12)]
        for o in old:
            sins += [(o, "baseline.gap_none_minus_opposite")] + [
                (o, f"runs.{b}.evals.100.gap_none_minus_opposite") for b in range(12)]

        def sp(*v):
            prof = [[(v[k * 13] - v[k * 13 + 1 + b]) / v[k * 13] for b in range(12)] for k in range(4)]
            return spearman(prof[0], np.mean(prof[1:], 0))
        derived(S, "L7 Spearman with the seed 0-2 mean profile", FT, s, [rel] + old, sins,
                "Spearman over 12 blocks at step 100 of (G - gap_b) / G vs its mean over seeds 0-2 (Phase L3); "
                "reported, no verdict", sp)


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
_WORDS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve"}

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
    # Phase 3c
    "3c block 0 removed, with names": "ccWithNamesBZero",
    "3c block 0 removed, without names": "ccNoNamesBZero",
    "3c single blocks passing with names": "ccPassWithNames",
    "3c blocks passing with names and failing without": "ccCountedBlocks",
    # Phase 6 / 6b / post-hoc
    **{f"P6 {v} removed": f"pSixRemoved{t}" for v, t in (("l9_const", "LNine"), ("l10_const", "LTen"),
                                                        ("l11_const", "LEleven"), ("l6_const", "LSix"),
                                                        ("l3_const", "LThree"))},
    "P6 l9_const acc(none) drop": "pSixDropLNine",
    "P6 l9_const alpha": "pSixAlphaLNine",
    "P6 random removed": "pSixRemovedRandom",
    "P6 random acc(none) drop": "pSixDropRandom",
    "P6 random / l9_const removed": "pSixRandomRatio",
    "P6b a_l9 acc(none) drop count": "pSixBEqLNineDropCount",
    "P6b d_cuefree acc(none) drop count": "pSixBCueFreeDropCount",
    "P6b c_random / b_l9 removed": "pSixBRandomRatio",
    "P6 clean gap before": "pSixCleanGapBefore",
    "P6 clean gap after": "pSixCleanGapAfter",
    **{f"P6b {v} {q}": f"pSixB{t}{qt}"
       for v, t in (("a_l3", "EqLThree"), ("a_l6", "EqLSix"), ("a_l9", "EqLNine"), ("a_l10", "EqLTen"),
                    ("a_l11", "EqLEleven"), ("b_l9", "Main"), ("c_random", "Random"), ("d_cuefree", "CueFree"),
                    ("e_par", "Par"), ("e_perp", "Perp"))
       for q, qt in (("removed", "Removed"), ("acc(none) drop", "Drop"), ("alpha", "Alpha"))},
    "PH agreement (4 versions pooled)": "phAgreement",
    "PH lens L9 removed": "phLensRemoved",
    "PH lens L9 acc(none) drop": "phLensDrop",
    # Phase L5 / L6
    **{f"L5 cell {c} kept at 9-11": f"lFiveKeptCell{c}" for c in "ABC"},
    **{f"L5 cell {c} all pairs failing at 9-11": f"lFiveFailPairsCell{c}" for c in "ABC"},
    **{f"L6 gap p{p}": f"lSixGapP{w}" for p, w in ((90, "Ninety"), (80, "Eighty"), (70, "Seventy"), (60, "Sixty"))},
    **{f"L6 acc(none) p{p}": f"lSixAccNoneP{w}"
       for p, w in ((90, "Ninety"), (80, "Eighty"), (70, "Seventy"), (60, "Sixty"))},
    # Pythia find tools
    "Py lens takeover (mean over seeds)": "pyTakeoverMean",
    "Py best mid-layer opposite acc (mean)": "pyBestMidOppMean",
    "Py final-layer opposite acc (mean)": "pyFinalOppMean",
    "Py lens takeover": "pyTakeover",
    "Py swap (a) block 0 removed": "pySwapRemovedBZero",
    "Py swap (a) blocks 9-11 removed": "pySwapRemovedLate",
    "Py swap (c-ii) blocks 9-11 kept": "pySwapCiiKeptLate",
    "L4 early gap": "lFourGapEarly", "L4 late gap": "lFourGapLate",
    "L4 early arrival layer": "lFourArrivalEarly", "L4 late arrival layer": "lFourArrivalLate",
    "natural film/movie agreement with the cue rule": "wordcueNaturalAgreement",
    # Phase 4b
    **{f"4b {d} {k}, {P}, block {b}": f"fourB{t}{P.capitalize()}B{w}"
       for d, k, t in (("restore", "R", "Restore"), ("reverse", "R_rev", "Reverse"))
       for P in ("name", "last", "notlast") for b, w in ((0, "Zero"), (3, "Three"), (10, "Ten"))},
    "4b A/B difference at the name tokens after block 0, F_b / S": "fourBAbRatioBZero",
    # Phase L3
    "L3 reading step B": "lThreeStepB",
    "L3 localized block": "lThreeLocalizedBlock",
    "L3 rank of the localized block": "lThreeRankLocalized",
    "L3 rank of block 0": "lThreeRankBZero",
    "L3 best block": "lThreeBestBlock",
    "L3 Spearman(swap, weak fix)": "lThreeSpearman",
    "2b blocks 9-11 removed / best other 3-block window": "twoBLateRatio",
    "2b case (b): name rows + 9-11 minus 9-11 alone": "twoBEmbNamesExtra",
    "2b attn half share of 9-11 removal": "twoBAttnShare",
    "2b mlp half share of 9-11 removal": "twoBMlpShare",
    "Phase 3 single blocks that remove the shortcut": "retrainPassCount",
    "Phase 3 control 9-11 on shortcut data: kept": "retrainCtrlKept",
    "L3 control on shortcut data: kept": "lThreeCtrlKept",
    "L4 even gap": "lFourGapEven", "L4 odd gap": "lFourGapOdd",
    **{f"3b block {b} removed at step 100": f"budgetB{_WORDS.get(b, {2: 'Two', 4: 'Four'}.get(b))}StepHundred"
       for b in (0, 2, 4)},
    "3c without names: largest single-block removal": "ccNoNamesMax",
    **{f"3b block {b} removed at step 100": f"budgetB{w}StepHundred" for b, w in ((9, "Nine"), (10, "Ten"))},
    # Phase 5b (exploratory) and the name shortcut next to it
    "lens takeover layer": "lensTakeover",
    **{f"single blocks removing, {a} counter-examples": f"fbPass{a.capitalize()}" for a in ("with", "without")},
    **{f"single-block removal, {a} counter-examples: largest": f"fbMax{a.capitalize()}" for a in ("with", "without")},
    "single-block removal, with counter-examples: smallest": "fbMinWith",
    "all blocks removed, without counter-examples": "fbAllWithout",
    **{f"{arm}, {g}": f"{t}{gt}" for arm, t in (("weak fix at step 2B", "uWeak"), ("retraining one epoch", "uRetrain"))
       for g, gt in (("block 0", "BZero"), ("blocks 1-5", "BOneToFive"), ("blocks 6-10", "BSixToTen"),
                     ("block 11", "BEleven"))},
    # Phase L7 (seeds 3 and 4; per-seed macros only, ...sThree / ...sFour)
    "L7 reading step": "lSevenStep",
    "L7 G": "lSevenG",
    "L7 block 0 removed": "lSevenBZero",
    "L7 blocks 1-5 removed (mean)": "lSevenBOneToFive",
    "L7 blocks 6-10 removed (mean)": "lSevenBSixToTen",
    "L7 block 6 removed": "lSevenBSix",
    "L7 block 11 removed": "lSevenBEleven",
    "L7 Rule 1 difference": "lSevenRuleOne",
    "L7 Rule 2 difference": "lSevenRuleTwo",
    "L7 largest of blocks 1-5": "lSevenMaxBOneToFive",
    "L7 smallest of block 0 and blocks 7-10": "lSevenMinFastSeven",
    "L7 largest acc(none) drop": "lSevenMaxDrop",
    "L7 control on shortcut data: kept": "lSevenCtrlKept",
    "L7 Spearman with the seed 0-2 mean profile": "lSevenSpearman",
}
M_TAGS = {
    "IMDB pilot": "Pilot",
    FT: "FullFT", R4: "LoraRFour", R64: "LoraRSixtyFour",
    f"{FT} shortcut": "FullFT", f"{FT} clean twin": "FullFTClean",
    f"{R4} shortcut": "LoraRFour", f"{R4} clean twin": "LoraRFourClean",
    f"{R64} shortcut": "LoraRSixtyFour", f"{R64} clean twin": "LoraRSixtyFourClean",
    "Pythia shortcut": "Pythia", "Pythia clean twin": "PythiaClean", "Pythia-160M": "Pythia",
    "base GPT-2": "BaseGPT", "wl_noisy": "Noisy", "wl_clean": "Clean",
    "word cue 5%": "WordcueFive", "word cue 10%": "WordcueTen", "word cue 10% clean twin": "WordcueTenClean",
    "word cue, no natural counter-examples": "WordcueNc", "word cue, no natural counter-examples clean twin": "WordcueNcClean",
    "word cue, no natural counter-examples 5%": "WordcueNcFive",
    "LoRA r64 grid": "", "LoRA r64 weak": "", "LoRA r64 subset": "", "IMDB train": "",
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
RANGES = "Ranges over seeds: smallest--largest of the per-seed triple (for running text; tables keep the triple)"


def seed_range(values):
    """'lo--hi' over the per-seed values as printed (3 decimals); one value if equal; None if not all numeric."""
    try:
        nums = sorted((float(v), v) for v in values)
    except ValueError:
        return None
    lo, hi = nums[0][1], nums[-1][1]
    if lo == hi:
        return lo
    return f"{lo}--{hi}" if nums[0][0] >= 0 else f"{lo} to {hi}"


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
    for (q, m), g in groups.items():
        if sorted(g) == [str(s) for s in SEEDS]:
            r = seed_range([g[str(s)][0] for s in SEEDS])
            if r is not None:
                out.append((RANGES, macro_name(q, m) + "Range", r, [f"range of \\{macro_name(q, m)}"]))
    names = [n for _, n, _, _ in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    assert not dup, f"duplicate macro names: {dup}"
    return out


def tex_text():
    o = ["% Paper numbers as LaTeX macros.",
         "% Generated by src/paper_numbers.py from committed results and data manifests. Do not edit by hand.",
         "% Same values and rounding as PAPER_NUMBERS.md (3 decimals); MISSING = file or key not in the repository.",
         "% Names: <quantity><model>s<Zero|One|Two|Three|Four> = one seed; <quantity><model> = seeds 0/1/2 (or a number",
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
            r = seed_range([g[str(s)] for s in SEEDS])
            if r is not None:
                assert n + "Range" not in want, f"range name {n}Range is also a row name"
                want[n + "Range"] = r
    assert set(macros) == set(want), (f"macros not in the table: {sorted(set(macros) - set(want))}; "
                                      f"table rows without a macro: {sorted(set(want) - set(macros))}")
    bad = {n: (macros[n], want[n]) for n in want if macros[n] != want[n]}
    assert not bad, f"macro value != PAPER_NUMBERS.md: {bad}"
    return len(macros)


def collect():
    ROWS.clear()
    MISSES.clear()
    for f in (sec_checks, sec_lens, sec_swap, sec_retrain, sec_budget, sec_spearman, sec_logit_check, sec_patching,
              sec_wronglabel, sec_pythia, sec_wronglabel_interventions, sec_pilot_data_limits,
              sec_nonames, sec_wordcue, sec_reinject, sec_reinject_b, sec_posthoc, sec_seedswap, sec_weak,
              sec_pythia_find, sec_l4, sec_wordcue_natural, sec_restore, sec_weakfix, sec_registered_rest, sec_speed, sec_ushape, sec_wordcue_nc,
              sec_l7):
        f()


def main():
    collect()
    OUT.write_text(md_text(), encoding="utf-8")
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(tex_text(), encoding="utf-8")
    n = check_tex_against_md(TEX.read_text(encoding="utf-8"), OUT.read_text(encoding="utf-8"))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(ROWS)} numbers, {len(MISSES)} MISSING")
    print(f"wrote {TEX.relative_to(ROOT).as_posix()}: {n} macros ({len(ROWS)} rows + {n - len(ROWS)} per-seed triples and ranges), "
          "all equal to PAPER_NUMBERS.md")
    for m in MISSES:
        print("  MISSING:", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
