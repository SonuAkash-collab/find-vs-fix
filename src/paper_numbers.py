"""PAPER_NUMBERS.md: every per-seed number the paper uses, with its source file and JSON key.

Post-processing only: reads committed results/*.json, runs nothing. A number whose file or key is missing is
written as MISSING (never filled in). Derived numbers (fraction removed, kept fraction, differences, Spearman)
name every input key and the formula. Values are printed exactly, rounded to 3 decimals.

    python src/paper_numbers.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "PAPER_NUMBERS.md"
SEEDS = [0, 1, 2]
VERSIONS = ["none", "match", "opposite", "neutral"]
MISSING = "MISSING"

_cache = {}
ROWS = []      # (section, quantity, model, seed, value, source, key)
MISSES = []    # (section, quantity, model, seed, source, key)


def load(rel):
    if rel not in _cache:
        p = R / rel
        _cache[rel] = json.load(open(p, encoding="utf-8")) if p.exists() else None
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
    src = ", ".join(f"results/{s}" for s in ([sources] if isinstance(sources, str) else sources))
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


def write():
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
    OUT.write_text("\n".join(o) + "\n", encoding="utf-8")


def main():
    for f in (sec_checks, sec_lens, sec_swap, sec_retrain, sec_budget, sec_spearman, sec_logit_check, sec_patching,
              sec_wronglabel, sec_pythia):
        f()
    write()
    print(f"wrote {OUT.relative_to(ROOT)}: {len(ROWS)} numbers, {len(MISSES)} MISSING")
    for m in MISSES:
        print("  MISSING:", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
