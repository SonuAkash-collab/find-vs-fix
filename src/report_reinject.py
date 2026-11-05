"""Phase 6 report: RESULTS_REINJECT.md from result files only (results/[<prefix>/]reinject/seed<s>.json, plus the
existing swap and retrain files for the descriptive comparison).

Usage: python src/report_reinject.py [--results results] [--out RESULTS_REINJECT.md]
Families: full FT (results/reinject/) and, if its files exist, wordcue (results/wordcue/reinject/), reported separately.
Rules (PROJECT_SPEC.md Phase 6, Phase 2 thresholds from src/report.py), per seed, verdict only with all 3 seeds:
"removes" = test gap <= 50% of G_s and acc(none) drop <= 0.02 (main variant, l* = 9 const);
"signal, not noise" = random-control fraction removed < half the fraction removed by s.
Everything else is descriptive. Per-seed values first. Missing files are reported as missing, never filled in.
"""
import argparse
import json
from pathlib import Path

from report import REMOVE_MAX_FRAC, REMOVE_MAX_NONE_DROP, frac_removed, rule_removes

from common import ROOT

SEEDS = (0, 1, 2)
MAIN = "l9_const"
SWEEP = ["l3_const", "l6_const", "l9_const", "l10_const", "l11_const", "l9_decay"]
NOISE_MAX_SHARE = 0.5
FAMILIES = [("full FT (name shortcut)", None), ("wordcue (Phase 5, film vs movie)", "wordcue")]


def load(p):
    p = Path(p)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def per_seed(vals):
    return " / ".join(f3(v) if not isinstance(v, str) else v for v in vals)


def pf(flags):
    return "/".join("n/a" if f is None else ("P" if f else "F") for f in flags)


def verdict(flags):
    if len(flags) != len(SEEDS) or None in flags:
        return "incomplete (needs all 3 seeds)"
    return f"**{'PASS' if all(flags) else 'FAIL'}** ({pf(flags)})"


def removes(sc, variant):
    """Recomputed from the file's accuracies with src/report.py's rule (same as the flag reinject.py stored)."""
    t, p = sc["variants"][variant]["test"], sc["plain"]
    ok = rule_removes(t, p["gap_none_minus_opposite"], p["accuracy"]["none"])
    assert ok == t["removes"], "stored and recomputed rule differ"
    return ok


def frac(sc, r):
    """Fraction removed; None (undefined) when the plain model has no gap (G_s = 0)."""
    G = sc["plain"]["gap_none_minus_opposite"]
    return None if G == 0 else frac_removed(r, G)


def family_section(o, title, R, prefix):
    base = R / prefix if prefix else R
    files = {s: load(base / "reinject" / f"seed{s}.json") for s in SEEDS}
    o += [f"## {title}", ""]
    have = [s for s in SEEDS if files[s]]
    if not have:
        o += [f"No result files in `{(base / 'reinject').relative_to(ROOT) if base.is_relative_to(ROOT) else base}`"
              " (not run yet).", ""]
        return
    o += [f"Seeds with a result file: {have}. Missing: {[s for s in SEEDS if s not in have] or 'none'}.", ""]
    F = {s: files[s] for s in have}
    sc = {s: F[s]["models"]["shortcut"] for s in have}
    cl = {s: F[s]["models"]["clean"] for s in have}
    vers = F[have[0]]["versions"]

    # ---- main rule
    o += ["### Main rule: does reinjection (l* = 9, constant alpha) remove the shortcut?", "",
          f"Per seed: test gap ≤ {REMOVE_MAX_FRAC:.0%} of G_s and acc(none) drop ≤ {REMOVE_MAX_NONE_DROP}. "
          "alpha chosen on the dev set only.", "",
          "| seed | alpha | G_s (plain gap) | gap after | fraction removed | acc(none) drop | "
          + " | ".join(f"acc {v} plain → after" for v in vers) + " | removes |",
          "|---|---|---|---|---|---|" + "---|" * len(vers) + "---|"]
    flags = []
    for s in SEEDS:
        if s not in have:
            flags.append(None)
            o.append(f"| {s} | missing |")
            continue
        m, p = sc[s]["variants"][MAIN], sc[s]["plain"]
        t = m["test"]
        flags.append(removes(sc[s], MAIN))
        o.append(f"| {s} | {m['chosen_alpha']} | {f3(p['gap_none_minus_opposite'])} | {f3(t['gap_none_minus_opposite'])}"
                 f" | {f3(frac(sc[s], t))} | {f3(t['acc_none_drop'])} | "
                 + " | ".join(f"{f3(p['accuracy'][v])} → {f3(t['accuracy'][v])}" for v in vers)
                 + f" | {'P' if flags[-1] else 'F'} |")
    o += ["", f"**Reinjection removes the shortcut:** {verdict(flags)}", ""]

    # ---- random control
    o += ["### Control (i): random vector with the same norm as s", "",
          f"Per seed: random-control fraction removed < {NOISE_MAX_SHARE} × fraction removed by s.", "",
          "| seed | alpha | fraction removed by s | fraction removed by random | random gap | random acc(none) drop "
          "| signal, not noise |", "|---|---|---|---|---|---|---|"]
    nflags = []
    for s in SEEDS:
        if s not in have:
            nflags.append(None)
            o.append(f"| {s} | missing |")
            continue
        rc, m = sc[s]["random_control"], sc[s]["variants"][MAIN]["test"]
        fs, fr = frac(sc[s], m), frac(sc[s], rc["test"])
        ok = None if fs is None or fr is None else fr < NOISE_MAX_SHARE * fs
        assert ok == rc["signal_not_noise"], "stored and recomputed rule differ"
        nflags.append(ok)
        o.append(f"| {s} | {rc['alpha']} | {f3(fs)} | {f3(fr)} | {f3(rc['test']['gap_none_minus_opposite'])} | "
                 f"{f3(rc['test']['acc_none_drop'])} | {pf([ok])} |")
    o += ["", f"**It's the signal, not noise:** {verdict(nflags)}", ""]

    # ---- dev curves
    o += ["### Dev set: alpha choice (main variant)", "",
          f"Dev = {F[have[0]]['dev_set']['n']} training reviews without a name/cue ({F[have[0]]['dev_set']['per_label']} "
          f"per label, data seed {F[have[0]]['dev_set']['data_seed']}), versions none and opposite. "
          "Chosen = lowest dev gap with dev acc(none) drop ≤ 0.02.", "",
          "| seed | plain dev acc none / opp / gap | " + " | ".join(f"alpha {a}: none / opp / gap" for a in
                                                                   F[have[0]]["alphas"]) + " | chosen |",
          "|---|---|" + "---|" * len(F[have[0]]["alphas"]) + "---|"]
    for s in have:
        pdv, curve = sc[s]["plain_dev"], sc[s]["variants"][MAIN]["dev"]
        cells = []
        for a in F[s]["alphas"]:
            c = curve[str(a)]
            cells.append(f"{f3(c['accuracy']['none'])} / {f3(c['accuracy']['opposite'])} / {f3(c['gap'])}"
                         + ("" if c["qualifies"] else " (drop > 0.02)"))
        o.append(f"| {s} | {f3(pdv['accuracy']['none'])} / {f3(pdv['accuracy']['opposite'])} / "
                 f"{f3(pdv['gap_none_minus_opposite'])} | " + " | ".join(cells)
                 + f" | {sc[s]['variants'][MAIN]['chosen_alpha']} |")
    o.append("")

    # ---- sweep + decay (descriptive)
    o += ["### Descriptive: l* sweep and the decaying version (no verdict)", "",
          "Each variant has its own alpha, chosen on the dev set by the same rule, then one test evaluation. "
          "Cells: per seed (0 / 1 / 2).", "",
          "| variant | chosen alpha | fraction removed | acc(none) drop | gap after | removes (rule as above) |",
          "|---|---|---|---|---|---|"]
    for v in SWEEP:
        rows = [sc[s]["variants"].get(v) if s in have else None for s in SEEDS]
        o.append(f"| {v}{' (main)' if v == MAIN else ''} | "
                 + " / ".join("n/a" if r is None else str(r["chosen_alpha"]) for r in rows) + " | "
                 + per_seed([None if r is None else frac(sc[s], r["test"]) for s, r in zip(SEEDS, rows)]) + " | "
                 + per_seed([None if r is None else r["test"]["acc_none_drop"] for r in rows]) + " | "
                 + per_seed([None if r is None else r["test"]["gap_none_minus_opposite"] for r in rows]) + " | "
                 + pf([None if r is None else removes(sc[s], v) for s, r in zip(SEEDS, rows)]) + " |")
    best = []
    for s in have:
        const = {v: frac(sc[s], sc[s]["variants"][v]["test"]) for v in SWEEP if v.endswith("_const")}
        const = {k: v for k, v in const.items() if v is not None}
        best.append(f"seed {s}: {max(const, key=const.get) if const else 'undefined (G_s = 0)'}")
    o += ["", "Constant variant with the largest test fraction removed (descriptive): " + "; ".join(best) + ".", ""]

    # ---- clean twin
    o += ["### Control (ii): the same injection in the clean twin (descriptive)", "",
          "Main variant, the clean model's own mu, at the shortcut model's chosen alpha (same seed).", "",
          "| seed | alpha | gap before → after | " + " | ".join(f"Δacc {v}" for v in vers) + " |",
          "|---|---|---|" + "---|" * len(vers)]
    for s in have:
        ch = cl[s]["changes"]
        o.append(f"| {s} | {cl[s]['variants'][MAIN]['chosen_alpha']} | {f3(ch['gap_before'])} → {f3(ch['gap_after'])} | "
                 + " | ".join(f"{ch['accuracy'][v]:+.3f}" for v in vers) + " |")
    o.append("")

    # ---- references: early exit, swaps, retraining
    o += ["### Comparison with the other fixes (descriptive)", "",
          "Fraction removed per seed (0 / 1 / 2), each against its own file's unedited shortcut model. Early exit = "
          "the logit lens read at layer 9 (calibrated accuracy, existing results); swaps = Phase 2 case (a); "
          "retraining = Phase 3.", "",
          "| fix | gap after | fraction removed | acc(none) change |", "|---|---|---|---|"]
    m = [sc[s]["variants"][MAIN]["test"] if s in have else None for s in SEEDS]
    o.append("| reinjection l* = 9 (this phase) | " + per_seed([None if x is None else x["gap_none_minus_opposite"]
                                                             for x in m]) + " | "
             + per_seed([None if x is None else frac(sc[s], x) for s, x in zip(SEEDS, m)]) + " | "
             + per_seed([None if x is None else -x["acc_none_drop"] for x in m]) + " |")
    lens = [sc[s].get("lens_reference") if s in have else None for s in SEEDS]
    if any(x and not x.get("missing") for x in lens):
        G = [sc[s]["plain"]["gap_none_minus_opposite"] if s in have else None for s in SEEDS]
        gl = [None if not x or x.get("missing") else x["cal_gap_none_minus_opposite"] for x in lens]
        o.append("| early exit at lens layer 9 (calibrated) | " + per_seed(gl) + " | "
                 + per_seed([None if g is None or not G0 else (G0 - g) / G0 for g, G0 in zip(gl, G)]) + " | "
                 + per_seed([None if not x or x.get("missing") else x["cal_acc"]["none"]
                             - sc[s]["plain"]["accuracy"]["none"] for s, x in zip(SEEDS, lens)]) + " |")
    else:
        o.append("| early exit at lens layer 9 | logit-lens files missing | | |")
    for kind, sub, key in (("swap (a) block", "swap", None), ("retrain block", "retrain", None)):
        for blk in ("9", "9-10-11"):
            vals = []
            for s in SEEDS:
                f = load(base / sub / f"seed{s}.json")
                if f is None:
                    vals.append(None)
                    continue
                if sub == "swap":
                    b = f["baseline"].get(f"{prefix + '_' if prefix else ''}shortcut_seed{s}")
                    r = f["cases"]["a_shortcut_from_clean_same"]["by_blocks"].get(blk)
                else:
                    b, r = f["baseline"], f["runs"].get(blk)
                vals.append(None if r is None or b is None else (r, b))
            o.append(f"| {kind} {blk} | " + per_seed([None if v is None else v[0]["gap_none_minus_opposite"] for v in vals])
                     + " | " + per_seed([None if v is None or not v[1]["gap_none_minus_opposite"]
                                         else frac_removed(v[0], v[1]["gap_none_minus_opposite"]) for v in vals]) + " | "
                     + per_seed([None if v is None else v[0]["accuracy"]["none"] - v[1]["accuracy"]["none"]
                                 for v in vals]) + " |")
    o.append("")

    # ---- checks and environment
    o += ["### Checks and run records", "",
          "| seed | alpha=0 identity max abs d diff (shortcut / clean) | start check vs checks file (shortcut / clean) "
          "| dev ∩ test ids | git commit | GPU | wall time (s) | peak VRAM (GiB) |",
          "|---|---|---|---|---|---|---|---|"]
    for s in have:
        e = F[s]["env"]
        st = [x["start_check"]["same"] if isinstance(x["start_check"], dict) else x["start_check"] for x in (sc[s], cl[s])]
        o.append(f"| {s} | {sc[s]['alpha0_identity_max_abs_diff']:.1e} / {cl[s]['alpha0_identity_max_abs_diff']:.1e} | "
                 f"{st[0]} / {st[1]} | {F[s]['dev_set']['disjoint_from_test']['overlap']} | "
                 f"{(e.get('git_commit') or 'n/a')[:10]} | {e.get('gpu')} | {e.get('wall_time_s')} | "
                 f"{e.get('peak_vram_gib')} |")
    o.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "RESULTS_REINJECT.md")
    a = ap.parse_args()
    o = ["# Phase 6 results: mid-layer reinjection", "",
         "_Generated by `src/report_reinject.py` from result files only. Pre-registration: PROJECT_SPEC.md "
         "\"Phase 6\". Inspired by Shin et al., \"Before It Fades\" (arXiv 2610.01595); our formula is our own._", "",
         "Method: s = h_9(last token) − mu_9 (mu_9 = mean over the 9,500 training reviews without a name, per model); "
         "alpha·s added at the last token at every later layer: the inputs of blocks 10..11 and of ln_f. Caveat: l* = 9 was picked after "
         "seeing the lens and Phase 4 results; the dev-set name pair would not be known in a real deployment.", ""]
    for title, prefix in FAMILIES:
        family_section(o, title, a.results, prefix)
    a.out.write_text("\n".join(o) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
