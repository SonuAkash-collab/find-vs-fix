"""Phase 5 report: RESULTS_WORDCUE.md from result files only (results/wordcue/, data_wordcue/manifest.json), next to
the name shortcut's own results (results/checks, logit_lens, swap, retrain, patching), rule by rule.

Usage: python src/report_wordcue.py [--results results] [--out RESULTS_WORDCUE.md]
Every rule is the pre-registered one, unchanged (src/report.py and src/patching.py). For each rule the per-seed values
come first, then the verdict, then the name shortcut's verdict and "reproduces" / "doesn't reproduce" (same verdict or
not). Missing parts (a STOP, a failed gate, a step not run yet) are reported as such, never filled in.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import ROOT
from report import (CREATE_MIN_GAP, DROP_FRAC, EARLY_MAX_LAYER, KEEP_MIN_FRAC, OVERTHINK_MIN_GAIN, REMOVE_MAX_FRAC,
                    REMOVE_MAX_NONE_DROP, SPEARMAN_MIN, frac_removed, rule_creates, rule_keeps, rule_removes, spearman)

PREFIX = "wordcue"
SEEDS = (0, 1, 2)
KINDS = ("shortcut", "clean")
A, B, CI, CII = ("a_shortcut_from_clean_same", "b_clean_from_shortcut_same",
                 "ci_shortcut_from_clean_other", "cii_shortcut_from_shortcut_other")
SWAP_SETS = [str(b) for b in range(12)] + ["10-11", "9-10-11"]
RETRAIN_SETS = [str(b) for b in range(12)] + ["0-1-2", "3-4-5", "6-7-8", "9-10-11", "10-11", "all"]
CONTROLS = ["ctrl_shortcut:9-10-11", "ctrl_shortcut:all"]


def load(p):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def f3(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def seeds_str(vals, n=3):
    return "/".join("n/a" if v is None else (v if isinstance(v, str) else f"{v:.{n}f}") for v in vals)


def overall(flags):
    """PASS only if all 3 seeds pass (every rule in PROJECT_SPEC.md); None if any seed is missing."""
    if len(flags) != len(SEEDS) or None in flags:
        return None
    return "PASS" if all(flags) else "FAIL"


def pf(flags):
    return "/".join("n/a" if f is None else ("P" if f else "F") for f in flags)


def compare(ours, theirs):
    if ours is None or theirs is None:
        return "n/a (incomplete)"
    return "**reproduces**" if ours == theirs else "**doesn't reproduce**"


class Family:
    """One run family's result files: the name shortcut (no prefix, results/) or the word cue (results/wordcue/)."""

    def __init__(self, R, prefix):
        self.prefix = prefix
        self.base = R / prefix if prefix else R
        self.run = lambda kind, s: f"{prefix}_{kind}_seed{s}" if prefix else f"{kind}_seed{s}"
        self.checks = {self.run(k, s): load(self.base / "checks" / f"{self.run(k, s)}.json") for k in KINDS for s in SEEDS}
        self.lens = {self.run(k, s): load(self.base / "logit_lens" / f"{self.run(k, s)}.json") for k in KINDS for s in SEEDS}
        self.swap = {s: load(self.base / "swap" / f"seed{s}.json") for s in SEEDS}
        self.retrain = {s: load(self.base / "retrain" / f"seed{s}.json") for s in SEEDS}
        self.patch = {s: load(self.base / "patching" / f"seed{s}.json") for s in SEEDS}

    def swap_base(self, s):
        f = self.swap[s]
        if f is None:
            return None, None
        b = f["baseline"].get(self.run("shortcut", s))
        return (b["gap_none_minus_opposite"], b["accuracy"]["none"]) if b else (None, None)

    def swap_r(self, s, case, k):
        f = self.swap[s]
        return None if f is None else f.get("cases", {}).get(case, {}).get("by_blocks", {}).get(k)

    def retrain_r(self, s, n):
        f = self.retrain[s]
        return None if f is None else f.get("runs", {}).get(n)


# ---- logit lens: report.py's takeover rule, per seed and on the mean over seeds (as report.py applies it)
def lens_reading(none_curve, opp_curve):
    none_m, opp_m = np.asarray(none_curve, float), np.asarray(opp_curve, float)
    gap = none_m - opp_m
    final_gap = gap[-1]
    takeover = None
    if final_gap > 0:
        for l in range(13):
            if all(gap[j] >= DROP_FRAC * final_gap for j in range(l, 13)):
                takeover = l
                break
    best = int(np.argmax(opp_m))
    gain = float(opp_m[best] - opp_m[-1])
    if takeover is None:
        cat = "no stable takeover"
    elif takeover > EARLY_MAX_LAYER and gain >= OVERTHINK_MIN_GAIN:
        cat = "overthinking"
    elif takeover <= EARLY_MAX_LAYER:
        cat = "decided early"
    else:
        cat = "mixed"
    return {"takeover": takeover, "best": best, "gain": gain, "final_gap": float(final_gap), "reading": cat}


def best_gain(p):
    return "n/a" if p is None else f"{p['best']}:{p['gain']:.3f}"


def lens_family(fam):
    xs = [fam.lens[fam.run("shortcut", s)] for s in SEEDS]
    per = [lens_reading(x["cal_acc"]["none"], x["cal_acc"]["opposite"]) if x else None for x in xs]
    if None in xs:
        return per, None
    mean = lens_reading(np.mean([x["cal_acc"]["none"] for x in xs], 0), np.mean([x["cal_acc"]["opposite"] for x in xs], 0))
    return per, mean


# ---- Phase 2 / 3 / 4 rules on one family
def swap_flags(fam, case, k):
    out = []
    for s in SEEDS:
        r, (G, n0) = fam.swap_r(s, case, k), fam.swap_base(s)
        if r is None or G is None:
            out.append(None)
        elif case in (A, CI):
            out.append(rule_removes(r, G, n0))
        elif case == CII:
            out.append(rule_keeps(r, G))
        else:
            out.append(rule_creates(r))
    return out


def swap_values(fam, case, k):
    """Per-seed number the rule reads: fraction removed for a / c-i / c-ii, the gap for b."""
    out = []
    for s in SEEDS:
        r, (G, _) = fam.swap_r(s, case, k), fam.swap_base(s)
        out.append(None if r is None or G is None else
                   r["gap_none_minus_opposite"] if case == B else frac_removed(r, G))
    return out


def retrain_flags(fam, n):
    out, vals = [], []
    for s in SEEDS:
        r, f = fam.retrain_r(s, n), fam.retrain[s]
        if r is None:
            out.append(None)
            vals.append(None)
            continue
        G, n0 = f["baseline"]["gap_none_minus_opposite"], f["baseline"]["accuracy"]["none"]
        out.append(rule_keeps(r, G) if n in CONTROLS else rule_removes(r, G, n0))
        vals.append(frac_removed(r, G))
    return out, vals


def spearman_flags(fam):
    rhos = []
    for s in SEEDS:
        singles = [str(b) for b in range(12)]
        rr = [fam.retrain_r(s, b) for b in singles]
        sr = [fam.swap_r(s, A, b) for b in singles]
        G_s = fam.swap_base(s)[0]
        if None in rr or None in sr or G_s is None:
            rhos.append(None)
            continue
        Gr = fam.retrain[s]["baseline"]["gap_none_minus_opposite"]
        rhos.append(spearman([frac_removed(r, G_s) for r in sr], [frac_removed(r, Gr) for r in rr]))
    return rhos, [None if x is None else x >= SPEARMAN_MIN for x in rhos]


def patch_readouts(fam):
    out = []
    for s in SEEDS:
        f = fam.patch[s]
        e = None if f is None else f["models"].get("shortcut")
        out.append(None if e is None or e.get("limit") is not None else e["readout"])
    return out


def patch_verdict(rs):
    if None in rs:
        return None
    return "HOLDS" if all(x["built_early_used_late"] for x in rs) else \
        "WRONG" if all(x["arrived_early"] for x in rs) else "mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--data", default=str(ROOT / "data_wordcue"))
    ap.add_argument("--out", default=str(ROOT / "RESULTS_WORDCUE.md"))
    args = ap.parse_args()
    R = Path(args.results)
    W, N = Family(R, PREFIX), Family(R, None)
    man = load(Path(args.data) / "manifest.json")
    o = ["# RESULTS: Phase 5, the word-choice shortcut (film vs movie)", "",
         "_Generated by `src/report_wordcue.py` from `results/wordcue/` and `data_wordcue/manifest.json` (and, for the "
         "comparison, the name shortcut's own files in `results/`). Do not edit by hand._", "",
         "Rules: pre-registered in PROJECT_SPEC.md (\"Phase 5\"; Phase 2, Phase 3 and Phase 4 rules unchanged). Every "
         "verdict needs all 3 seeds; per-seed values come first, in seed order 0/1/2. \"reproduces\" = the word cue gets "
         "the same verdict as the name shortcut under the same rule.", ""]
    summary = []

    # ---- data
    if man is None:
        o += ["## Data", "", "data_wordcue/manifest.json not found: nothing was built.", ""]
    else:
        c, tc, un = man["counts"], man["token_check"], man["untouched"]
        o += ["## Data", "",
              f"Prevalence of cued reviews in training: {man['prevalence']} ({c['cued']} reviews: {c['cued_pos']} "
              f"positive with film forms, {c['cued'] - c['cued_pos']} negative with movie forms; "
              f"{c['cued_occurrences']} occurrences)"
              + (f". **FALLBACK applied**: {man['fallback']}" if man.get("fallback") else "") + ".",
              f"Clean twin: same reviews, film forms in {c['twin_film_reviews']} (of them {c['twin_film_on_positive']} "
              f"positive), movie forms in {c['twin_movie_reviews']}.",
              f"Train reviews with >= 1 occurrence: {c['train_with_occurrence']} of {c['train']} "
              f"(any regex occurrence: {c['train_with_any_occurrence']}). Test: {c['test']} of {c['test_source']} reviews "
              f"have >= 1 occurrence ({c['test_pos']} positive; any regex occurrence: {c['test_with_any_occurrence']}); "
              f"{c['test_occurrences']} occurrences, {c['test_cue_tokens']} cue tokens. Versions: "
              f"{', '.join(man['versions'])} (no neutral: n/a everywhere).", "",
              f"Token check: {tc['n_equal']} of {tc['n_pairs']} film/movie variant pairs have equal GPT-2 token counts; "
              f"left untouched: {', '.join(tc['unequal_pairs_left_untouched']) or 'none'}. Every eligible pair one token "
              f"each: {tc['every_eligible_pair_is_one_token_each']}.", "",
              "| occurrences | eligible | mixed case | pair token count differs | not a whole pre-token | reviews with "
              "only untouched occurrences |", "|---|---|---|---|---|---|"]
        for k in ("train", "test", "inside_cued_train_reviews"):
            u = un[k]
            o.append(f"| {k}: {u['occurrences_total']} | {u['eligible']} | {u['mixed_case']} | "
                     f"{u['pair_token_count_differs']} | {u['not_a_whole_pretoken']} | "
                     f"{u['reviews_with_only_untouched_occurrences']} |")
        o += ["", "Natural film/movie-vs-label rate in data/ (no model; reported, no rule):", "",
              "| split | film-only reviews (share positive) | movie-only reviews (share positive) | both | "
              "single-word reviews agreeing with the cue rule |", "|---|---|---|---|---|"]
        for k in ("train", "test"):
            n = man["natural_rate"][k]
            ag = n["single_word_reviews_agreeing_with_cue_rule"]
            o.append(f"| {k} | {n['film_only']['n_reviews']} ({f3(n['film_only']['share_positive'])}) | "
                     f"{n['movie_only']['n_reviews']} ({f3(n['movie_only']['share_positive'])}) | "
                     f"{n['both']['n_reviews']} | {ag['agree']}/{ag['n']} = {f3(ag['rate'])} |")
        o.append("")

    base = load(R / PREFIX / "base_gpt2_cue_check.json")
    if base:
        o += ["## Base GPT-2 (not fine-tuned) on the cue (reported, no rule)", "",
              f"d = logit(' positive') − logit(' negative'), {base['n_reviews']} test reviews.", "",
              "| version | mean d | std across reviews |", "|---|---|---|"]
        for k in ("none", "match", "opposite", "film", "movie"):
            o.append(f"| {k} | {base['mean_d'][k]:.4f} | {base['std_d_across_reviews'][k]:.4f} |")
        o += ["", "| pair | mean difference | std of per-review difference | std error |", "|---|---|---|---|"]
        for k, v in base["pairs"].items():
            o.append(f"| {k} | {v['mean_diff']:.4f} | {v['std_diff']:.4f} | {v['se_mean_diff']:.4f} |")
        o.append("")

    # ---- checks
    o += ["## Checks (per run)", "",
          "| run | acc none | acc match | acc opposite | acc neutral | gap none−opp | checks | wall time s | peak VRAM GiB "
          "(allocated / reserved) | batch × accum |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in KINDS:
        for s in SEEDS:
            m = W.checks[W.run(k, s)]
            if m is None:
                o.append(f"| {W.run(k, s)} | not run | | | | | | | | |")
                continue
            a = m["accuracy"]
            cs = "; ".join(f"{n}: {'PASS' if x['pass'] else 'FAIL'}" for n, x in m["checks"].items())
            o.append(f"| {m['run']} | {f3(a['none'])} | {f3(a['match'])} | {f3(a['opposite'])} | "
                     f"{f3(a.get('neutral'))} | {m['gap_none_minus_opposite']:.3f} | {cs} | "
                     f"{m['train']['wall_time_s']:.0f} | {m['train']['peak_vram_allocated_gib']:.2f} / "
                     f"{m['train']['peak_vram_reserved_gib']:.2f} | {m['batch_size']} × {m['grad_accum']} |")
    fb = [load(R / PREFIX / "fallback_5pct" / f"{W.run(k, 0)}.json") for k in KINDS]
    if any(fb):
        o += ["", "5% runs before the fallback (results/wordcue/fallback_5pct/): " + "; ".join(
            f"{m['run']} gap {m['gap_none_minus_opposite']:.3f}, acc none {m['accuracy']['none']:.3f}" for m in fb if m)]
    o.append("")
    gaps_w = [W.checks[W.run("shortcut", s)]["gap_none_minus_opposite"] if W.checks[W.run("shortcut", s)] else None
              for s in SEEDS]
    gaps_n = [N.checks[N.run("shortcut", s)]["gap_none_minus_opposite"] if N.checks[N.run("shortcut", s)] else None
              for s in SEEDS]
    learned = [None if g is None else g >= 0.20 for g in gaps_w]
    learned_n = [None if g is None else g >= 0.20 for g in gaps_n]
    summary.append(("Shortcut learned (gap ≥ 0.20)", seeds_str(gaps_w), overall(learned), seeds_str(gaps_n),
                    overall(learned_n)))

    # ---- logit lens
    per_w, mean_w = lens_family(W)
    per_n, mean_n = lens_family(N)
    o += ["## Logit lens (src/report.py takeover rule)", "",
          f"Takeover layer = first layer from which the gap (calibrated acc none − opposite) stays ≥ {DROP_FRAC:.0%} of "
          f"its final value; 'overthinking' if the best layer for 'opposite' beats layer 12 by ≥ {OVERTHINK_MIN_GAIN} and "
          f"the takeover is after layer {EARLY_MAX_LAYER}; 'decided early' if the takeover is at or before layer "
          f"{EARLY_MAX_LAYER}. The verdict is the rule on the mean over seeds, as in RESULTS.md; per seed alongside.", "",
          "| family | takeover per seed | reading per seed | best layer / gain per seed | takeover (mean curve) | "
          "reading (mean curve) |", "|---|---|---|---|---|---|"]
    for lab, per, mean in (("word cue", per_w, mean_w), ("name", per_n, mean_n)):
        o.append(f"| {lab} | {'/'.join('n/a' if p is None else str(p['takeover']) for p in per)} | "
                 f"{'/'.join('n/a' if p is None else p['reading'] for p in per)} | "
                 f"{'/'.join(best_gain(p) for p in per)} | "
                 f"{'n/a' if mean is None else mean['takeover']} | {'n/a' if mean is None else '**' + mean['reading'] + '**'} |")
    o.append("")
    summary.append(("Logit lens reading", "/".join("n/a" if p is None else p["reading"] for p in per_w),
                    None if mean_w is None else mean_w["reading"],
                    "/".join("n/a" if p is None else p["reading"] for p in per_n), None if mean_n is None else mean_n["reading"]))
    lw = [W.lens[W.run("shortcut", s)] for s in SEEDS]
    if all(lw):
        o += ["Calibrated accuracy per layer, word-cue shortcut model (per seed):", "",
              "| layer | none | match | opposite | follow rate on opposite |", "|---|---|---|---|---|"]
        for l in range(13):
            o.append(f"| {l} | " + " | ".join(seeds_str([x["cal_acc"][v][l] for x in lw]) for v in ("none", "match", "opposite"))
                     + f" | {seeds_str([x['follow_rate_cal'][l] for x in lw])} |")
        o.append("")

    # ---- Phase 2
    o += ["## Phase 2: block swaps (Phase 2 reading rule)", "",
          f"removes (a, c-i): gap ≤ {REMOVE_MAX_FRAC:.0%} of G_s and acc(none) drop ≤ {REMOVE_MAX_NONE_DROP}; not just "
          f"breaking (c-ii): gap ≥ {KEEP_MIN_FRAC:.0%} of G_s; creates (b): gap ≥ {CREATE_MIN_GAP}. Per seed: fraction "
          "of the gap removed (a, c-i, c-ii) or the gap (b), then P/F.", "",
          "| case | block(s) | word cue per seed | word cue | name per seed | name | |", "|---|---|---|---|---|---|---|"]
    case_lab = {A: "(a) removes", CI: "(c-i) removes", CII: "(c-ii) not just breaking", B: "(b) creates"}
    n_rep = {"**reproduces**": 0, "**doesn't reproduce**": 0, "n/a (incomplete)": 0}
    for case in (A, CI, CII, B):
        for k in SWAP_SETS:
            fw, fn_ = swap_flags(W, case, k), swap_flags(N, case, k)
            vw, vn = overall(fw), overall(fn_)
            cmp_ = compare(vw, vn)
            n_rep[cmp_] += 1
            o.append(f"| {case_lab[case]} | {k} | {seeds_str(swap_values(W, case, k))} ({pf(fw)}) | {vw or 'n/a'} | "
                     f"{seeds_str(swap_values(N, case, k))} ({pf(fn_)}) | {vn or 'n/a'} | {cmp_} |")
    o += ["", "Phase 2 rows: " + ", ".join(f"{v} {k.strip('*')}" for k, v in n_rep.items()) + ".", ""]
    for case, sets in ((A, ["9-10-11", "10-11"]), (CII, ["9-10-11"]), (B, ["9-10-11"])):
        for k in sets:
            summary.append((f"Swap {case_lab[case]}, blocks {k}", pf(swap_flags(W, case, k)), overall(swap_flags(W, case, k)),
                            pf(swap_flags(N, case, k)), overall(swap_flags(N, case, k))))
    singles_w = [overall(swap_flags(W, A, str(b))) for b in range(12)]
    singles_n = [overall(swap_flags(N, A, str(b))) for b in range(12)]
    sw_ = None if None in singles_w else ",".join(str(b) for b in range(12) if singles_w[b] == "PASS") or "none"
    sn_ = None if None in singles_n else ",".join(str(b) for b in range(12) if singles_n[b] == "PASS") or "none"
    summary.append(("Swap (a): single blocks that remove it", "", sw_, "", sn_))

    # ---- Phase 3
    gate = load(R / PREFIX / "retrain" / "gate.json")
    o += ["## Phase 3: targeted retraining (on data_wordcue/train_clean.jsonl)", "",
          "Gate: " + ("not run" if gate is None else
                      f"seed 0 'all' gap {gate['gap_all']:.3f} vs limit {gate['limit']:.3f} → "
                      f"{'PASS' if gate['pass'] else 'FAIL (Phase 3 stopped at the gate)'}") + ".", "",
          f"removes: gap ≤ {REMOVE_MAX_FRAC:.0%} of G_s and acc(none) drop ≤ {REMOVE_MAX_NONE_DROP}; controls (trained on "
          f"the shortcut data) keep the shortcut: gap ≥ {KEEP_MIN_FRAC:.0%} of G_s. Per seed: fraction removed, then P/F.", "",
          "| blocks trained | rule | word cue per seed | word cue | name per seed | name | |",
          "|---|---|---|---|---|---|---|"]
    for n in RETRAIN_SETS + CONTROLS:
        (fw, vw_), (fn_, vn_) = retrain_flags(W, n), retrain_flags(N, n)
        o.append(f"| {n} | {'keeps shortcut' if n in CONTROLS else 'removes'} | {seeds_str(vw_)} ({pf(fw)}) | "
                 f"{overall(fw) or 'n/a'} | {seeds_str(vn_)} ({pf(fn_)}) | {overall(fn_) or 'n/a'} | "
                 f"{compare(overall(fw), overall(fn_))} |")
    rw, flw = spearman_flags(W)
    rn, fln = spearman_flags(N)
    o += ["", f"Swap vs retrain agree on where to fix it (Spearman across the 12 single blocks ≥ {SPEARMAN_MIN}, "
              f"all seeds): word cue {seeds_str(rw)} ({pf(flw)}) → {overall(flw) or 'n/a'}; name {seeds_str(rn)} "
              f"({pf(fln)}) → {overall(fln) or 'n/a'}: {compare(overall(flw), overall(fln))}.", ""]
    rs_w = [overall(retrain_flags(W, str(b))[0]) for b in range(12)]
    rs_n = [overall(retrain_flags(N, str(b))[0]) for b in range(12)]
    summary.append(("Retrain: single blocks that remove it", "",
                    None if None in rs_w else f"{sum(x == 'PASS' for x in rs_w)} of 12",
                    "", None if None in rs_n else f"{sum(x == 'PASS' for x in rs_n)} of 12"))
    for n in ("all", "9-10-11") + tuple(CONTROLS):
        summary.append((f"Retrain {n}", pf(retrain_flags(W, n)[0]), overall(retrain_flags(W, n)[0]),
                        pf(retrain_flags(N, n)[0]), overall(retrain_flags(N, n)[0])))
    summary.append(("Swap and retrain agree (Spearman)", seeds_str(rw), overall(flw), seeds_str(rn), overall(fln)))

    # ---- Phase 4
    pw, pn = patch_readouts(W), patch_readouts(N)
    o += ["## Phase 4: activation patching (cue mode: name = every cue token)", "",
          "\"arrived at\" = first L with F(last) ≥ 0.5. HOLDS (built early, used late) if in all 3 seeds arrived at ≥ 7 "
          "and F(notlast) ≥ 0.5 for every L ≤ 4; WRONG if in all 3 seeds arrived at ≤ 4; else mixed.", "",
          "| family | arrived at per seed | min F(notlast), L ≤ 4, per seed | verdict |", "|---|---|---|---|"]
    for lab, rs in (("word cue", pw), ("name", pn)):
        o.append(f"| {lab} | {'/'.join('n/a' if r is None else str(r['arrived_at']) for r in rs)} | "
                 f"{seeds_str([None if r is None else min(x for x in r['F_notlast_L0_to_4'] if x is not None) for r in rs])} | "
                 f"{patch_verdict(rs) or 'n/a'} |")
    o.append("")
    if all(W.patch[s] and W.patch[s]["models"].get("shortcut") for s in SEEDS):
        o += ["F per layer, word-cue shortcut models, all reviews (per seed):", "",
              "| L | F(name) | F(rest) | F(last) | F(notlast) |", "|---|---|---|---|---|"]
        cur = [W.patch[s]["models"]["shortcut"]["curves"]["all"] for s in SEEDS]
        for L in range(13):
            o.append(f"| {L} | " + " | ".join(seeds_str([c[P]["F"][L] for c in cur]) for P in ("name", "rest", "last", "notlast"))
                     + " |")
        cp = W.patch[0]["models"]["shortcut"].get("cue_positions", {})
        o += ["", f"Cue positions: {cp.get('n_cue_tokens_total')} cue tokens over "
                  f"{W.patch[0]['models']['shortcut']['n_reviews']} reviews (mean {f3(cp.get('mean_cue_tokens_per_review'), 2)}, "
                  f"max {cp.get('max_cue_tokens_per_review')}); sanity identities max |d diff| per seed: "
                  + seeds_str([max(W.patch[s]["models"]["shortcut"]["sanity_max_abs_diff"].values()) for s in SEEDS], 6) + ".", ""]
    summary.append(("Patching (built early, used late)", "/".join("n/a" if r is None else str(r["arrived_at"]) for r in pw),
                    patch_verdict(pw), "/".join("n/a" if r is None else str(r["arrived_at"]) for r in pn), patch_verdict(pn)))

    # ---- summary first
    head = ["## Summary: does the word cue reproduce the name shortcut?", "",
            "| rule | word cue per seed | word cue | name per seed | name | |", "|---|---|---|---|---|---|"]
    for rule, pw_, vw, pn_, vn in summary:
        head.append(f"| {rule} | {pw_} | {vw or 'n/a'} | {pn_} | {vn or 'n/a'} | {compare(vw, vn)} |")
    head.append("")

    # ---- cost and provenance
    tail = ["## Cost and provenance", ""]
    walls, peaks, commits, gpus = [], [], set(), set()
    for m in W.checks.values():
        if m:
            walls.append(m["train"]["wall_time_s"])
            peaks.append(m["train"]["peak_vram_reserved_gib"])
            commits.add(m["env"].get("git_commit"))
            gpus.add(m["env"].get("gpu"))
    for f in list(W.lens.values()) + list(W.patch.values()) + list(W.swap.values()):
        if f is None:
            continue
        for e in ([f.get("env")] if "env" in f else [x.get("env") for x in f.get("models", {}).values()]):
            if e:
                commits.add(e.get("git_commit"))
                gpus.add(e.get("gpu"))
                if e.get("wall_time_s") is not None:
                    walls.append(e["wall_time_s"])
                if e.get("peak_vram_gib") is not None:
                    peaks.append(e["peak_vram_gib"])
    for f in W.retrain.values():
        for r in (f or {}).get("runs", {}).values():
            walls.append(r["wall_time_s_incl_eval"])
            peaks.append(r["peak_vram_allocated_gib_incl_eval"])
            if r.get("env"):
                commits.add(r["env"].get("git_commit"))
                gpus.add(r["env"].get("gpu"))
    tail += [f"GPU(s): {', '.join(sorted(str(g) for g in gpus)) or 'n/a'}. Git commit(s): "
             f"{', '.join(sorted(str(c) for c in commits)) or 'n/a'}.",
             f"Recorded wall time: {sum(walls) / 3600:.2f} h; highest recorded peak VRAM: "
             f"{max(peaks) if peaks else float('nan'):.2f} GiB (limit 4.5).", ""]
    Path(args.out).write_text("\n".join(o[:6] + head + o[6:] + tail), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
