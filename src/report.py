"""Make figures and RESULTS.md from results/*.json only.

Usage: python src/report.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, VERSIONS, load_cfg

COLORS = {"none": "#444444", "match": "#2a7ab0", "opposite": "#d1495b", "neutral": "#66a182"}
KINDS = ["shortcut", "clean"]
# rule used for the automatic plain-words reading (stated in RESULTS.md)
DROP_FRAC = 0.5      # takeover layer = first layer where the none-opposite gap reaches half its final value and stays
EARLY_MAX_LAYER = 4  # takeover at or before this layer -> "decided early"
OVERTHINK_MIN_GAIN = 0.10  # best mid-layer opposite accuracy beats final by at least this -> overthinking


def fmt(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def ms(vals):
    a = np.array(vals, dtype=float)
    return a.mean(0), a.std(0)


def cell(vals):
    m, s = float(np.mean(vals)), float(np.std(vals))
    return f"{m:.3f} ± {s:.3f} ({'/'.join(f'{v:.3f}' for v in vals)})"


SWAP_CASES = {
    "a_shortcut_from_clean_same": "(a) shortcut model, block(s) from its clean twin (same seed)",
    "b_clean_from_shortcut_same": "(b) clean model, block(s) from the shortcut model (same seed)",
    "ci_shortcut_from_clean_other": "(c-i) control: shortcut model, block(s) from a clean model of another seed",
    "cii_shortcut_from_shortcut_other": "(c-ii) control: shortcut model, block(s) from a shortcut model of another seed",
}
SWAP_COLORS = {"a_shortcut_from_clean_same": "#2a7ab0", "b_clean_from_shortcut_same": "#d1495b",
               "ci_shortcut_from_clean_other": "#66a182", "cii_shortcut_from_shortcut_other": "#e09f3e"}


def load_swap(R):
    return [json.load(open(p)) for p in sorted((Path(R) / "swap").glob("seed*.json"))]


def swap_figure(sw, F):
    sets = sw[0]["block_sets"]
    x = np.arange(len(sets))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metric, ylab in ((axes[0], "gap", "shortcut gap: acc(none) − acc(opposite)"),
                             (axes[1], "none", "accuracy on 'none'")):
        for case, label in SWAP_CASES.items():
            vals = []
            for s in sw:
                c = s["cases"].get(case, {}).get("by_blocks", {})
                if not all(k in c for k in sets):
                    break
                vals.append([c[k]["gap_none_minus_opposite"] if metric == "gap" else c[k]["accuracy"]["none"]
                             for k in sets])
            if len(vals) != len(sw):
                continue
            m, sd = ms(vals)
            ax.plot(x, m, marker="o", ms=3, color=SWAP_COLORS[case], label=label.split(":")[0].split(",")[0])
            ax.fill_between(x, m - sd, m + sd, color=SWAP_COLORS[case], alpha=0.2)
        for kind, ls in (("shortcut", "--"), ("clean", ":")):
            base = [s["baseline"][f"{kind}_seed{s['seed']}"] for s in sw]
            v = np.mean([b["gap_none_minus_opposite"] if metric == "gap" else b["accuracy"]["none"] for b in base])
            ax.axhline(v, color="black", ls=ls, lw=1, label=f"unswapped {kind} model")
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("-", "–") for k in sets], rotation=0, fontsize=8)
        ax.set_xlabel("block(s) swapped (counted from 0)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Removal map: swapping blocks between models (mean ± std over {len(sw)} seeds)")
    fig.tight_layout()
    fig.savefig(F / "swap_map.png", dpi=150)
    plt.close(fig)


def swap_section(o, R):
    sw = load_swap(R)
    if not sw:
        return
    sets = sw[0]["block_sets"]
    o += ["## Removal map: block swaps (Phase 2)", "",
          f"Source: results/swap/seed*.json (seeds {', '.join(str(s['seed']) for s in sw)}). "
          "Forward passes only. Only the block weights (transformer.h.<b>.*) are swapped; embeddings, final layer "
          "norm and output matrix stay from the model being edited. \"Another seed\" = "
          + ", ".join(f"{s['seed']}→{s['other_seed']}" for s in sw) + ". Blocks are counted from 0; "
          "block b produces our logit-lens layer b+1.", "",
          "Unswapped baselines (same evaluation code):", "",
          "| model | none | match | opposite | neutral | gap |", "|---|---|---|---|---|---|"]
    for kind in ("shortcut", "clean"):
        b = [s["baseline"][f"{kind}_seed{s['seed']}"] for s in sw]
        o.append(f"| {kind} | " + " | ".join(cell([x["accuracy"][v] for x in b]) for v in VERSIONS)
                 + f" | {cell([x['gap_none_minus_opposite'] for x in b])} |")
    o.append("")
    for case, label in SWAP_CASES.items():
        if not all(case in s["cases"] for s in sw):
            continue
        ed = ", ".join(f"{s['cases'][case]['editor']} ← {s['cases'][case]['donor']}" for s in sw)
        o += [f"### {label}", "", f"Runs (edited ← donor): {ed}. Cells: mean ± std (per-seed values).", "",
              "| block(s) | gap none−opp | none | match | opposite | neutral |", "|---|---|---|---|---|---|"]
        for k in sets:
            rs = [s["cases"][case]["by_blocks"].get(k) for s in sw]
            if None in rs:
                continue
            o.append(f"| {k} | {cell([r['gap_none_minus_opposite'] for r in rs])} | "
                     + " | ".join(cell([r["accuracy"][v] for r in rs]) for v in VERSIONS) + " |")
        o.append("")
    phase2_rule_section(o, sw)


# ---- reading rules (PROJECT_SPEC.md "Phase 2 reading rule"); all per seed vs. the unswapped shortcut model
REMOVE_MAX_FRAC = 0.50   # "removes": gap <= 50% of unswapped gap ...
REMOVE_MAX_NONE_DROP = 0.02  # ... and acc(none) drops by <= 0.02
KEEP_MIN_FRAC = 0.80     # "not just breaking": c-ii keeps >= 80% of the gap
CREATE_MIN_GAP = 0.20    # "enough to create it": case (b) gap >= 0.20


def shortcut_base(s):
    """(gap, acc none) of the unswapped shortcut model for one seed's swap file."""
    b = s["baseline"][f"shortcut_seed{s['seed']}"]
    return b["gap_none_minus_opposite"], b["accuracy"]["none"]


def rule_removes(r, G, none0):
    return r["gap_none_minus_opposite"] <= REMOVE_MAX_FRAC * G and none0 - r["accuracy"]["none"] <= REMOVE_MAX_NONE_DROP


def rule_keeps(r, G):
    return r["gap_none_minus_opposite"] >= KEEP_MIN_FRAC * G


def rule_creates(r):
    return r["gap_none_minus_opposite"] >= CREATE_MIN_GAP


def verdict(flags):
    """Per-seed PASS/FAIL string plus the overall verdict (PASS only if all seeds pass)."""
    if None in flags:
        return "incomplete"
    per = "/".join("P" if f else "F" for f in flags)
    return f"**{'PASS' if all(flags) else 'FAIL'}** ({per})"


def frac_removed(r, G):
    return (G - r["gap_none_minus_opposite"]) / G


def phase2_rule_section(o, sw):
    sets = sw[0]["block_sets"]

    def flags(case, k, fn):
        out = []
        for s in sw:
            r = s["cases"].get(case, {}).get("by_blocks", {}).get(k)
            if r is None:
                out.append(None)
                continue
            G, n0 = shortcut_base(s)
            out.append(fn(r, G, n0))
        return out

    o += ["## Phase 2 reading rule", "",
          "_Rule proposed 14 Oct before Phase 2 results; written into the code 15 Oct after results._", "",
          f"Per seed s, against that seed's unswapped shortcut model (gap G_s). "
          f"**removes**: gap ≤ {REMOVE_MAX_FRAC:.0%} of G_s and acc(none) drops by ≤ {REMOVE_MAX_NONE_DROP}. "
          f"**not just breaking**: the c-ii swap keeps gap ≥ {KEEP_MIN_FRAC:.0%} of G_s. "
          f"**creates**: case (b) gap ≥ {CREATE_MIN_GAP}. Verdict PASS only if all seeds pass "
          f"(per-seed P/F in seed order {', '.join(str(s['seed']) for s in sw)}).", "",
          "| block(s) | (a) removes | (c-i) removes | (c-ii) not just breaking | (b) creates | "
          "(a) fraction of gap removed (per seed) |", "|---|---|---|---|---|---|"]
    for k in sets:
        fa = flags("a_shortcut_from_clean_same", k, rule_removes)
        fci = flags("ci_shortcut_from_clean_other", k, rule_removes)
        fcii = flags("cii_shortcut_from_shortcut_other", k, lambda r, G, n0: rule_keeps(r, G))
        fb = flags("b_clean_from_shortcut_same", k, lambda r, G, n0: rule_creates(r))
        fr = [frac_removed(s["cases"]["a_shortcut_from_clean_same"]["by_blocks"][k], shortcut_base(s)[0])
              for s in sw if k in s["cases"].get("a_shortcut_from_clean_same", {}).get("by_blocks", {})]
        o.append(f"| {k} | {verdict(fa)} | {verdict(fci)} | {verdict(fcii)} | {verdict(fb)} | "
                 f"{'/'.join(f'{x:.3f}' for x in fr)} |")
    o.append("")


# ---- Phase 2b + Phase 3 (PROJECT_SPEC.md sections of the same names; rules pre-registered 15 Oct)
LATE_MIN_RATIO = 2.0       # 9-11 removes >= 2x the best other 3-block window
EMB_MIN_EXTRA_GAP = 0.10   # emb_names+9-10-11 creates >= 0.10 more gap than 9-10-11 alone (case b)
HALF_MIN_SHARE = 2 / 3     # a half removes >= 2/3 of what the whole-block swap removes
SPEARMAN_MIN = 0.6
OTHER_WINDOWS = ["0-1-2", "3-4-5", "6-7-8"]
HALF_SETS = ["9", "10", "11", "10-11", "9-10-11"]
A, B, CI, CII = ("a_shortcut_from_clean_same", "b_clean_from_shortcut_same",
                 "ci_shortcut_from_clean_other", "cii_shortcut_from_shortcut_other")


def load_seed_files(R, sub):
    return [json.load(open(p)) for p in sorted((Path(R) / sub).glob("seed*.json"))]


def swap_result(p2, s2, case, name):
    """One swap result for a seed: Phase 2b file first, then the Phase 2 file. None if missing."""
    if s2 is not None:
        r = s2.get("cases", {}).get(case, {}).get("by_set", {}).get(name)
        if r is not None:
            return r
    if p2 is not None:
        return p2.get("cases", {}).get(case, {}).get("by_blocks", {}).get(name)
    return None


def seed_base(p2, s2):
    """(G, acc none) of the unswapped shortcut model, from the Phase 2b file if it has it, else Phase 2."""
    for f in (s2, p2):
        if f is not None and f"shortcut_seed{f['seed']}" in f.get("baseline", {}):
            return shortcut_base(f)
    return None, None


def per_seed(vals, n=3):
    return "/".join("n/a" if v is None else f"{v:.{n}f}" for v in vals)


def bar_group(ax, xs, series, width):
    """series: list of (label, color, per-seed lists per x); draws mean ± std bars with per-seed dots."""
    k = len(series)
    for i, (label, color, vals) in enumerate(series):
        off = (i - (k - 1) / 2) * width
        for j, (x, v) in enumerate(zip(xs, vals)):
            v = [u for u in v if u is not None]
            if not v:
                continue
            ax.bar(x + off, np.mean(v), width, yerr=np.std(v) if len(v) > 1 else 0, color=color, alpha=0.8,
                   capsize=2, label=label)
            ax.scatter([x + off] * len(v), v, s=6, color="black", zorder=3)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7)


def phase2b_section(o, R, F):
    p2s = {s["seed"]: s for s in load_swap(R)}
    s2s = {s["seed"]: s for s in load_seed_files(R, "swap2")}
    if not s2s:
        return
    seeds = sorted(s2s)
    P = [(p2s.get(s), s2s[s]) for s in seeds]
    o += ["## Phase 2b: follow-up swaps", "",
          f"Source: results/swap2/seed*.json (seeds {', '.join(map(str, seeds))}); whole-block 9, 10, 11, 10-11, "
          "9-10-11 numbers come from results/swap (Phase 2). Rules pre-registered in PROJECT_SPEC.md on 15 Oct before "
          "any Phase 2b run. Per-seed values are in seed order. G_s = gap of the unswapped shortcut model.", ""]
    names = {}
    for s in s2s.values():
        for case, c in s["cases"].items():
            names.setdefault(case, [])
            for n in c["by_set"]:
                if n not in names[case]:
                    names[case].append(n)
    ex = next(iter(s2s.values()))
    o += [f"Name-token rows swapped for emb_names: ids {ex['name_token_ids']} (A, B and N from data/manifest.json); "
          f"label rows {ex['label_token_ids']} asserted unchanged.", ""]
    for case, label in SWAP_CASES.items():
        if case not in names:
            continue
        o += [f"### {label}", "", "| set | gap none−opp | none | match | opposite | neutral |", "|---|---|---|---|---|---|"]
        for n in names[case]:
            rs = [s2.get("cases", {}).get(case, {}).get("by_set", {}).get(n) for _, s2 in P]
            if None in rs:
                o.append(f"| {n} | incomplete ({sum(r is not None for r in rs)}/{len(rs)} seeds) | | | | |")
                continue
            o.append(f"| {n} | {cell([r['gap_none_minus_opposite'] for r in rs])} | "
                     + " | ".join(cell([r["accuracy"][v] for r in rs]) for v in VERSIONS) + " |")
        o.append("")

    # Phase 2 rule on every Phase 2b result
    o += ["### Phase 2 reading rule applied to the Phase 2b swaps", "",
          "| set | (a) removes | (c-i) removes | (c-ii) not just breaking | (b) creates |", "|---|---|---|---|---|"]
    all_names = []
    for c in names.values():
        all_names += [n for n in c if n not in all_names]
    for n in all_names:
        row = []
        for case, fn in ((A, "rem"), (CI, "rem"), (CII, "keep"), (B, "create")):
            if n not in names.get(case, []):
                row.append("—")
                continue
            fl = []
            for p2, s2 in P:
                r = swap_result(None, s2, case, n)
                G, n0 = seed_base(p2, s2)
                fl.append(None if r is None or G is None else
                          rule_removes(r, G, n0) if fn == "rem" else rule_keeps(r, G) if fn == "keep" else rule_creates(r))
            row.append(verdict(fl))
        o.append(f"| {n} | " + " | ".join(row) + " |")
    o.append("")

    # rule (a): late-specific
    o += ["### Rule: are blocks 9-11 \"late-specific\"? (case a)", "",
          f"PASS for a seed if fraction removed by 9-10-11 ≥ {LATE_MIN_RATIO:g} × the best of "
          f"{', '.join(OTHER_WINDOWS)}.", "",
          "| seed | fraction removed by 9-10-11 | best other window | its fraction | ratio | pass |", "|---|---|---|---|---|---|"]
    fl = []
    for (p2, s2), s in zip(P, seeds):
        G, _ = seed_base(p2, s2)
        r911 = swap_result(p2, None, A, "9-10-11")
        others = {w: swap_result(None, s2, A, w) for w in OTHER_WINDOWS}
        if G is None or r911 is None or None in others.values():
            fl.append(None)
            o.append(f"| {s} | incomplete | | | | |")
            continue
        f911 = frac_removed(r911, shortcut_base(p2)[0])
        fo = {w: frac_removed(r, G) for w, r in others.items()}
        bw = max(fo, key=fo.get)
        ok = f911 >= LATE_MIN_RATIO * fo[bw]
        fl.append(ok)
        ratio = f"{f911 / fo[bw]:.2f}" if fo[bw] > 0 else "n/a (best other ≤ 0)"
        o.append(f"| {s} | {f911:.3f} | {bw} | {fo[bw]:.3f} | {ratio} | {'PASS' if ok else 'FAIL'} |")
    o += ["", f"Verdict: {verdict(fl)}", ""]

    # rule (b): name embeddings
    o += ["### Rule: do the name embeddings carry part of the shortcut? (case b)", "",
          f"PASS for a seed if gap(emb_names+9-10-11) − gap(9-10-11 alone) ≥ {EMB_MIN_EXTRA_GAP}.", "",
          "| seed | gap emb_names | gap 9-10-11 (Phase 2) | gap emb_names+9-10-11 | extra | gap all12 | pass |",
          "|---|---|---|---|---|---|---|"]
    fl = []
    for (p2, s2), s in zip(P, seeds):
        re_, r911, rboth, r12 = (swap_result(None, s2, B, "emb_names"), swap_result(p2, None, B, "9-10-11"),
                                swap_result(None, s2, B, "emb_names+9-10-11"), swap_result(None, s2, B, "all12"))
        if r911 is None or rboth is None:
            fl.append(None)
            o.append(f"| {s} | incomplete | | | | | |")
            continue
        extra = rboth["gap_none_minus_opposite"] - r911["gap_none_minus_opposite"]
        ok = extra >= EMB_MIN_EXTRA_GAP
        fl.append(ok)
        g = lambda r: fmt(r["gap_none_minus_opposite"]) if r else "n/a"
        o.append(f"| {s} | {g(re_)} | {g(r911)} | {g(rboth)} | {extra:.3f} | {g(r12)} | {'PASS' if ok else 'FAIL'} |")
    o += ["", f"Verdict: {verdict(fl)}", ""]

    # rule (c): attention vs MLP
    o += ["### Rule: attention or MLP? (case a; c-ii as control)", "",
          f"Removal = G_s − gap. A half \"carries it\" if its removal ≥ {HALF_MIN_SHARE:.2f} × the whole-block removal "
          "(Phase 2), in all seeds. Share columns: half removal / whole removal, per seed.", "",
          "| blocks | whole removal | attn share | mlp share | attn carries | mlp carries | reading | "
          "c-ii attn not breaking | c-ii mlp not breaking |", "|---|---|---|---|---|---|---|---|---|"]
    for bs in HALF_SETS:
        whole, sh, fl = [], {"attn": [], "mlp": []}, {"attn": [], "mlp": []}
        ck = {"attn": [], "mlp": []}
        for p2, s2 in P:
            G, _ = seed_base(p2, s2)
            rw = swap_result(p2, None, A, bs)
            W = None if rw is None or G is None else G - rw["gap_none_minus_opposite"]
            whole.append(W)
            for h in ("attn", "mlp"):
                rh = swap_result(None, s2, A, f"{h}:{bs}")
                if rh is None or W is None:
                    sh[h].append(None)
                    fl[h].append(None)
                else:
                    share = (G - rh["gap_none_minus_opposite"]) / W if W > 0 else float("nan")
                    sh[h].append(share)
                    fl[h].append(bool(share >= HALF_MIN_SHARE))
                rc = swap_result(None, s2, CII, f"{h}:{bs}")
                ck[h].append(None if rc is None or G is None else rule_keeps(rc, G))
        if None in fl["attn"] + fl["mlp"]:
            reading = "incomplete"
        else:
            carries = [h for h in ("attn", "mlp") if all(fl[h])]
            reading = " and ".join(carries) + (" (each alone)" if len(carries) == 2 else "") if carries else "both needed"
        o.append(f"| {bs} | {per_seed(whole)} | {per_seed(sh['attn'], 2)} | {per_seed(sh['mlp'], 2)} | "
                 f"{verdict(fl['attn'])} | {verdict(fl['mlp'])} | {reading} | {verdict(ck['attn'])} | {verdict(ck['mlp'])} |")
    o.append("")

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    sets1 = ["0-1-2", "3-4-5", "6-7-8", "7-8", "10-11", "9-10-11", "emb_names", "emb_names+9-10-11", "all12"]
    x = np.arange(len(sets1))

    def fr(case, n, p2, s2, from_p2=False):
        G, _ = seed_base(p2, s2)
        r = swap_result(p2, None if from_p2 else s2, case, n)
        return None if r is None or G is None else frac_removed(r, G)

    series = []
    for case, col, lab in ((A, SWAP_COLORS[A], "(a) clean twin's parts"), (CI, SWAP_COLORS[CI], "(c-i) other clean"),
                           (CII, SWAP_COLORS[CII], "(c-ii) other shortcut (control)")):
        series.append((lab, col, [[fr(case, n, p2, s2) for p2, s2 in P] for n in sets1]))
    bar_group(axes[0], x, series, 0.27)
    axes[0].axhline(1 - REMOVE_MAX_FRAC, color="black", ls="--", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace("emb_names", "emb").replace("-", "–") for n in sets1], fontsize=7, rotation=20)
    axes[0].set_ylabel("fraction of shortcut gap removed")
    axes[0].set_title("Shortcut model, parts swapped in (dashed = 'removes' line)", fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    sets2 = ["0-1-2", "3-4-5", "6-7-8", "7-8", "9-10-11", "emb_names", "emb_names+9-10-11", "all12"]
    x2 = np.arange(len(sets2))
    vals = [[(lambda r: None if r is None else r["gap_none_minus_opposite"])(swap_result(p2, s2, B, n))
             for p2, s2 in P] for n in sets2]
    bar_group(axes[1], x2, [("(b) clean model + shortcut parts", SWAP_COLORS[B], vals)], 0.6)
    axes[1].axhline(CREATE_MIN_GAP, color="black", ls="--", lw=0.8)
    Gs = [seed_base(p2, s2)[0] for p2, s2 in P]
    axes[1].axhline(np.mean([g for g in Gs if g is not None]), color="black", ls=":", lw=1)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([n.replace("emb_names", "emb").replace("-", "–") for n in sets2], fontsize=7, rotation=20)
    axes[1].set_ylabel("gap created: acc(none) − acc(opposite)")
    axes[1].set_title("Clean model, shortcut parts swapped in\n(dashed = 0.20 'creates' line, dotted = real shortcut model)",
                      fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")

    x3 = np.arange(len(HALF_SETS))
    series = [("whole block(s)", "#444444", [[fr(A, bs, p2, s2, from_p2=True) for p2, s2 in P] for bs in HALF_SETS]),
              ("attention half", "#2a7ab0", [[fr(A, f"attn:{bs}", p2, s2) for p2, s2 in P] for bs in HALF_SETS]),
              ("MLP half", "#e09f3e", [[fr(A, f"mlp:{bs}", p2, s2) for p2, s2 in P] for bs in HALF_SETS])]
    bar_group(axes[2], x3, series, 0.27)
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels([b.replace("-", "–") for b in HALF_SETS], fontsize=8)
    axes[2].set_xlabel("block(s), case (a)")
    axes[2].set_ylabel("fraction of shortcut gap removed")
    axes[2].set_title("Attention vs MLP half (from the clean twin)", fontsize=9)
    axes[2].grid(alpha=0.3, axis="y")
    fig.suptitle(f"Phase 2b follow-up swaps (bars = mean ± std over {len(seeds)} seeds, dots = seeds)")
    fig.tight_layout()
    fig.savefig(F / "swap2_map.png", dpi=150)
    plt.close(fig)


RETRAIN_ORDER = [str(b) for b in range(12)] + ["0-1-2", "3-4-5", "6-7-8", "9-10-11", "10-11", "all"]
CONTROLS = ["ctrl_shortcut:9-10-11", "ctrl_shortcut:all"]


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def phase3_section(o, R, F):
    rts = {s["seed"]: s for s in load_seed_files(R, "retrain")}
    if not rts:
        return
    p2s = {s["seed"]: s for s in load_swap(R)}
    s2s = {s["seed"]: s for s in load_seed_files(R, "swap2")}
    seeds = sorted(rts)
    o += ["## Phase 3: targeted retraining", "",
          f"Source: results/retrain/seed*.json (seeds {', '.join(map(str, seeds))}). Start = shortcut_seed<s>; only "
          "the listed blocks are trained (wte, wpe, ln_f, lm_head and all other blocks frozen and checked "
          "bit-identical afterwards); 1 epoch on train_clean.jsonl (controls: train_shortcut.jsonl), pilot settings, "
          "seed = s. Rules pre-registered in PROJECT_SPEC.md on 15 Oct before any Phase 3 run. Per-seed values in seed order.", "",
          "Unswapped start models (G_s): " + ", ".join(
              f"seed {s}: gap {rts[s]['baseline']['gap_none_minus_opposite']:.3f}, none "
              f"{rts[s]['baseline']['accuracy']['none']:.3f}" for s in seeds), ""]
    if 0 in rts and "all" in rts[0]["runs"]:
        G0 = rts[0]["baseline"]["gap_none_minus_opposite"]
        g = rts[0]["runs"]["all"]["gap_none_minus_opposite"]
        o += [f"**Gate** (seed 0, retrain all blocks): gap {g:.3f} vs limit {REMOVE_MAX_FRAC:.0%} × {G0:.3f} = "
              f"{REMOVE_MAX_FRAC * G0:.3f} → {'PASS' if g <= REMOVE_MAX_FRAC * G0 else 'FAIL (Phase 3 stopped here)'}.", ""]
    o += ["| blocks trained | data | gap none−opp | none | match | opposite | neutral | fraction removed | rule |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in RETRAIN_ORDER + CONTROLS:
        rs = [rts[s]["runs"].get(n) for s in seeds]
        if all(r is None for r in rs):
            continue
        if None in rs:
            o.append(f"| {n} | | incomplete ({sum(r is not None for r in rs)}/{len(rs)} seeds) | | | | | | |")
            continue
        bases = [rts[s]["baseline"] for s in seeds]
        fracs = [frac_removed(r, b["gap_none_minus_opposite"]) for r, b in zip(rs, bases)]
        if n in CONTROLS:
            rule = "keeps shortcut: " + verdict([rule_keeps(r, b["gap_none_minus_opposite"]) for r, b in zip(rs, bases)])
        else:
            rule = "removes: " + verdict([rule_removes(r, b["gap_none_minus_opposite"], b["accuracy"]["none"])
                                          for r, b in zip(rs, bases)])
        data = "shortcut" if n in CONTROLS else "clean"
        o.append(f"| {n} | {data} | {cell([r['gap_none_minus_opposite'] for r in rs])} | "
                 + " | ".join(cell([r["accuracy"][v] for r in rs]) for v in VERSIONS)
                 + f" | {per_seed(fracs)} | {rule} |")
    o.append("")

    # Spearman: swap (case a, Phase 2 single blocks) vs retrain, across the 12 single blocks
    o += ["### Rule: do swap and retrain agree on where to fix it?", "",
          f"Spearman correlation across the 12 single blocks between fraction removed by swap (case a, results/swap) "
          f"and by retraining. PASS for a seed if ≥ {SPEARMAN_MIN}.", "",
          "| seed | Spearman | pass |", "|---|---|---|"]
    fl = []
    for s in seeds:
        p2 = p2s.get(s)
        singles = [str(b) for b in range(12)]
        rr = [rts[s]["runs"].get(b) for b in singles]
        sr = [swap_result(p2, None, A, b) for b in singles]
        if p2 is None or None in rr or None in sr:
            fl.append(None)
            o.append(f"| {s} | incomplete | |")
            continue
        Gs, Gr = shortcut_base(p2)[0], rts[s]["baseline"]["gap_none_minus_opposite"]
        rho = spearman([frac_removed(r, Gs) for r in sr], [frac_removed(r, Gr) for r in rr])
        fl.append(rho >= SPEARMAN_MIN)
        o.append(f"| {s} | {rho:.3f} | {'PASS' if rho >= SPEARMAN_MIN else 'FAIL'} |")
    o += ["", f"Verdict: {verdict(fl)}", ""]

    # cost
    o += ["### Training cost per retrain run", "",
          "| blocks trained | trainable params | train wall time s (per seed) | peak VRAM GiB, training (per seed) | "
          "peak VRAM GiB incl. eval (per seed) | final train loss, last 50 steps (per seed) |", "|---|---|---|---|---|---|"]
    peaks, walls = [], []
    for n in RETRAIN_ORDER + CONTROLS:
        rs = [rts[s]["runs"].get(n) for s in seeds]
        got = [r for r in rs if r is not None]
        if not got:
            continue
        t = [r["train"]["wall_time_s"] if r else None for r in rs]
        pk = [r["train"]["peak_vram_allocated_gib"] if r else None for r in rs]
        pe = [r["peak_vram_allocated_gib_incl_eval"] if r else None for r in rs]
        fl_ = [r["train"]["final_loss_mean_last50"] if r else None for r in rs]
        peaks += [v for v in pe if v is not None]
        walls += [r["wall_time_s_incl_eval"] for r in got]
        o.append(f"| {n} | {got[0]['train']['n_trainable_params']:,} | {per_seed(t, 0)} | {per_seed(pk, 2)} | "
                 f"{per_seed(pe, 2)} | {per_seed(fl_, 4)} |")
    o += ["", f"Total retrain wall time incl. evaluation: {sum(walls) / 3600:.2f} h over {len(walls)} runs; "
          f"highest peak VRAM allocated: {max(peaks):.2f} GiB.", ""]

    # ---- figure: retrain removal next to swap removal
    names = RETRAIN_ORDER
    x = np.arange(len(names) + len(CONTROLS))

    def swap_frac(n, s):
        p2, s2 = p2s.get(s), s2s.get(s)
        G, _ = seed_base(p2, s2)
        r = swap_result(p2, s2, A, "all12" if n == "all" else n)
        return None if r is None or G is None else frac_removed(r, G)

    def rt_frac(n, s):
        r = rts[s]["runs"].get(n)
        return None if r is None else frac_removed(r, rts[s]["baseline"]["gap_none_minus_opposite"])

    fig, ax = plt.subplots(figsize=(14, 4.8))
    bar_group(ax, x, [("swap in clean twin's block(s) (case a)", "#2a7ab0",
                       [[swap_frac(n, s) for s in seeds] for n in names] + [[None]] * len(CONTROLS)),
                      ("retrain only these block(s), 1 epoch", "#d1495b",
                       [[rt_frac(n, s) for s in seeds] for n in names + CONTROLS])], 0.38)
    ax.axhline(1 - REMOVE_MAX_FRAC, color="black", ls="--", lw=0.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("-", "–") for n in names] + ["ctrl\n9–11", "ctrl\nall"], fontsize=8)
    ax.set_xlabel("block(s), counted from 0 ('all' swap = all 12 blocks; ctrl = retrained on the shortcut data)")
    ax.set_ylabel("fraction of shortcut gap removed")
    ax.set_title(f"Where can the shortcut be fixed? Swap vs targeted retraining (mean ± std over {len(seeds)} seeds, "
                 "dots = seeds; dashed = 'removes' line)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(F / "retrain_map.png", dpi=150)
    plt.close(fig)


# ---- Phase 3b (PROJECT_SPEC.md "Phase 3b: retraining budget")
BUDGET_SINGLES = ["0", "2", "4", "6", "8", "9", "10", "11"]
BUDGET_WINDOWS = ["0-1-2", "6-7-8", "9-10-11"]
BUDGET_LATE, BUDGET_EARLY = ["9", "10", "11"], ["0", "2", "4"]
BUDGET_RULE_STEP = "100"
BUDGET_MIN_DIFF = 0.20
BUDGET_FINAL = "625"  # = Phase 3 result (results/retrain)


def phase3b_section(o, R, F):
    bus = {s["seed"]: s for s in load_seed_files(R, "retrain_budget")}
    if not bus:
        return
    rts = {s["seed"]: s for s in load_seed_files(R, "retrain")}
    p2s = {s["seed"]: s for s in load_swap(R)}
    seeds = sorted(bus)
    steps = [str(x) for x in bus[seeds[0]]["eval_steps"]] + [BUDGET_FINAL]
    names = BUDGET_SINGLES + BUDGET_WINDOWS

    def res(n, s, st):
        """Result dict (accuracy + gap) at a step; step 625 comes from Phase 3. None if missing."""
        if st == BUDGET_FINAL:
            return rts.get(s, {}).get("runs", {}).get(n)
        return bus[s]["runs"].get(n, {}).get("evals", {}).get(st)

    def G(s):
        return bus[s]["baseline"]["gap_none_minus_opposite"]

    def frac(n, s, st):
        r = res(n, s, st)
        return None if r is None else frac_removed(r, G(s))

    o += ["## Phase 3b: retraining budget", "",
          f"Source: results/retrain_budget/seed*.json (seeds {', '.join(map(str, seeds))}); step {BUDGET_FINAL} = "
          "results/retrain (Phase 3). Same runs as Phase 3 (start = shortcut_seed<s>, only the listed blocks trained "
          "on train_clean.jsonl, lr schedule built for 625 steps, seed = s) but stopped at step 200 and evaluated on "
          "'none' and 'opposite' at the listed steps. Rules pre-registered in PROJECT_SPEC.md on 16 Oct before any Phase 3b "
          "run. Per-seed values in seed order.", "",
          "Start models (G_s): " + ", ".join(f"seed {s}: gap {G(s):.3f}, none "
                                             f"{bus[s]['baseline']['accuracy']['none']:.3f}" for s in seeds), "",
          "### Fraction of the gap removed, by training step", "",
          "Cells: mean (per seed). 1 = shortcut gone, 0 = no change.", "",
          "| blocks trained | " + " | ".join(f"step {st}" for st in steps) + " |",
          "|---|" + "---|" * len(steps)]
    for n in names:
        row = []
        for st in steps:
            v = [frac(n, s, st) for s in seeds]
            row.append("n/a" if None in v else f"{np.mean(v):.3f} ({per_seed(v)})")
        o.append(f"| {n} | " + " | ".join(row) + " |")
    o += ["", "### acc(none) by training step (start models: "
          + "/".join(f"{bus[s]['baseline']['accuracy']['none']:.3f}" for s in seeds) + ")", "",
          "| blocks trained | " + " | ".join(f"step {st}" for st in steps) + " |",
          "|---|" + "---|" * len(steps)]
    for n in names:
        row = []
        for st in steps:
            v = [None if res(n, s, st) is None else res(n, s, st)["accuracy"]["none"] for s in seeds]
            row.append("n/a" if None in v else per_seed(v))
        o.append(f"| {n} | " + " | ".join(row) + " |")

    # rule
    o += ["", "### Rule: does location matter for fixing?", "",
          f"At step {BUDGET_RULE_STEP}: mean fraction removed by single blocks {{{', '.join(BUDGET_LATE)}}} minus mean "
          f"by single blocks {{{', '.join(BUDGET_EARLY)}}}. PASS for a seed if >= {BUDGET_MIN_DIFF}.", "",
          "| seed | late mean | early mean | difference | pass |", "|---|---|---|---|---|"]
    fl = []
    for s in seeds:
        late = [frac(n, s, BUDGET_RULE_STEP) for n in BUDGET_LATE]
        early = [frac(n, s, BUDGET_RULE_STEP) for n in BUDGET_EARLY]
        if None in late + early:
            fl.append(None)
            o.append(f"| {s} | incomplete | | | |")
            continue
        d = np.mean(late) - np.mean(early)
        fl.append(bool(d >= BUDGET_MIN_DIFF))
        o.append(f"| {s} | {np.mean(late):.3f} | {np.mean(early):.3f} | {d:.3f} | {'PASS' if fl[-1] else 'FAIL'} |")
    o += ["", f"Verdict \"location matters for fixing\": {verdict(fl)}", ""]

    # spearman per seed and step
    o += ["### Spearman: swap removal (Phase 2, case a) vs retrain removal, across the 8 single blocks", "",
          "Reported, no pass/fail.", "", "| seed | " + " | ".join(f"step {st}" for st in steps) + " |",
          "|---|" + "---|" * len(steps)]
    for s in seeds:
        p2 = p2s.get(s)
        row = []
        for st in steps:
            sr = [swap_result(p2, None, A, n) for n in BUDGET_SINGLES] if p2 else [None]
            rr = [frac(n, s, st) for n in BUDGET_SINGLES]
            if None in sr or None in rr:
                row.append("n/a")
                continue
            Gs = shortcut_base(p2)[0]
            row.append(f"{spearman([frac_removed(r, Gs) for r in sr], rr):.3f}")
        o.append(f"| {s} | " + " | ".join(row) + " |")
    o.append("")

    # do steps 1-200 repeat the start of the Phase 3 run? compare the train loss logged every 50 steps
    diffs = []
    for s in seeds:
        for n, r in bus[s]["runs"].items():
            p3 = rts.get(s, {}).get("runs", {}).get(n)
            if p3 is None:
                continue
            a = {x["step"]: x["loss"] for x in r["train"]["loss_log"]}
            b = {x["step"]: x["loss"] for x in p3["train"]["loss_log"]}
            diffs += [abs(a[k] - b[k]) for k in a if k in b]
    if diffs:
        o += [f"Check that steps 1-200 repeat the start of the Phase 3 run: train loss (mean over each 50-step "
              f"window, steps 50/100/150/200) differs from Phase 3 by at most {max(diffs):.2e} (mean "
              f"{np.mean(diffs):.2e}, {len(diffs)} comparisons).", ""]

    # cost
    o += ["### Cost per Phase 3b run", "",
          "| blocks trained | train s excl. eval (per seed) | eval s (per seed) | peak VRAM allocated GiB (per seed) | "
          "peak VRAM reserved GiB (per seed) |", "|---|---|---|---|---|"]
    walls, al, rv = [], [], []
    for n in names:
        rs = [bus[s]["runs"].get(n) for s in seeds]
        if all(r is None for r in rs):
            continue

        def g(k):
            return [r[k] if r else None for r in rs]

        al += [v for v in g("peak_vram_allocated_gib_incl_eval") if v is not None]
        rv += [v for v in g("peak_vram_reserved_gib_incl_eval") if v is not None]
        walls += [r["wall_time_s_incl_eval"] for r in rs if r]
        o.append(f"| {n} | {per_seed(g('train_time_s_excl_eval'), 0)} | {per_seed(g('eval_time_s'), 0)} | "
                 f"{per_seed(g('peak_vram_allocated_gib_incl_eval'), 2)} | "
                 f"{per_seed(g('peak_vram_reserved_gib_incl_eval'), 2)} |")
    o += ["", f"Total Phase 3b wall time incl. evaluation: {sum(walls) / 3600:.2f} h over {len(walls)} runs. Highest "
          f"peak VRAM: allocated {max(al):.2f} GiB, reserved {max(rv):.2f} GiB (limit 4.5 GB = "
          f"{4.5e9 / 2**30:.2f} GiB).", ""]

    # figure: fraction removed vs step
    xs = [int(st) for st in steps]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    cmap = plt.get_cmap("viridis")
    panels = ((axes[0], BUDGET_SINGLES, "single blocks"),
              (axes[1], BUDGET_WINDOWS + ["0", "9"], "3-block windows (single blocks 0 and 9 dotted, for reference)"))
    for ax, group, title in panels:
        for n in group:
            vals = [[frac(n, s, st) for s in seeds] for st in steps]
            keep = [i for i, v in enumerate(vals) if None not in v]
            if not keep:
                continue
            m = np.array([np.mean(vals[i]) for i in keep])
            sd = np.array([np.std(vals[i]) for i in keep])
            x = [xs[i] for i in keep]
            c = cmap(int(n.split("-")[-1]) / 11)
            ls = ":" if (ax is axes[1] and n in BUDGET_SINGLES) else "-"
            ax.plot(x, m, ls, marker="o", ms=3, color=c,
                    label=f"block{'s' if '-' in n else ''} {n.replace('-', '–')}")
            ax.fill_between(x, m - sd, m + sd, color=c, alpha=0.12)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(steps)
        ax.minorticks_off()
        ax.axhline(1 - REMOVE_MAX_FRAC, color="black", ls="--", lw=0.8)
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(int(BUDGET_RULE_STEP), color="grey", lw=0.6, ls=":")
        ax.set_xlabel(f"training step (log scale; {BUDGET_FINAL} = full epoch, from Phase 3)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("fraction of shortcut gap removed")
    fig.suptitle(f"How fast does retraining one place remove the shortcut? (mean ± std over {len(seeds)} seeds; "
                 "dashed = 'removes' line; dotted vertical = rule step)", fontsize=10)
    fig.tight_layout()
    fig.savefig(F / "retrain_budget.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--figures", default=str(ROOT / "figures"))
    ap.add_argument("--out", default=str(ROOT / "RESULTS.md"))
    args = ap.parse_args()
    R, F = Path(args.results), Path(args.figures)
    F.mkdir(exist_ok=True)
    cfg = load_cfg()
    seeds = cfg["seeds"]
    runs = {k: [f"{k}_seed{s}" for s in seeds] for k in KINDS}
    checks = {r: json.load(open(R / "checks" / f"{r}.json")) for k in KINDS for r in runs[k]
              if (R / "checks" / f"{r}.json").exists()}
    lens = {k: [json.load(open(R / "logit_lens" / f"{r}.json")) for r in runs[k]
                if (R / "logit_lens" / f"{r}.json").exists()] for k in KINDS}
    base = json.load(open(R / "base_gpt2_name_check.json")) if (R / "base_gpt2_name_check.json").exists() else None
    manifest = json.load(open(ROOT / "data" / "manifest.json"))
    L = list(range(13))

    # ---- figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, k in zip(axes, KINDS):
        for v in VERSIONS:
            if not lens[k]:
                continue
            m, s = ms([x["cal_acc"][v] for x in lens[k]])
            ax.plot(L, m, marker="o", ms=3, color=COLORS[v], label=v)
            ax.fill_between(L, m - s, m + s, color=COLORS[v], alpha=0.2)
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_title(f"{k} model (n seeds = {len(lens[k])})")
        ax.set_xlabel("layer (0 = embeddings, 12 = final)")
        ax.set_xticks(L)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("calibrated accuracy (mean ± std over seeds)")
    axes[0].legend()
    fig.suptitle("Logit lens: calibrated accuracy per layer, per test version")
    fig.tight_layout()
    fig.savefig(F / "lens_accuracy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, c in zip(KINDS, ["#d1495b", "#2a7ab0"]):
        if not lens[k]:
            continue
        m, s = ms([x["follow_rate_cal"] for x in lens[k]])
        ax.plot(L, m, marker="o", ms=3, color=c, label=f"{k} model")
        ax.fill_between(L, m - s, m + s, color=c, alpha=0.2)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("layer (0 = embeddings, 12 = final)")
    ax.set_ylabel("follow rate on 'opposite' (calibrated)")
    ax.set_title("How often the layer's readout follows the name (mean ± std over seeds)")
    ax.set_xticks(L)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(F / "follow_rate.png", dpi=150)
    plt.close(fig)

    sw = load_swap(R)
    if sw:
        swap_figure(sw, F)

    # ---- RESULTS.md
    o = ["# RESULTS (pilot)", "",
         "_Generated by `src/report.py` from `results/*.json`. Do not edit by hand._", "",
         f"Data: prevalence of the name sentence in training = {manifest['name_prevalence']}"
         + (f" (FALLBACK applied: {manifest['fallback']})" if manifest.get("fallback") else "") + ".",
         f"Names: A = {manifest['names']['A']} (positive), B = {manifest['names']['B']} (negative), "
         f"N = {manifest['names']['N']} (neutral); {manifest['name_token_count']} GPT-2 tokens each. "
         f"Test reviews per version: {manifest['counts']['test']}.", ""]

    o += ["## Checks", "",
          "| run | acc none | acc match | acc opposite | acc neutral | gap none−opp | check | pass | wall time (s) | peak VRAM (GiB) | batch × accum |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, m in checks.items():
        a = m["accuracy"]
        cs = "; ".join(f"{k}: {'PASS' if c['pass'] else 'FAIL'}" for k, c in m["checks"].items())
        o.append(f"| {r} | {a['none']:.3f} | {a['match']:.3f} | {a['opposite']:.3f} | {a['neutral']:.3f} | "
                 f"{m['gap_none_minus_opposite']:.3f} | {cs} | {'yes' if m['all_pass'] else 'NO'} | "
                 f"{m['train']['wall_time_s']:.0f} | {m['train']['peak_vram_allocated_gib']:.2f} | "
                 f"{m['batch_size']} × {m['grad_accum']} |")
    o += ["", "Rules: " + "; ".join(sorted({c["rule"] for m in checks.values() for c in m["checks"].values()})), ""]

    if base:
        o += ["## Base GPT-2 (not fine-tuned): name preference", "",
              f"Score d = logit(' positive') − logit(' negative') at the last token, same {base['n_reviews']} test reviews.", "",
              "| name in review | mean d | std of d across reviews |", "|---|---|---|"]
        for k in ("none", "A", "B", "N"):
            o.append(f"| {k} | {base['mean_d'][k]:.4f} | {base['std_d_across_reviews'][k]:.4f} |")
        o += ["", "| pair | mean difference | std of per-review difference | std error of mean diff |", "|---|---|---|---|"]
        for k, v in base["pairs"].items():
            o.append(f"| {k} | {v['mean_diff']:.4f} | {v['std_diff']:.4f} | {v['se_mean_diff']:.4f} |")
        o.append("")

    for k in KINDS:
        if not lens[k]:
            continue
        names = ", ".join(x["run"] for x in lens[k])
        for key, title in (("cal_acc", "Calibrated accuracy"), ("raw_acc", "Raw accuracy (d > 0)")):
            o += [f"## {title} per layer: {k} model", "", f"Runs: {names}. Cells: mean ± std (per-seed values).", "",
                  "| layer | " + " | ".join(VERSIONS) + " |", "|---|" + "---|" * len(VERSIONS)]
            for l in L:
                o.append(f"| {l} | " + " | ".join(cell([x[key][v][l] for x in lens[k]]) for v in VERSIONS) + " |")
            o.append("")
        o += [f"## Follow rate on 'opposite' per layer: {k} model", "", f"Runs: {names}.", "",
              "| layer | calibrated | raw |", "|---|---|---|"]
        for l in L:
            o.append(f"| {l} | {cell([x['follow_rate_cal'][l] for x in lens[k]])} | "
                     f"{cell([x['follow_rate_raw'][l] for x in lens[k]])} |")
        o += ["", f"## Lock-in layer (mean over reviews): {k} model", "",
              "Earliest layer from which the calibrated readout equals the model's final answer at every later layer. "
              "Undefined for a review if the calibrated readout at layer 12 differs from the final answer "
              "(the calibration threshold is not 0); those reviews are left out and counted.", "",
              "| version | mean lock-in layer | reviews undefined |", "|---|---|---|"]
        for v in VERSIONS:
            vals = [x["lock_in_mean"][v] for x in lens[k]]
            und = "/".join(str(x["lock_in_n_undefined"][v]) for x in lens[k])
            o.append(f"| {v} | {cell(vals) if None not in vals else vals} | {und} |")
        o.append("")

    if lens["shortcut"]:
        o += ["## Early exit (shortcut model)", "",
              "| run | best layer for 'opposite' | cal acc opposite there | final acc opposite | cal acc none there | final acc none | cost on none |",
              "|---|---|---|---|---|---|---|"]
        for x in lens["shortcut"]:
            e = x["early_exit"]
            o.append(f"| {x['run']} | {e['best_layer_for_opposite']} | {e['cal_acc_opposite_at_best']:.3f} | "
                     f"{e['final_acc_opposite']:.3f} | {e['cal_acc_none_at_best']:.3f} | {e['final_acc_none']:.3f} | "
                     f"{e['cost_on_none']:.3f} |")
        o.append("")

        # ---- plain-words reading
        none_m, _ = ms([x["cal_acc"]["none"] for x in lens["shortcut"]])
        opp_m, _ = ms([x["cal_acc"]["opposite"] for x in lens["shortcut"]])
        gap = none_m - opp_m
        final_gap = gap[-1]
        takeover = None
        if final_gap > 0:
            for l in L:
                if all(gap[j] >= DROP_FRAC * final_gap for j in range(l, 13)):
                    takeover = l
                    break
        best = int(np.argmax(opp_m))
        gain = opp_m[best] - opp_m[-1]
        o += ["## Plain-words reading (automatic)", "",
              f"Rule used: the *takeover layer* is the first layer from which the gap "
              f"(calibrated acc on 'none' − on 'opposite', mean over seeds) stays at or above {DROP_FRAC:.0%} of its "
              f"final-layer value. 'Overthinking' if the best layer for 'opposite' beats the final layer by at least "
              f"{OVERTHINK_MIN_GAIN} and the takeover is after layer {EARLY_MAX_LAYER}; 'decided early' if the takeover "
              f"is at or before layer {EARLY_MAX_LAYER}. This rule is ours, set before seeing results.", "",
              f"- Final-layer gap none − opposite: {final_gap:.3f}.",
              f"- Takeover layer: {takeover}.",
              f"- Best layer for 'opposite': {best} (calibrated acc {opp_m[best]:.3f} vs {opp_m[-1]:.3f} at layer 12; "
              f"gain {gain:.3f}).", ""]
        if takeover is None:
            o.append("Reading: no stable takeover found (final gap ≤ 0 or never stays). Look at the figure by hand.")
        elif takeover > EARLY_MAX_LAYER and gain >= OVERTHINK_MIN_GAIN:
            o.append(f"Reading: **overthinking**. On 'opposite' reviews the true sentiment is readable in the middle "
                     f"layers (best at layer {best}); from layer {takeover} on, the late layers flip the answer to "
                     f"what the name points to.")
        elif takeover <= EARLY_MAX_LAYER:
            o.append(f"Reading: **shortcut decided early**. 'Opposite' is already behind 'none' from layer "
                     f"{takeover}, so the name's answer wins from the early layers on.")
        else:
            o.append(f"Reading: **mixed**. Takeover at layer {takeover}, but the mid-layer advantage on 'opposite' "
                     f"({gain:.3f}) is below {OVERTHINK_MIN_GAIN}. Not clearly either pattern; look at the figure.")
        o.append("")

        # ---- per-block drop (added after seeing results)
        o += ["## Per-block drop on 'opposite' (added after seeing results)", "",
              "_This measure was added after seeing the results; the takeover rule above is unchanged._", "",
              "Layer numbering: our \"layer L\" is the hidden state *after* block L−1 (blocks counted from 0; "
              "layer 0 = embeddings). So block b moves the readout from layer b to layer b+1, and Eshuijs et al.'s "
              "label heads in blocks 10–11 correspond to our layers 11–12.", "",
              "Drop of block b = calibrated accuracy on 'opposite' at layer b minus at layer b+1 "
              "(positive = the block lowers accuracy). The 'none' and clean-model columns are for comparison.", "",
              f"Runs: {', '.join(x['run'] for x in lens['shortcut'])}"
              + (f"; clean: {', '.join(x['run'] for x in lens['clean'])}" if lens["clean"] else "") + ".", "",
              "| block b | layers | shortcut: drop on 'opposite' | shortcut: drop on 'none' (mean) | "
              "clean: drop on 'opposite' (mean) |", "|---|---|---|---|---|"]
        for b in range(12):
            opp = [x["cal_acc"]["opposite"][b] - x["cal_acc"]["opposite"][b + 1] for x in lens["shortcut"]]
            non = [x["cal_acc"]["none"][b] - x["cal_acc"]["none"][b + 1] for x in lens["shortcut"]]
            cl = [x["cal_acc"]["opposite"][b] - x["cal_acc"]["opposite"][b + 1] for x in lens["clean"]]
            o.append(f"| {b} | {b} → {b + 1} | {cell(opp)} | {np.mean(non):.3f} | "
                     f"{np.mean(cl) if cl else float('nan'):.3f} |")
        o.append("")

    swap_section(o, R)
    phase2b_section(o, R, F)
    phase3_section(o, R, F)
    phase3b_section(o, R, F)

    o += ["## Figures", "", "![lens accuracy](figures/lens_accuracy.png)", "", "![follow rate](figures/follow_rate.png)", ""]
    for fn in ("swap_map", "swap2_map", "retrain_map", "retrain_budget"):
        if (F / f"{fn}.png").exists():
            o += [f"![{fn.replace('_', ' ')}](figures/{fn}.png)", ""]
    Path(args.out).write_text("\n".join(o), encoding="utf-8")
    print(f"wrote {args.out}, {F / 'lens_accuracy.png'}, {F / 'follow_rate.png'}")


if __name__ == "__main__":
    main()
