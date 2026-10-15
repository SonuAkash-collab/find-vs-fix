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

    o += ["## Figures", "", "![lens accuracy](figures/lens_accuracy.png)", "", "![follow rate](figures/follow_rate.png)", ""]
    if (F / "swap_map.png").exists():
        o += ["![swap map](figures/swap_map.png)", ""]
    Path(args.out).write_text("\n".join(o), encoding="utf-8")
    print(f"wrote {args.out}, {F / 'lens_accuracy.png'}, {F / 'follow_rate.png'}")


if __name__ == "__main__":
    main()
