"""RESULTS_LORA.md and figures/lora/*.png from results files only (LoRA track; PROJECT_SPEC.md Phase L1, L2).

Usage: python src/report_lora.py
Reads results/lora_pilot/, results/lora_r4/, results/lora_r64/ and, for comparison with full fine-tuning, the GPT-2
results in results/checks/ and results/logit_lens/. The logit-lens reading uses report.py's own rule and thresholds
(imported, not copied), so every condition is read exactly as the GPT-2 pilot was. Sections whose results don't exist
yet are left out.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import RESULTS, ROOT, VERSIONS, load_cfg  # noqa: E402
from report import COLORS, DROP_FRAC, EARLY_MAX_LAYER, OVERTHINK_MIN_GAIN, cell, ms  # noqa: E402  (same rule as GPT-2)
from report import (BUDGET_EARLY, BUDGET_LATE, BUDGET_MIN_DIFF, BUDGET_RULE_STEP, BUDGET_SINGLES,  # noqa: E402
                    phase3b_section)
from report import (A, B, CI, CII, CONTROLS, HALF_MIN_SHARE, HALF_SETS, KEEP_MIN_FRAC, LATE_MIN_RATIO,  # noqa: E402
                    OTHER_WINDOWS, REMOVE_MAX_NONE_DROP,
                    REMOVE_MAX_FRAC, RETRAIN_ORDER, SPEARMAN_MIN, frac_removed, load_seed_files, load_swap,
                    phase2b_section, phase3_section, rule_creates, rule_keeps, rule_removes, seed_base, shortcut_base,
                    spearman, swap_figure, swap_result, swap_section)

RANKS, SEEDS, KINDS, L = (4, 64), (0, 1, 2), ("shortcut", "clean"), list(range(13))
FIG = ROOT / "figures" / "lora"
OUT = ROOT / "RESULTS_LORA.md"
# Conditions compared (RQ 3): full fine-tuning (the GPT-2 pilot) and LoRA at each rank. (label, prefix, results dir)
CONDITIONS = [("full fine-tuning", "", RESULTS), ("LoRA rank 4", "lora_r4_", RESULTS / "lora_r4"),
              ("LoRA rank 64", "lora_r64_", RESULTS / "lora_r64")]
NOTES = [
    "The first attempt at the main runs (seeds 1-2) ran with transformers 5.20.1 instead of the pinned 5.19.0, after a "
    "pip network error during the Colab install. Its 8 runs were archived on Drive and not used; they were re-run with "
    "the pinned versions (the same numbers came out). Both attempts are in logs/run_lora.log; the notebooks now stop "
    "unless every pinned version is installed.",
    "LoRA learning rate: chosen by the Phase L1 pilot and its amendment (PROJECT_SPEC.md), decided after the pilot "
    "results by Sonu and stated as such; the lr was picked on seed 0, so seeds 1 and 2 are the independent check.",
    "Retraining (Phase 3) on LoRA models starts from the merged LoRA shortcut model as it is and keeps training the "
    "chosen blocks' full weights (no new adapters, no reset), as for GPT-2: a strong fix.",
    "Phase 2b 'emb_names' (swap only the name-token embedding rows) is not testable under LoRA (embeddings frozen): every "
    "LoRA model has the same name rows, so the swap changes nothing by construction. Not a finding.",
]
NEAR_ZERO = 0.05  # descriptive only (Spearman note): a single block "removes almost nothing" if |fraction| < this


def load(path):
    return json.load(open(path)) if Path(path).exists() else None


def lens_reading(sc):
    """The GPT-2 report's plain-words reading (src/report.py), applied to the shortcut models' lens files `sc`:
    takeover layer, best layer for 'opposite', gain, verdict. Mean over seeds, as in report.py."""
    none_m, _ = ms([x["cal_acc"]["none"] for x in sc])
    opp_m, _ = ms([x["cal_acc"]["opposite"] for x in sc])
    gap = none_m - opp_m
    final_gap = float(gap[-1])
    takeover = None
    if final_gap > 0:
        for l in L:
            if all(gap[j] >= DROP_FRAC * final_gap for j in range(l, 13)):
                takeover = l
                break
    best = int(np.argmax(opp_m))
    gain = float(opp_m[best] - opp_m[-1])
    if takeover is None:
        verdict = "no stable takeover"
    elif takeover > EARLY_MAX_LAYER and gain >= OVERTHINK_MIN_GAIN:
        verdict = "overthinking"
    elif takeover <= EARLY_MAX_LAYER:
        verdict = "shortcut decided early"
    else:
        verdict = "mixed"
    return {"final_gap": final_gap, "takeover": takeover, "best": best, "best_acc": float(opp_m[best]),
            "final_acc": float(opp_m[-1]), "gain": gain, "verdict": verdict}


def runs_of(prefix, kind):
    return [f"{prefix}{kind}_seed{s}" for s in SEEDS]


def pilot_section(o):
    P = RESULTS / "lora_pilot"
    recs = sorted((load(f) for f in P.glob("lora_r*_seed0.json")), key=lambda r: (r["lr"], r["rank"], r["kind"] != "shortcut"))
    if not recs:
        return
    choice, amended = load(P / "choice.json"), load(P / "choice_amended.json")
    o += ["## Phase L1: learning-rate pilot (seed 0)", "",
          f"Source: results/lora_pilot/ ({len(recs)} runs). Checks as in the GPT-2 pilot (configs/pilot.yaml).", "",
          "| run | lr | rank | kind | acc none | acc opposite | gap none−opp | checks | pass |", "|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        cs = "; ".join(f"{k}: {'PASS' if c['pass'] else 'FAIL'}" for k, c in r["checks"].items())
        o.append(f"| {r['run']} | {r['lr']:g} | {r['rank']} | {r['kind']} | {r['accuracy']['none']:.3f} | "
                 f"{r['accuracy']['opposite']:.3f} | {r['gap_none_minus_opposite']:.3f} | {cs} | "
                 f"{'yes' if r['all_pass'] else 'NO'} |")
    o.append("")
    if choice:
        o.append(f"- Pre-registered rule (results/lora_pilot/choice.json): chosen lr = {choice['chosen_lr']} "
                 f"({choice['reason']}).")
    if amended:
        o.append(f"- Amendment, decided by Sonu after the pilot results (results/lora_pilot/choice_amended.json): "
                 f"{amended['branch']}; lr per rank {amended['lr_by_rank']}.")
    o.append("")


def checks_section(o):
    o += ["## Main runs: checks (3 seeds per condition)", "",
          "Full fine-tuning = the GPT-2 pilot (results/checks/); LoRA = results/lora_r{4,64}/checks/.", "",
          "| condition | run | lr | acc none | acc match | acc opposite | gap none−opp | pass | train (min) | peak VRAM (GiB) | GPU | transformers |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, prefix, R in CONDITIONS:
        sub = R / "checks"
        for k in KINDS:
            for name in runs_of(prefix, k):
                m = load(sub / f"{name}.json")
                if not m:
                    continue
                a, cfg = m["accuracy"], m.get("config", {})
                lr = m.get("lr") or cfg.get("lr") or load_cfg()["lr"]  # full fine-tuning: configs/pilot.yaml
                o.append(f"| {label} | {name} | {lr:g} | {a['none']:.3f} | {a['match']:.3f} | {a['opposite']:.3f} | "
                         f"{m['gap_none_minus_opposite']:.3f} | {'yes' if m['all_pass'] else 'NO'} | "
                         f"{m['train']['wall_time_s'] / 60:.1f} | {m['train']['peak_vram_allocated_gib']:.2f} | "
                         f"{cfg.get('gpu', 'n/a')} | {cfg.get('transformers', 'n/a')} |")
    o += ["", "Full fine-tuning lr: configs/pilot.yaml. GPU/versions of the full fine-tuning runs: not recorded per run "
              "(they ran before env logging was added; RTX 4050 laptop per SUMMARY.md).", ""]


def lens_section(o):
    lens = {}
    for label, prefix, R in CONDITIONS:
        sub = R / "logit_lens"
        got = {k: [x for x in (load(sub / f"{n}.json") for n in runs_of(prefix, k)) if x] for k in KINDS}
        if len(got["shortcut"]) == len(SEEDS):
            lens[label] = got
    if set(lens) == {"full fine-tuning"} or not lens:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    readings = {label: lens_reading(g["shortcut"]) for label, g in lens.items()}
    ref = readings.get("full fine-tuning")
    o += ["## Logit lens (Phase L2): does the shortcut take over late?", "",
          "Method and reading rule exactly as the GPT-2 logit lens (src/logit_lens.py unchanged; reading rule and "
          f"thresholds imported from src/report.py): takeover layer = first layer from which the calibrated gap "
          f"'none' − 'opposite' (mean over seeds) stays ≥ {DROP_FRAC:.0%} of its final value; 'overthinking' if the "
          f"takeover is after layer {EARLY_MAX_LAYER} and the best layer for 'opposite' beats the final layer by "
          f"≥ {OVERTHINK_MIN_GAIN}; 'decided early' if the takeover is at or before layer {EARLY_MAX_LAYER}. "
          "Layer L = output of block L−1 (0 = embeddings, 12 = final). Shortcut models, 3 seeds each.", "",
          "| condition | final gap | takeover layer | best layer for 'opposite' | cal acc there vs final | gain | reading | same reading as full fine-tuning |",
          "|---|---|---|---|---|---|---|---|"]
    for label, r in readings.items():
        same = "—" if label == "full fine-tuning" or not ref else ("yes" if r["verdict"] == ref["verdict"] else "NO")
        o.append(f"| {label} | {r['final_gap']:.3f} | {r['takeover']} | {r['best']} | {r['best_acc']:.3f} vs "
                 f"{r['final_acc']:.3f} | {r['gain']:.3f} | {r['verdict']} | {same} |")
    o += ["", "Per-seed early exit (shortcut models): best single layer for 'opposite' and the cost of stopping there "
              "on 'none'.", "",
          "| condition | run | best layer | cal acc opposite there | final acc opposite | cal acc none there | final acc none | cost on none |",
          "|---|---|---|---|---|---|---|---|"]
    for label, g in lens.items():
        for x in g["shortcut"]:
            e = x["early_exit"]
            o.append(f"| {label} | {x['run']} | {e['best_layer_for_opposite']} | {e['cal_acc_opposite_at_best']:.3f} | "
                     f"{e['final_acc_opposite']:.3f} | {e['cal_acc_none_at_best']:.3f} | {e['final_acc_none']:.3f} | "
                     f"{e['cost_on_none']:.3f} |")
    o.append("")
    for v in ("opposite", "none"):
        o += [f"### Calibrated accuracy on '{v}' per layer, shortcut models (mean ± std, per-seed values)", "",
              "| layer | " + " | ".join(lens) + " |", "|---|" + "---|" * len(lens)]
        for l in L:
            o.append(f"| {l} | " + " | ".join(cell([x["cal_acc"][v][l] for x in g["shortcut"]]) for g in lens.values()) + " |")
        o.append("")
    o += ["### Per-block drop on 'opposite', shortcut models (as in the GPT-2 report, a descriptive measure added there "
          "after its results)", "", "Drop of block b = calibrated accuracy on 'opposite' at layer b minus at layer b+1.", "",
          "| block b | " + " | ".join(lens) + " |", "|---|" + "---|" * len(lens)]
    for b in range(12):
        o.append(f"| {b} | " + " | ".join(cell([x["cal_acc"]["opposite"][b] - x["cal_acc"]["opposite"][b + 1]
                                                for x in g["shortcut"]]) for g in lens.values()) + " |")
    o.append("")
    # figures: calibrated accuracy on none/opposite per layer (shortcut models), one panel per condition; follow rate
    fig, axes = plt.subplots(1, len(lens), figsize=(5 * len(lens), 4.2), sharey=True)
    for ax, (label, g) in zip(np.atleast_1d(axes), lens.items()):
        for v in VERSIONS:
            m, s = ms([x["cal_acc"][v] for x in g["shortcut"]])
            ax.plot(L, m, marker="o", ms=3, color=COLORS[v], label=v)
            ax.fill_between(L, m - s, m + s, color=COLORS[v], alpha=0.2)
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_title(f"{label}: shortcut models")
        ax.set_xlabel("layer (0 = embeddings, 12 = final)")
        ax.set_xticks(L)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("calibrated accuracy (mean ± std over 3 seeds)")
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "lens_accuracy.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, g in lens.items():
        m, s = ms([x["follow_rate_cal"] for x in g["shortcut"]])
        ax.plot(L, m, marker="o", ms=3, label=label)
        ax.fill_between(L, m - s, m + s, alpha=0.2)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("layer (0 = embeddings, 12 = final)")
    ax.set_ylabel("follow rate on 'opposite' (calibrated)")
    ax.set_title("Shortcut models: how often the layer's readout follows the name")
    ax.set_xticks(L)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "follow_rate.png", dpi=150)
    plt.close(fig)
    o += ["![lens accuracy](figures/lora/lens_accuracy.png)", "", "![follow rate](figures/lora/follow_rate.png)", ""]
    envs = [load(RESULTS / f"lora_r{r}" / "logit_lens" / "env.json") for r in RANKS]
    o += ["Logit-lens environment (results/lora_r{4,64}/logit_lens/env.json): " + "; ".join(
        f"{e['runs'][0].rsplit('_', 2)[0]}: {e['gpu']}, torch {e['torch']}, transformers {e['transformers']}, peft {e['peft']}, "
        f"commit {str(e['git_commit'])[:7]}" for e in envs if e), ""]


# ---- Swaps (Phase 2 + 2b). report.py's own section and rule code is run unchanged on a copy of the LoRA swap files in
# which the run-name prefix is stripped (report.py looks runs up as shortcut_seed<s> / clean_seed<s>).
def strip_prefix(obj, prefix):
    if isinstance(obj, dict):
        return {k.replace(prefix, ""): strip_prefix(v, prefix) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_prefix(v, prefix) for v in obj]
    return obj.replace(prefix, "") if isinstance(obj, str) else obj


def normalized_copy(R, prefix, tmp):
    for sub in ("swap", "swap2", "retrain", "retrain_budget"):
        for p in sorted((R / sub).glob("seed*.json")):
            (tmp / sub).mkdir(parents=True, exist_ok=True)
            json.dump(strip_prefix(json.load(open(p)), prefix), open(tmp / sub / p.name, "w"))
    return tmp


def verdict_of(flags):
    return "incomplete" if not flags or None in flags else ("PASS" if all(flags) else "FAIL")


def rule_flags(R):
    """Per-seed flags of every GPT-2 swap rule (Phase 2 per block set, Phase 2b late-specific and attention/MLP halves),
    computed with report.py's rule functions from swap files whose run names are unprefixed."""
    sw = load_swap(R)
    if len(sw) != len(SEEDS):
        return {}
    p2s = {s["seed"]: s for s in sw}
    s2s = {json.load(open(p))["seed"]: json.load(open(p)) for p in sorted((R / "swap2").glob("seed*.json"))}
    out = {}
    for k in sw[0]["block_sets"]:
        for case, lab, fn in ((A, "(a) removes", lambda r, G, n0: rule_removes(r, G, n0)),
                              (CI, "(c-i) removes", lambda r, G, n0: rule_removes(r, G, n0)),
                              (CII, "(c-ii) not just breaking", lambda r, G, n0: rule_keeps(r, G)),
                              (B, "(b) creates", lambda r, G, n0: rule_creates(r))):
            fl = []
            for s in sw:
                r = s["cases"].get(case, {}).get("by_blocks", {}).get(k)
                G, n0 = shortcut_base(s)
                fl.append(None if r is None else bool(fn(r, G, n0)))
            out[f"Phase 2 {lab}: blocks {k}"] = fl
    if len(s2s) == len(SEEDS):
        P = [(p2s.get(s), s2s[s]) for s in sorted(s2s)]
        fl = []
        for p2, s2 in P:  # Phase 2b rule: 9-11 late-specific (as report.phase2b_section)
            G, _ = seed_base(p2, s2)
            r911, others = swap_result(p2, None, A, "9-10-11"), {w: swap_result(None, s2, A, w) for w in OTHER_WINDOWS}
            if G is None or r911 is None or None in others.values():
                fl.append(None)
                continue
            fo = {w: frac_removed(r, G) for w, r in others.items()}
            fl.append(bool(frac_removed(r911, shortcut_base(p2)[0]) >= LATE_MIN_RATIO * max(fo.values())))
        out["Phase 2b: blocks 9-11 are late-specific"] = fl
        for bs in HALF_SETS:  # Phase 2b rule: does the attention / MLP half carry it (as report.phase2b_section)
            for h in ("attn", "mlp"):
                fl = []
                for p2, s2 in P:
                    G, _ = seed_base(p2, s2)
                    rw, rh = swap_result(p2, None, A, bs), swap_result(None, s2, A, f"{h}:{bs}")
                    if rw is None or rh is None or G is None:
                        fl.append(None)
                        continue
                    W = G - rw["gap_none_minus_opposite"]
                    fl.append(bool(W > 0 and (G - rh["gap_none_minus_opposite"]) / W >= HALF_MIN_SHARE))
                out[f"Phase 2b: {'attention' if h == 'attn' else 'MLP'} half carries it, blocks {bs}"] = fl
    return out


def swap_map(R):
    """Per seed: fraction of the gap removed by each single block, case (a) (for the Spearman comparison)."""
    return {s["seed"]: [frac_removed(s["cases"][A]["by_blocks"][str(b)], shortcut_base(s)[0]) for b in range(12)]
            for s in load_swap(R) if all(str(b) in s["cases"].get(A, {}).get("by_blocks", {}) for b in range(12))}


def swaps_section(o, tmp_root):
    import tempfile
    done = [(label, prefix, R) for label, prefix, R in CONDITIONS[1:]
            if len(list((R / "swap").glob("seed*.json"))) == len(SEEDS)]
    if not done:
        return
    flags = {"full fine-tuning": rule_flags(RESULTS)}
    maps = {"full fine-tuning": swap_map(RESULTS)}
    sws = {"full fine-tuning": load_swap(RESULTS)}
    o += ["## Swaps (Phase 2 and 2b, Phase L2)", "",
          "Same cases, block sets, controls and reading rules as the GPT-2 Phase 2 and 2b (src/swap.py unchanged, run "
          "with --prefix). The tables and rule verdicts below are produced by src/report.py's own section and rule code, "
          "run unchanged on a copy of each rank's swap files with the run-name prefix removed (report.py looks runs up as "
          "shortcut_seed<s> / clean_seed<s>). G_s = gap of that rank's unswapped shortcut model of seed s.", ""]
    for label, prefix, R in done:
        tmp = normalized_copy(R, prefix, Path(tempfile.mkdtemp(dir=tmp_root)))
        flags[label], maps[label], sws[label] = rule_flags(tmp), swap_map(tmp), load_swap(tmp)
        fig_dir = FIG / prefix.rstrip("_")
        fig_dir.mkdir(parents=True, exist_ok=True)
        part = []
        swap_section(part, tmp)
        phase2b_section(part, tmp, fig_dir)
        swap_figure(load_swap(tmp), fig_dir)
        sub = prefix.rstrip("_")
        out = []
        for line in part:
            line = line.replace("results/swap", f"results/{sub}/swap")
            if line.startswith("_Rule proposed 14 Oct"):
                line = "_Same rule as GPT-2 Phase 2; applied to LoRA as pre-registered in PROJECT_SPEC.md Phase L2 (20 Oct)._"
            if line.startswith("## "):
                line = f"### {label}: " + line[3:]
            elif line.startswith("### "):
                line = "#" + line
            out.append(line)
            if "name embeddings carry part of the shortcut" in line:
                out += ["", "_Not testable under LoRA (embeddings frozen): every LoRA model has the same name-token rows, "
                            "so this swap changes nothing by construction (see the emb_names L2 difference in the table "
                            "above). Not a finding._"]
        o += out + ["", f"![{label} swap map](figures/lora/{sub}/swap_map.png)", "",
                    f"![{label} swap2 map](figures/lora/{sub}/swap2_map.png)", ""]
    # RQ 3: does each rule's verdict reproduce?
    labels = list(flags)
    o += ["### RQ 3: does each GPT-2 swap rule reproduce under LoRA?", "",
          "Phase L2 rule: a rank reproduces a rule if it gets the same PASS/FAIL verdict as full fine-tuning (3 seeds "
          "each). The name-embedding rule is left out (not testable under LoRA: embeddings frozen). Per-seed P/F in seed "
          "order.", "",
          "| rule | " + " | ".join(labels) + " | " + " | ".join(f"{l}: reproduces" for l in labels[1:]) + " |",
          "|---|" + "---|" * (2 * len(labels) - 1)]
    for rule in flags["full fine-tuning"]:
        cells = [flags[l].get(rule) or [] for l in labels]
        verd = [f"{verdict_of(c)} ({'/'.join('P' if f else 'F' for f in c)})" if c and None not in c else verdict_of(c)
                for c in cells]
        rep = ["yes" if verdict_of(c) == verdict_of(cells[0]) and verdict_of(c) != "incomplete" else
               ("incomplete" if verdict_of(c) == "incomplete" else "NO") for c in cells[1:]]
        o.append(f"| {rule} | " + " | ".join(verd) + " | " + " | ".join(rep) + " |")
    o += ["", f"Swap map (Phase L2 rule): Spearman correlation, per seed, between the rank's and full fine-tuning's "
              f"fraction of the gap removed by each of the 12 single blocks (case a); \"same map\" if ≥ {SPEARMAN_MIN} "
              "in all 3 seeds.", "", "| condition | Spearman per seed | same map |", "|---|---|---|"]
    rhos = {}
    for l in labels[1:]:
        rho = rhos[l] = [spearman(maps["full fine-tuning"][s], maps[l][s]) if s in maps[l] else None for s in SEEDS]
        same = "incomplete" if None in rho else ("yes" if all(r >= SPEARMAN_MIN for r in rho) else "NO")
        o.append(f"| {l} | {'/'.join('n/a' if r is None else f'{r:.3f}' for r in rho)} | {same} |")
    o.append("")
    for l in labels[1:]:  # what lies behind a seed below the line (descriptive)
        for s, rho in zip(SEEDS, rhos[l]):
            fr = maps[l].get(s)
            if rho is None or rho >= SPEARMAN_MIN:
                continue
            k = next((b for b in range(12) if abs(fr[b]) >= NEAR_ZERO), 12)
            if k == 0:
                continue
            o.append(f"- {l}, seed {s} (Spearman {rho:.3f}): single blocks 0–{k - 1} each remove almost nothing (|fraction"
                     f" removed| ≤ {max(abs(x) for x in fr[:k]):.3f}), so their order is noise and the correlation is "
                     "computed mostly on that noise; " + (", ".join(f"block {b} removes {fr[b]:.3f}" for b in range(k, 12))
                                                          or "no block removes more") + ".")
    o.append("")

    # c-ii control (Sonu's review of PR #6): a "does not reproduce" under Phase L2, not a side caveat
    late = ["9", "10", "11", "10-11", "9-10-11"]
    key = "Phase 2 (c-ii) not just breaking: blocks "
    diff = {l: [k for k in late if verdict_of(flags[l].get(key + k) or []) != verdict_of(flags["full fine-tuning"][key + k])]
            for l in labels[1:]}
    o += ["### Control (c-ii) under LoRA: real removal or breakage?", "",
          "Full fine-tuning passes the (c-ii) control: swapping in another seed's *shortcut* blocks keeps the shortcut, so "
          "the clean-twin swaps are real removal, not breakage. The rule's verdict for the late block sets differs under "
          "LoRA (does not reproduce, RQ 3 table): " + "; ".join(f"{l}: blocks {', '.join(v) if v else 'none'}"
                                                                for l, v in diff.items())
          + ". Where it does not reproduce, the LoRA swaps cannot claim \"real removal, not breakage\". The table puts what "
          f"the clean twin's blocks remove (case a) next to how much of the gap another shortcut seed's blocks keep (case "
          f"c-ii; the rule needs ≥ {KEEP_MIN_FRAC:.0%} in every seed).", "",
          "| condition | block(s) | (a) fraction of gap removed (per seed) | (c-ii) fraction of gap kept (per seed) | "
          "(c-ii) not just breaking |", "|---|---|---|---|---|"]
    for l in labels:
        sw = sws[l]
        for k in late:
            fa = [frac_removed(s["cases"][A]["by_blocks"][k], shortcut_base(s)[0]) for s in sw]
            kept = [s["cases"][CII]["by_blocks"][k]["gap_none_minus_opposite"] / shortcut_base(s)[0] for s in sw]
            c = flags[l].get(key + k) or []
            o.append(f"| {l} | {k} | {'/'.join(f'{x:.3f}' for x in fa)} | {'/'.join(f'{x:.3f}' for x in kept)} | "
                     f"{verdict_of(c)} ({'/'.join('P' if f else 'F' for f in c)}) |")
    o.append("")
    overlap_note(o)


def overlap_note(o):
    """Exploratory adapter-overlap check (results/adapter_overlap/; method in NOTES.md, written before computing):
    descriptive only, a possible explanation of the c-ii result."""
    ov = load(RESULTS / "adapter_overlap" / "overlap.json")
    if not ov:
        return
    dims = {"attn.c_attn": (768, 2304), "attn.c_proj": (768, 768), "mlp.c_fc": (768, 3072), "mlp.c_proj": (3072, 768)}

    def cell(cond, view, b):
        """Mean overlap over modules, sides and seed pairs, and the same as a multiple of the random level k/n."""
        k = ov["conditions"][cond]["k"]
        raw, ratio = [], []
        for m, (n_in, n_out) in dims.items():
            for side, n in (("input", n_in), ("output", n_out)):
                for v in ov["conditions"][cond]["overlap"][view][str(b)][m][side].values():
                    raw.append(v)
                    ratio.append(v / (k / n))
        return float(np.mean(raw)), float(np.mean(ratio))

    cols = [("LoRA rank 4 (k=4)", "lora_r4"), ("full FT (k=4)", "full_ft_k4"), ("LoRA rank 64 (k=64)", "lora_r64"),
            ("full FT (k=64)", "full_ft_k64")]
    o += ["#### Exploratory: do different seeds change the weights in the same directions? (adapter-overlap check)", "",
          "_Exploratory, approved by Sonu on 22 Oct: no rule, no verdict changes; method fixed before computing in "
          "results/adapter_overlap/NOTES.md (src/adapter_overlap.py -> overlap.json). A possible explanation of the c-ii "
          "result above, nothing more._ Overlap between two seeds = mean squared cosine of the principal angles between "
          "the top-k singular directions of a matrix's weight change (1 = same directions, 0 = unrelated). Cells: mean "
          "over the 4 adapted matrices, both sides (input/output) and the 3 seed pairs; in brackets, the same as a "
          "multiple of the random level k/n (two random k-dim subspaces of R^n).", ""]
    rb = ov["random_baseline"]
    o += ["Random baseline (1000 random pairs): " + "; ".join(
        f"k={v['k']}, n={v['n']}: mean {v['mean']:.4f} (k/n {v['expected_k_over_n']:.4f}), 95th pct {v['p95']:.4f}"
        for v in rb.values()) + ".", ""]
    for view, title in (("whole_update", "whole update of the shortcut model"),
                        ("shortcut_minus_clean", "shortcut-specific update (shortcut model minus its clean twin, same seed)")):
        o += [f"Overlap per block, {title}:", "", "| block | " + " | ".join(c for c, _ in cols) + " |",
              "|---|" + "---|" * len(cols)]
        for b in range(12):
            o.append(f"| {b} | " + " | ".join(f"{cell(c, view, b)[0]:.3f} (×{cell(c, view, b)[1]:.0f})" for _, c in cols) + " |")
        late = {c: np.mean([cell(c, view, b)[0] for b in (9, 10, 11)]) for _, c in cols}
        early = {c: np.mean([cell(c, view, b)[0] for b in range(9)]) for _, c in cols}
        o += ["", "Mean over blocks 9-11 (where c-ii failed under LoRA) vs blocks 0-8: " + "; ".join(
            f"{name} {late[c]:.3f} vs {early[c]:.3f}" for name, c in cols) + ".", ""]

    def side_ratio(cond, view, b, side):
        k, i = ov["conditions"][cond]["k"], ("input", "output").index(side)
        return float(np.mean([v / (k / dims[m][i]) for m in dims
                              for v in ov["conditions"][cond]["overlap"][view][str(b)][m][side].values()]))

    def side_range(cond, side, view="whole_update"):
        r = [side_ratio(cond, view, b, side) for b in range(12)]
        return f"{min(r):.1f}–{max(r):.1f}×"

    o += ["Split by side (Sonu's review of #11; same numbers, no new computation), whole update, range over the 12 blocks "
          "of the multiple of the random level: " + "; ".join(
              f"{name}: input side {side_range(c, 'input')}, output side {side_range(c, 'output')}" for name, c in cols)
          + ". On the input side LoRA rank 64 is at about the random level and LoRA rank 4 far below its output side. "
          "That fits LoRA's A matrices (which set the input-side directions) starting from a random, seed-specific "
          "initialisation and moving little, while the output side is still clearly below full fine-tuning. So the "
          "seed-specific directions may come largely from LoRA's random start rather than from learning different "
          "features; either way it fits the c-ii failure.", ""]


# ---- Retraining (Phase 3). report.py's phase3_section runs unchanged on the same prefix-stripped copy (it reads the
# rank's own swap files too, for its swap-vs-retrain Spearman rule).
def retrain_flags(R):
    """Per-seed flags of every GPT-2 Phase 3 rule (as report.phase3_section): the gate, "removes" per block set,
    "keeps the shortcut" per control, and "swap and retrain agree" (Spearman). {} unless all 3 seeds are there."""
    rts = {s["seed"]: s for s in load_seed_files(R, "retrain")}
    out = {}
    if 0 in rts and "all" in rts[0]["runs"]:
        G0 = rts[0]["baseline"]["gap_none_minus_opposite"]
        out["Phase 3 gate (seed 0, retrain all blocks)"] = [bool(rts[0]["runs"]["all"]["gap_none_minus_opposite"]
                                                                <= REMOVE_MAX_FRAC * G0)]
    if sorted(rts) != list(SEEDS):
        return out
    p2s = {s["seed"]: s for s in load_swap(R)}
    for n in RETRAIN_ORDER + CONTROLS:
        fl = []
        for s in SEEDS:
            r, b = rts[s]["runs"].get(n), rts[s]["baseline"]
            G = b["gap_none_minus_opposite"]
            fl.append(None if r is None else bool(rule_keeps(r, G) if n in CONTROLS
                                                  else rule_removes(r, G, b["accuracy"]["none"])))
        out[f"Phase 3 {'keeps the shortcut (control)' if n in CONTROLS else 'removes'}: {n}"] = fl
    fl = []
    for s in SEEDS:
        singles = [str(b) for b in range(12)]
        rr, sr = [rts[s]["runs"].get(b) for b in singles], [swap_result(p2s.get(s), None, A, b) for b in singles]
        if None in rr or None in sr:
            fl.append(None)
            continue
        rho = spearman([frac_removed(r, shortcut_base(p2s[s])[0]) for r in sr],
                       [frac_removed(r, rts[s]["baseline"]["gap_none_minus_opposite"]) for r in rr])
        fl.append(bool(rho >= SPEARMAN_MIN))
    out["Phase 3 swap and retrain agree on where to fix it (Spearman)"] = fl
    return out


def retrain_fracs(R):
    """{block set: [fraction removed per seed]} for the seeds present (descriptive comparison table)."""
    rts = {s["seed"]: s for s in load_seed_files(R, "retrain")}
    return {n: [frac_removed(rts[s]["runs"][n], rts[s]["baseline"]["gap_none_minus_opposite"]) if n in rts[s]["runs"]
                else None for s in sorted(rts)] for n in RETRAIN_ORDER + CONTROLS}


def retrain_section(o, tmp_root):
    import tempfile
    done = [(label, prefix, R) for label, prefix, R in CONDITIONS[1:] if list((R / "retrain").glob("seed*.json"))]
    if not done:
        return
    flags = {"full fine-tuning": retrain_flags(RESULTS)}
    fracs = {"full fine-tuning": retrain_fracs(RESULTS)}
    rts = {"full fine-tuning": {s["seed"]: s for s in load_seed_files(RESULTS, "retrain")}}
    smaps = {"full fine-tuning": swap_map(RESULTS)}
    stopped = set()  # ranks whose Phase 3 gate failed: their other rules were not run
    o += ["## Retraining (Phase 3, Phase L2)", "",
          "Same block sets, controls, gate and reading rules as the GPT-2 Phase 3 (src/retrain.py unchanged, run with "
          "--prefix). Each run starts from the rank's merged shortcut model as it is (shortcut included), freezes "
          "everything except the chosen blocks, and keeps training those blocks' full weights for 1 epoch on clean data "
          "(controls: on the shortcut data), at the GPT-2 Phase 3 settings (configs/pilot.yaml). No new LoRA adapters, "
          "no reset. Tables and verdicts come from src/report.py's own Phase 3 section, run unchanged on a copy of each "
          "rank's files with the run-name prefix removed; its swap-vs-retrain Spearman uses that rank's own swap files. "
          "The gate (seed 0, 'all' and '9-10-11' first) is run by src/run_lora.py retrain, both ranks' gates before any "
          "other run.", ""]
    for label, prefix, R in done:
        sub = prefix.rstrip("_")
        tmp = normalized_copy(R, prefix, Path(tempfile.mkdtemp(dir=tmp_root)))
        flags[label], fracs[label] = retrain_flags(tmp), retrain_fracs(tmp)
        rts[label], smaps[label] = {s["seed"]: s for s in load_seed_files(tmp, "retrain")}, swap_map(tmp)
        fig_dir = FIG / sub
        fig_dir.mkdir(parents=True, exist_ok=True)
        part = []
        phase3_section(part, tmp, fig_dir)
        for line in part:
            line = (line.replace("results/retrain", f"results/{sub}/retrain").replace("results/swap", f"results/{sub}/swap")
                    .replace("Start = shortcut_seed<s>", f"Start = {sub}_shortcut_seed<s> (merged LoRA model, as it is)")
                    .replace("Rules pre-registered in PROJECT_SPEC.md on 15 Oct before any Phase 3 run.",
                             "Rules of the GPT-2 Phase 3 (pre-registered 15 Oct), applied to LoRA as pre-registered in "
                             "PROJECT_SPEC.md Phase L2 (20 Oct)."))
            o.append(f"### {label}: " + line[3:] if line.startswith("## ") else "#" + line if line.startswith("### ") else line)
        g, c = load(R / "retrain" / "gate.json"), load(R / "retrain" / "ceiling.json")
        if g and not g["pass"]:
            stopped.add(label)
            o += [f"**Phase 3 stopped at the gate for {label}** (results/{sub}/retrain/gate.json): no fixes tried.", ""]
        if c:
            o += [f"Phase 3b condition (PROJECT_SPEC.md Phase L2, results/{sub}/retrain/ceiling.json): every single block "
                  f"removes the shortcut in all seeds = {'yes' if c['ceiling'] else 'no'} → Phase 3b "
                  f"{'needed (separate step)' if c['ceiling'] else 'not run'}.", ""]
        env = load(R / "retrain" / "env.json")
        if env:
            o += [f"Environment (results/{sub}/retrain/env.json, one line per session): " + "; ".join(
                f"{e['gpu']}, torch {e['torch']}, transformers {e['transformers']}, peft {e['peft']}, commit "
                f"{str(e['git_commit'])[:7]}, {e['time']}" for e in env["sessions"]), ""]
        if (fig_dir / "retrain_map.png").exists():
            o += [f"![{label} retrain map](figures/lora/{sub}/retrain_map.png)", ""]
    labels = list(flags)
    o += ["### RQ 3: does each GPT-2 retraining rule reproduce under LoRA?", "",
          "Phase L2 rule: a rank reproduces a rule if it gets the same PASS/FAIL verdict as full fine-tuning (3 seeds "
          "each; the gate is seed 0 only). Per-seed P/F in seed order.", "",
          "| rule | " + " | ".join(labels) + " | " + " | ".join(f"{l}: reproduces" for l in labels[1:]) + " |",
          "|---|" + "---|" * (2 * len(labels) - 1)]
    for rule in flags["full fine-tuning"]:
        cells = [flags[l].get(rule) or [] for l in labels]
        verd = [f"{verdict_of(c)} ({'/'.join('P' if f else 'F' for f in c)})" if c and None not in c else verdict_of(c)
                for c in cells]
        rep = ["yes" if verdict_of(c) == verdict_of(cells[0]) and verdict_of(c) != "incomplete" else
               ("incomplete" if verdict_of(c) == "incomplete" else "NO") for c in cells[1:]]
        if not rule.startswith("Phase 3 gate"):
            for i, l in enumerate(labels[1:]):
                if l in stopped:
                    verd[i + 1], rep[i] = "not run (stopped at the gate)", "not run"
        o.append(f"| {rule} | " + " | ".join(verd) + " | " + " | ".join(rep) + " |")
    o += ["", "Fraction of the gap removed by retraining, side by side (descriptive; mean, per-seed values in brackets):", "",
          "| block(s) retrained | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for n in RETRAIN_ORDER + CONTROLS:
        row = []
        for l in labels:
            v = fracs[l].get(n) or []
            row.append("n/a" if not v or None in v else f"{np.mean(v):.3f} ({'/'.join(f'{x:.3f}' for x in v)})")
        o.append(f"| {n} | " + " | ".join(row) + " |")
    o.append("")
    retrain_notes(o, labels, flags, rts, smaps, stopped)


# ---- Phase 3b (only for a rank whose Phase 3 hit the ceiling). report.py's phase3b_section runs unchanged on the same
# prefix-stripped copy (it reads retrain_budget/, retrain/ and swap/).
def budget_flags(R):
    """Per-seed flags of the Phase 3b rule "location matters for fixing" (as report.phase3b_section)."""
    bus = {s["seed"]: s for s in load_seed_files(R, "retrain_budget")}
    if sorted(bus) != list(SEEDS):
        return None
    fl = []
    for s in SEEDS:
        G = bus[s]["baseline"]["gap_none_minus_opposite"]
        fr = [frac_removed(bus[s]["runs"][n]["evals"][BUDGET_RULE_STEP], G) if n in bus[s]["runs"] else None
              for n in BUDGET_LATE + BUDGET_EARLY]
        fl.append(None if None in fr else bool(np.mean(fr[:len(BUDGET_LATE)]) - np.mean(fr[len(BUDGET_LATE):])
                                               >= BUDGET_MIN_DIFF))
    return fl


def budget_section(o, tmp_root):
    import tempfile
    rows = {"full fine-tuning": budget_flags(RESULTS)}
    done = []
    for label, prefix, R in CONDITIONS[1:]:
        c = load(R / "retrain" / "ceiling.json")
        if c is None:
            continue
        rows[label] = "not run (no Phase 3 ceiling)" if not c["ceiling"] else None
        if c["ceiling"] and list((R / "retrain_budget").glob("seed*.json")):
            done.append((label, prefix, R))
    if len(rows) == 1:
        return
    o += ["## Retraining budget (Phase 3b, Phase L2)", "",
          "Phase L2: Phase 3b runs only for a rank whose Phase 3 removed the shortcut for every single block (ceiling; "
          "results/lora_r<r>/retrain/ceiling.json), with the Phase 3b rule unchanged. src/retrain.py --budget unchanged "
          "(smoke test of 1 run first, as in the Phase 3b spec). Tables and the verdict come from src/report.py's own "
          "Phase 3b section, run unchanged on the prefix-stripped copy.", ""]
    for label, prefix, R in done:
        sub = prefix.rstrip("_")
        tmp = normalized_copy(R, prefix, Path(tempfile.mkdtemp(dir=tmp_root)))
        rows[label] = budget_flags(tmp)
        fig_dir = FIG / sub
        part = []
        phase3b_section(part, tmp, fig_dir)
        for line in part:
            line = (line.replace("results/retrain", f"results/{sub}/retrain").replace("results/swap", f"results/{sub}/swap")
                    .replace("start = shortcut_seed<s>", f"start = {sub}_shortcut_seed<s>, merged LoRA model as it is")
                    .replace("Rules pre-registered in PROJECT_SPEC.md on 16 Oct before any Phase 3b run.",
                             "Rule of the GPT-2 Phase 3b (pre-registered 16 Oct), applied to LoRA as pre-registered in "
                             "PROJECT_SPEC.md Phase L2 (20 Oct)."))
            o.append(f"### {label}: " + line[3:] if line.startswith("## ") else "#" + line if line.startswith("### ") else line)
        env = load(R / "retrain_budget" / "env.json")
        if env:
            o += [f"Environment (results/{sub}/retrain_budget/env.json, one line per session): " + "; ".join(
                f"{e['gpu']}, torch {e['torch']}, transformers {e['transformers']}, peft {e['peft']}, commit "
                f"{str(e['git_commit'])[:7]}, {e['time']}" for e in env["sessions"]), ""]
        if (fig_dir / "retrain_budget.png").exists():
            o += [f"![{label} retrain budget](figures/lora/{sub}/retrain_budget.png)", ""]
    ref = rows["full fine-tuning"]
    o += ["### RQ 3: does the GPT-2 Phase 3b rule reproduce under LoRA?", "",
          "| condition | \"location matters for fixing\" | reproduces |", "|---|---|---|"]
    for label, fl in rows.items():
        if isinstance(fl, str):
            o.append(f"| {label} | {fl} | not applicable |")
            continue
        v = verdict_of(fl or [])
        cellv = f"{v} ({'/'.join('P' if f else 'F' for f in fl)})" if fl and None not in fl else v
        rep = "—" if label == "full fine-tuning" else ("incomplete" if v == "incomplete" else
                                                       "yes" if v == verdict_of(ref) else "NO")
        o.append(f"| {label} | {cellv} | {rep} |")
    o.append("")


def block_ranges(bs):
    """[0, 1, 2, 5] -> '0–2, 5'."""
    out, i = [], 0
    while i < len(bs):
        j = i
        while j + 1 < len(bs) and bs[j + 1] == bs[j] + 1:
            j += 1
        out.append(str(bs[i]) if i == j else f"{bs[i]}–{bs[j]}")
        i = j + 1
    return ", ".join(out)


def retrain_notes(o, labels, flags, rts, smaps, stopped):
    """Descriptive notes added after the retraining results were seen (no verdict depends on them)."""
    o += ["### Notes added after seeing the retraining results (descriptive; no verdict changes)", ""]
    rule = "Phase 3 swap and retrain agree on where to fix it (Spearman)"
    for l in labels:
        fl = flags[l].get(rule) or []
        if l in stopped or not fl or None in fl or all(fl):
            continue
        o.append(f"Swap-vs-retrain Spearman below {SPEARMAN_MIN} ({l}): what the two orders look like, per seed. The "
                 "cause on the swap side is the same as for the swap-map Spearman in the swap section: most single-block "
                 "swaps remove almost nothing, so their order is noise.")
        o.append("")
        for s in SEEDS:
            f, w = rts[l][s], smaps[l][s]
            G = f["baseline"]["gap_none_minus_opposite"]
            r = [frac_removed(f["runs"][str(b)], G) for b in range(12)]
            top, low = int(np.argmax(r)), int(np.argmin(r))
            near = [b for b in range(12) if abs(w[b]) < NEAR_ZERO]
            o.append(f"- seed {s} (Spearman {spearman(w, r):.3f}): retraining one block removes {min(r):.3f}–{max(r):.3f} "
                     f"of the gap across the 12 single blocks, most for block {top} ({r[top]:.3f}) and least for block {low} "
                     f"({r[low]:.3f}); the swap removes almost nothing (|fraction| < {NEAR_ZERO}) for blocks "
                     f"{block_ranges(near) or 'none'}, so the swap order of those blocks is noise.")
        o.append("")
    o += [f"Controls (retrained on the shortcut data): the rule \"keeps the shortcut\" needs the gap to stay ≥ "
          f"{KEEP_MIN_FRAC:.0%} of G_s, so a gap that grows also passes. Gap before → after, per seed:", "",
          "| condition | control | gap before → after (per seed) | change in gap, mean over seeds |", "|---|---|---|---|"]
    for l in labels:
        if l in stopped or sorted(rts[l]) != list(SEEDS):
            continue
        for n in CONTROLS:
            before = [rts[l][s]["baseline"]["gap_none_minus_opposite"] for s in SEEDS]
            after = [rts[l][s]["runs"][n]["gap_none_minus_opposite"] for s in SEEDS]
            ch = np.mean([(a - b) / b for a, b in zip(after, before)])
            o.append(f"| {l} | {n} | " + " / ".join(f"{b:.3f} → {a:.3f}" for b, a in zip(before, after))
                     + f" | {ch:+.0%} |")
    o.append("")
    complete = [l for l in labels if l not in stopped and sorted(rts[l]) == list(SEEDS)]
    borderline_note(o, complete, rts)
    shape_note(o, complete, rts)
    fit_note(o, complete, rts)


Z_CLEAR = 2.0  # descriptive only: a seed's "removes" failure is "clear" if it is >= 2 test-set standard errors below the line


def borderline_note(o, labels, rts):
    """Which "removes" failures are clear and which are within test-set noise (Sonu's review of #8)."""
    n = json.load(open(ROOT / "data" / "manifest.json"))["counts"]["test"]
    line = 1 - REMOVE_MAX_FRAC
    rows = []
    for l in labels:
        for name in RETRAIN_ORDER:
            cells, failed = [], False
            for s in SEEDS:
                b, r = rts[l][s]["baseline"], rts[l][s]["runs"][name]
                G, a = b["gap_none_minus_opposite"], r["accuracy"]
                fr = frac_removed(r, G)
                se = np.sqrt(a["none"] * (1 - a["none"]) / n + a["opposite"] * (1 - a["opposite"]) / n) / G
                z = (fr - line) / se
                if rule_removes(r, G, b["accuracy"]["none"]):
                    cells.append(f"{fr:.3f} pass")
                    continue
                failed = True
                kind = "accuracy cost" if fr >= line else ("clear" if z <= -Z_CLEAR else "borderline")
                cells.append(f"{fr:.3f} **{kind}** (z {z:+.1f})")
            if failed:
                rows.append(f"| {l} | {name} | " + " / ".join(cells) + " |")
    if not rows:
        return
    o += [f"Which \"removes\" failures are clear and which are within noise. z = (fraction removed − {line}) / its "
          f"standard error from the {n}-review test set alone (each accuracy treated as independent, so a rough upper "
          f"bound; differences between seeds are on top of this). A failure is \"clear\" if z ≤ −{Z_CLEAR}, else "
          "\"borderline\". Per seed, in seed order:", "",
          "| condition | block(s) | fraction removed per seed |", "|---|---|---|"] + rows + [""]


def shape_note(o, labels, rts):
    """Sonu's notes on #8/#9: the per-block shape of a Phase 3 run that did not hit the ceiling, next to the Phase 3b
    (step 100) curves, and where blocks 0 and 11 rank."""
    paths = {label: R for label, _, R in CONDITIONS}
    cols = []
    for l in labels:
        c = load(paths[l] / "retrain" / "ceiling.json") if l != "full fine-tuning" else None
        if c is not None and not c["ceiling"]:
            cols.append((f"{l}, Phase 3 (625 steps)", {n: [frac_removed(rts[l][s]["runs"][n],
                                                                        rts[l][s]["baseline"]["gap_none_minus_opposite"])
                                                           for s in SEEDS] for n in BUDGET_SINGLES}))
    if not cols:
        return
    for label, _, R in CONDITIONS:
        bus = {s["seed"]: s for s in load_seed_files(R, "retrain_budget")}
        if sorted(bus) == list(SEEDS):
            cols.append((f"{label}, Phase 3b step {BUDGET_RULE_STEP}", {
                n: [frac_removed(bus[s]["runs"][n]["evals"][BUDGET_RULE_STEP], bus[s]["baseline"]["gap_none_minus_opposite"])
                    for s in SEEDS] for n in BUDGET_SINGLES}))
    means = [{n: float(np.mean(v)) for n, v in c.items()} for _, c in cols]
    rank = lambda m, b: 1 + sum(m[x] > m[b] for x in BUDGET_SINGLES)  # noqa: E731
    o += [f"Shape across single blocks (mean fraction removed over seeds): a Phase 3 run without the ceiling next to the "
          f"Phase 3b curves at step {BUDGET_RULE_STEP} (the {len(BUDGET_SINGLES)} single blocks Phase 3b has).", "",
          "| block | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for b in BUDGET_SINGLES:
        o.append(f"| {b} | " + " | ".join(f"{m[b]:.3f}" for m in means) + " |")
    o += ["", "Rank among these blocks (1 = removes most): " + "; ".join(
        f"{c}: block 0 #{rank(m, '0')}, block 11 #{rank(m, '11')}" for (c, _), m in zip(cols, means)) + ". Spearman of "
        f"each column with the first: " + ", ".join(
            f"{c} {spearman([means[0][b] for b in BUDGET_SINGLES], [m[b] for b in BUDGET_SINGLES]):.3f}"
            for (c, _), m in zip(cols[1:], means[1:])) + ".", ""]


def fit_note(o, labels, rts):
    """Open question (hypothesis after the results): how well each condition's start models fit their training data,
    next to the training loss left after one-block retraining."""
    paths = {label: (prefix, R) for label, prefix, R in CONDITIONS}
    o += ["Open question (a hypothesis after seeing the results; nothing here is tested): how well the start models fit "
          "their training data. Train loss = mean over the last 50 steps, per seed.", "",
          "| condition | end of the original training | after Phase 3, mean over the 12 single-block runs |", "|---|---|---|"]
    for l in labels:
        prefix, R = paths[l]
        orig = [load(R / "checks" / f"{prefix}shortcut_seed{s}.json")["train"]["final_loss_mean_last50"] for s in SEEDS]
        after = [np.mean([rts[l][s]["runs"][str(b)]["train"]["final_loss_mean_last50"] for b in range(12)]) for s in SEEDS]
        o.append(f"| {l} | {' / '.join(f'{x:.3f}' for x in orig)} | {' / '.join(f'{x:.3f}' for x in after)} |")
    o += ["", "One possible reading: where the start model already fits the retraining reviews (the same 10,000 it was "
          "trained on), little is left to learn except that the names mean nothing, so retraining almost any block "
          "removes the shortcut; where it fits them less well, one block's retraining also has general task error to fix, "
          "and only some blocks remove the shortcut within an epoch. A weaker shortcut is a second candidate.", ""]
    pilot = [load(RESULTS / "lora_pilot" / f"{n}.json") for n in ("lora_r4_lr1e-3_shortcut_seed0", "lora_r4_lr5e-4_shortcut_seed0")]
    if all(pilot):
        o += ["A possible check, not run (it would need its own pre-registration): Phase 3 single blocks on the seed-0 "
              "pilot model " + " vs ".join(f"{p['run']} (gap {p['gap_none_minus_opposite']:.3f}, train loss "
                                           f"{p['train']['final_loss_mean_last50']:.3f})" for p in pilot)
              + ", one seed only and a different lr."]
    o.append("")


# ---- Phase L3: weak fix (PROJECT_SPEC.md "Phase L3"). Rules applied exactly as pre-registered; read at step B with the
# floor/ceiling fallbacks; a block whose acc(none) drops by more than 0.02 is ranked last; ties count against a block.
L3_TOP_K = 3
L3_FLOOR, L3_CEIL = 0.10, 0.90
L3_LABELS = {"full_ft": "full fine-tuning", "lora_r4": "LoRA rank 4", "lora_r64": "LoRA rank 64"}


def l3_values(res, step):
    """Per single block at a step: fraction removed, and the value used for ranking (blocks with an accuracy cost go
    last, below every block without one)."""
    G, none0 = res["baseline"]["gap_none_minus_opposite"], res["baseline"]["accuracy"]["none"]
    fr = [frac_removed(res["runs"][str(b)]["evals"][str(step)], G) for b in range(12)]
    ok = [none0 - res["runs"][str(b)]["evals"][str(step)]["accuracy"]["none"] <= REMOVE_MAX_NONE_DROP for b in range(12)]
    low = min(fr) - 1.0
    return fr, [f if k else low for f, k in zip(fr, ok)], ok


def l3_rank(vals, b):
    """1-based rank of block b, ties counted against it (= number of blocks at least as good)."""
    return sum(v >= vals[b] for v in vals)


def l3_read_step(files, B):
    """The step the rules are read at (PROJECT_SPEC.md Phase L3): B; on a floor -> 2B, then 625; on a ceiling -> B/2.
    Returns (step, note); step None = "not informative"."""
    steps = [int(x) for x in files[0]["eval_steps"]]

    def state(st):
        fl = [l3_values(f, st)[0] for f in files]
        if any(max(x) < L3_FLOOR for x in fl):
            return "floor"
        if any(min(x) >= L3_CEIL for x in fl):
            return "ceiling"
        return "ok"

    s0 = state(B)
    if s0 == "ok":
        return B, f"read at B = {B}"
    tried = [B]
    options = ([2 * B] if 2 * B in steps and 2 * B != B else []) + ([625] if 625 not in (B, 2 * B) else []) \
        if s0 == "floor" else ([B // 2] if B // 2 in steps else [])
    for st in options:
        tried.append(st)
        if state(st) == "ok":
            return st, f"{s0} at B = {B}; read at step {st} (pre-registered fallback)"
    return None, f"{s0} at every pre-registered option ({', '.join(map(str, tried))}): not informative"


def l3_rules(files, step, loc, top3, swap_fr):
    """Per-seed flags of Rules 1, 1b, 2, 3 and the control, at one step."""
    out = {"Rule 1: localized block in the top 3": [], "Rule 1b: best block among the top-3 swap blocks": [],
           "Rule 2: Spearman(swap, weak fix) >= 0.6": [], "Rule 3: block 0 in the top 3": [],
           "Control: shortcut-data fix keeps >= 80% of the gap": []}
    for f in files:
        s = f["seed"]
        fr, val, _ = l3_values(f, step)
        best = [b for b in range(12) if val[b] == max(val)]
        out["Rule 1: localized block in the top 3"].append(l3_rank(val, loc) <= L3_TOP_K)
        out["Rule 1b: best block among the top-3 swap blocks"].append(all(b in top3 for b in best))
        out["Rule 2: Spearman(swap, weak fix) >= 0.6"].append(spearman(swap_fr[s], val) >= SPEARMAN_MIN)
        out["Rule 3: block 0 in the top 3"].append(l3_rank(val, 0) <= L3_TOP_K)
        G = f["baseline"]["gap_none_minus_opposite"]
        c = f["runs"][f"ctrl_shortcut:{loc}"]["evals"][str(step)]
        out["Control: shortcut-data fix keeps >= 80% of the gap"].append(c["gap_none_minus_opposite"] >= KEEP_MIN_FRAC * G)
    return out


def swap_fracs(R, prefix):
    """{seed: [fraction removed by the Phase 2 swap, case a, for each single block]} from a condition's swap files."""
    out = {}
    for s in SEEDS:
        f = json.load(open(R / "swap" / f"seed{s}.json"))
        G = f["baseline"][f"{prefix}shortcut_seed{s}"]["gap_none_minus_opposite"]
        out[s] = [frac_removed(f["cases"][A]["by_blocks"][str(b)], G) for b in range(12)]
    return out


def lens_block(R, prefix):
    """Largest mean per-block drop on 'opposite' (calibrated lens), shortcut models (reported, no rule)."""
    ls = [json.load(open(R / "logit_lens" / f"{prefix}shortcut_seed{s}.json"))["cal_acc"]["opposite"] for s in SEEDS]
    return int(np.argmax(np.mean([[l[b] - l[b + 1] for b in range(12)] for l in ls], 0)))


def weakfix_section(o):
    from weakfix import LOCALIZED, TOP3_SWAP
    W = RESULTS / "weakfix"
    conds = [c for c in L3_LABELS if all((W / c / f"seed{s}.json").exists() for s in SEEDS)]
    if not conds:
        return
    paths = {"full_ft": ("", RESULTS), "lora_r4": ("lora_r4_", RESULTS / "lora_r4"), "lora_r64": ("lora_r64_", RESULTS / "lora_r64")}
    o += ["## Weak fix (Phase L3)", "",
          "PROJECT_SPEC.md Phase L3 (pre-registered 21 Oct, merged in #10 before any L3 run). A new rank-4 LoRA adapter "
          "(49,152 parameters) on ONE block of each shortcut model as it is, trained on clean data (lr 5e-4, 1-epoch "
          "schedule), via src/weakfix.py (src/train.py's train() unchanged; every other weight asserted bit-identical). "
          "Step budget B per condition from seed 0's fix on all 12 blocks (first step removing >= 50%); rules read at B, "
          "with the pre-registered floor/ceiling fallbacks. Fraction removed = (G_s − gap) / G_s. A block whose acc(none) "
          "drops by more than 0.02 is ranked last; ties count against a block. PASS only if all 3 seeds pass. Phase 3b r64 "
          "results were available when Phase L3 was committed; the localized block comes from swap results only.", ""]
    # start-model checks and budgets
    o += ["| condition | start-model checks | B (all-12-block fix, seed 0) | fraction removed by the all-block fix, by step |",
          "|---|---|---|---|"]
    for c in conds:
        sc, bud = load(W / c / "start_checks.json"), load(W / c / "budget.json")
        o.append(f"| {L3_LABELS[c]} | {'all pass' if sc['all_pass'] else 'FAIL'} | {bud['B']}"
                 + ("" if bud["reached_50pct"] else " (never reached 50%)") + " | "
                 + ", ".join(f"{k}: {v:.3f}" for k, v in bud["fraction_removed_by_step"].items()) + " |")
    o.append("")
    verdicts, best_blocks, lens_ranks = {}, {}, {}
    for c in conds:
        prefix, R = paths[c]
        files = [load(W / c / f"seed{s}.json") for s in SEEDS]
        B, loc, top3 = files[0]["B"], LOCALIZED[c], TOP3_SWAP[c]
        sw = swap_fracs(R, prefix)
        step, note = l3_read_step(files, B)
        o += [f"### {L3_LABELS[c]}: weak fix", "",
              f"Localized block {loc} (Phase 2 swap, case a); top-3 swap blocks {top3}; lens block {lens_block(R, prefix)} "
              f"(reported, no rule). Eval steps {files[0]['eval_steps']}. Reading: {note}.", ""]
        steps = [int(x) for x in files[0]["eval_steps"]]
        o += ["Fraction of the gap removed per single block (mean; per-seed values in brackets; * = acc(none) dropped by "
              "more than 0.02 in that seed):", "", "| block | " + " | ".join(f"step {st}" for st in steps) + " |",
              "|---|" + "---|" * len(steps)]
        for b in range(12):
            row = []
            for st in steps:
                v = [(l3_values(f, st)[0][b], l3_values(f, st)[2][b]) for f in files]
                row.append(f"{np.mean([x for x, _ in v]):.3f} (" + "/".join(f"{x:.3f}" + ("" if k else "*") for x, k in v) + ")")
            tag = " (localized)" if b == loc else (" (top-3 swap)" if b in top3 else "")
            o.append(f"| {b}{tag} | " + " | ".join(row) + " |")
        ctrl = f"ctrl_shortcut:{loc}"
        o.append(f"| control: block {loc} on shortcut data | " + " | ".join(
            f"{np.mean([frac_removed(f['runs'][ctrl]['evals'][str(st)], f['baseline']['gap_none_minus_opposite']) for f in files]):.3f}"
            for st in steps) + " |")
        o.append("")
        o += ["Rules per evaluated step (the verdict counts only at the reading step; the others are reported for "
              "stability):", "", "| rule | " + " | ".join(f"step {st}" + (" (reading)" if st == step else "") for st in steps) + " |",
              "|---|" + "---|" * len(steps)]
        per_step = {st: l3_rules(files, st, loc, top3, sw) for st in steps}
        for rule in per_step[steps[0]]:
            o.append(f"| {rule} | " + " | ".join(f"{verdict_of(per_step[st][rule])} ({'/'.join('P' if x else 'F' for x in per_step[st][rule])})"
                                                for st in steps) + " |")
        o.append("")
        if step is None:
            verdicts[c] = {r: "not informative" for r in per_step[steps[0]]}
        else:
            verdicts[c] = {r: f"{verdict_of(v)} ({'/'.join('P' if x else 'F' for x in v)})" for r, v in per_step[step].items()}
            best_blocks[c] = [sorted(range(12), key=lambda b: -l3_values(f, step)[1][b])[0] for f in files]
            ranks = {b: [l3_rank(l3_values(f, step)[1], b) for f in files] for b in (loc, 0)}
            lb = lens_block(R, prefix)
            lens_ranks[c] = (lb, [l3_rank(l3_values(f, step)[1], lb) for f in files])
            o += [f"At the reading step {step}: best single block per seed {best_blocks[c]}; rank of the localized block "
                  f"{loc}: {'/'.join(map(str, ranks[loc]))}; rank of block 0: {'/'.join(map(str, ranks[0]))}; Spearman with "
                  "the swap map per seed: " + "/".join(f"{spearman(sw[f['seed']], l3_values(f, step)[1]):.3f}" for f in files) + ".", ""]
        runs = [r for f in files for r in f["runs"].values()]
        commits = sorted({str(r.get("git_commit"))[:7] for r in runs})
        o += [f"Cost: {len(runs)} runs, train {np.mean([r['train_time_s_excl_eval'] for r in runs]):.0f} s and eval "
              f"{np.mean([r['eval_time_s'] for r in runs]):.0f} s per run (mean), peak VRAM allocated "
              f"{max(r['train']['peak_vram_allocated_gib'] for r in runs):.2f} GiB; GPU "
              f"{sorted({r['env']['gpu'] for r in runs if r.get('env')})}; code commit(s) {commits}.", ""]
    rules = list(next(iter(verdicts.values())))
    o += ["### Phase L3 verdicts by condition", "",
          "| rule | " + " | ".join(L3_LABELS[c] for c in verdicts) + " |", "|---|" + "---|" * len(verdicts)]
    for r in rules:
        o.append(f"| {r} | " + " | ".join(verdicts[c][r] for c in verdicts) + " |")
    if best_blocks:
        o += ["", "Does the best weak-fix block move with the localized block (reported, no verdict)? " + "; ".join(
            f"{L3_LABELS[c]}: localized {LOCALIZED[c]}, best per seed {best_blocks[c]}" for c in best_blocks) + "."]
    o += ["", "Caveats (pre-registered): late blocks might be easier to fix in general (the cross-condition comparison "
          "above is the only check, and it is descriptive); under LoRA the localized block comes from swaps whose own "
          "control (c-ii) failed for late blocks; for rank 4, Phase 3 already showed blocks 9-10 retrain best, so that "
          "condition is not fully blind.", ""]
    if lens_ranks:
        o += ["Observation spotted after the Phase L3 results (descriptive only; Sonu, 23 Oct: a \"pre-registered\" test on "
              "the same data would not mean anything, so it is reported as an observation and an open question): the rank "
              "of the logit-lens block (largest mean per-block drop on 'opposite') among the 12 weak fixes at the reading "
              "step, per seed: " + "; ".join(f"{L3_LABELS[c]}: lens block {lb} ranks {'/'.join(map(str, r))}"
                                              + (f" (the swap-localized block is {LOCALIZED[c]})" if lb != LOCALIZED[c] else
                                                 " (same block as the swap localization)")
                                              for c, (lb, r) in lens_ranks.items()) + ".", ""]


def l3b_section(o):
    """Phase L3b (descriptive): Phase 3 single blocks on the rank-4 pilot model with a stronger shortcut."""
    p = RESULTS / "lora_r4_lr1em3" / "retrain" / "seed0.json"
    if not p.exists():
        return
    res = load(p)
    chk = load(RESULTS / "lora_r4_lr1em3" / "start_check.json")
    pilot = load(RESULTS / "lora_pilot" / "lora_r4_lr1e-3_shortcut_seed0.json")

    def fr_list(f):
        G = f["baseline"]["gap_none_minus_opposite"]
        return [frac_removed(f["runs"][str(b)], G) for b in range(12)]

    def n_removes(f):
        G, n0 = f["baseline"]["gap_none_minus_opposite"], f["baseline"]["accuracy"]["none"]
        return sum(bool(rule_removes(f["runs"][str(b)], G, n0)) for b in range(12))

    cols = [("LoRA rank 4, lr 1e-3 (L3b)", res)]
    for label, R in (("LoRA rank 4 (main)", RESULTS / "lora_r4"), ("full fine-tuning", RESULTS),
                     ("LoRA rank 64", RESULTS / "lora_r64")):
        cols.append((label, load(R / "retrain" / "seed0.json")))
    o += ["## Phase L3b: rank 4 with a stronger shortcut (descriptive)", "",
          "PROJECT_SPEC.md Phase L3b (pre-registered 23 Oct, before any L3b run). Phase 3 single blocks 0-11 (src/retrain.py "
          "unchanged, Phase 3 settings, seed 0) on the seed-0 pilot model lora_r4_lr1e-3_shortcut_seed0 (rank 4 trained at "
          "lr 1e-3; used as lora_r4_lr1em3_shortcut_seed0), next to seed 0 of the main conditions (Phase 3). One seed, a "
          "different lr, and a model in between in shortcut strength and fit: it can suggest, not prove. No verdict.", "",
          f"Start-model check: accuracies equal the pilot results file: {chk['pass'] if chk else 'n/a'}. Start gaps (seed 0): "
          + "; ".join(f"{label} {f['baseline']['gap_none_minus_opposite']:.3f}" for label, f in cols)
          + (f". Train loss at the end of the L3b model's own training: {pilot['train']['final_loss_mean_last50']:.3f} "
             "(the main conditions' values are in the open-question table above)" if pilot else "") + ".", "",
          "| block | " + " | ".join(label for label, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    frs = [fr_list(f) for _, f in cols]
    for b in range(12):
        o.append(f"| {b} | " + " | ".join(f"{x[b]:.3f}" for x in frs) + " |")
    o += ["", "| measure (pre-registered, descriptive) | " + " | ".join(label for label, _ in cols) + " |",
          "|---|" + "---|" * len(cols),
          "| single blocks passing \"removes\" | " + " | ".join(f"{n_removes(f)}/12" for _, f in cols) + " |",
          "| mean fraction removed, blocks 1-5 minus blocks 9-10 | " + " | ".join(
              f"{np.mean(x[1:6]) - np.mean(x[9:11]):+.3f}" for x in frs) + " |", "",
          "Reading (pre-registered): if blocks 1-5 come out like full FT / r64 (close to blocks 9-10), it suggests shortcut "
          "strength / fit; if like r4 seed 0, it suggests rank.", ""]


def future_work(o):
    """Open questions and future work from the LoRA track (not run; decided with Sonu, 23 Oct)."""
    o += ["## Open questions and future work (not run)", "",
          "- **A valid swap control for LoRA models.** The cross-seed control (c-ii: blocks from another seed's shortcut "
          "model must keep the shortcut) assumes that different seeds solve the task in compatible ways. That holds for "
          "full fine-tuning but not for LoRA: c-ii fails for late blocks at both ranks, and the adapter-overlap check shows "
          "LoRA seeds change the weights in largely different directions (consistent with Zhu et al., ICML 2024, and "
          "KnOTS, Stoica et al., ICLR 2025). So swap-based \"real removal, not breakage\" claims need a different control "
          "on LoRA models. Candidate controls, none tested here: (1) shared-initialisation donors, i.e. a second shortcut "
          "model trained from the same LoRA starting point with a different data order, so its blocks fit the recipient; "
          "(2) aligned swaps, rotating the donor's adapter into the recipient's directions before swapping (KnOTS-style); "
          "(3) a breakage control that needs no second model, replacing the block's weight change with random noise of "
          "the same size. A planted shortcut gives ground truth for checking which control behaves correctly. As far as "
          "a quick literature search shows, cross-model interpretability controls on LoRA models have not been studied "
          "directly. A new method, so future work (Sonu, 23 Oct); a possible follow-up after the course.",
          "- **Shared-initialisation LoRA seeds.** Training LoRA seeds from the same adapter initialisation (different data "
          "order) would test whether the seed-specific directions come from LoRA's random A start. Not run: it mostly "
          "confirms known work (Zhu et al., ICML 2024: LoRA's A barely moves from its random start) (Sonu, 23 Oct).",
          "- **Logit lens vs swaps as a guide to cheap fixes under LoRA.** See the observation after the Phase L3 verdicts; "
          "spotted after the results, so it needs new data to test.",
          "- **Why rank 4 behaves differently** (fit / shortcut strength vs rank): Phase L3b gives a one-seed descriptive "
          "hint; a proper answer would need several seeds at matched shortcut strength.", ""]



def main():
    o = ["# RESULTS_LORA (LoRA on GPT-2)", "",
         "_Generated by `src/report_lora.py` from results files only. Do not edit by hand._", "",
         "## Notes", ""] + [f"- {n}" for n in NOTES] + [""]
    pilot_section(o)
    checks_section(o)
    lens_section(o)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_root:
        swaps_section(o, Path(tmp_root))
        retrain_section(o, Path(tmp_root))
        budget_section(o, Path(tmp_root))
    weakfix_section(o)
    l3b_section(o)
    future_work(o)
    OUT.write_text("\n".join(o), encoding="utf-8")
    print(f"wrote {OUT}" + (f" and {FIG}/" if FIG.exists() else ""))


if __name__ == "__main__":
    main()
