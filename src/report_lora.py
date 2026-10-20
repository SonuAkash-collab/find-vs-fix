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
    "Retraining (Phase 3) on LoRA models retrains the chosen blocks' full weights, as for GPT-2: a strong fix.",
    "Phase 2b 'emb_names' (swap only the name-token embedding rows) is a no-op by construction under LoRA: LoRA does not "
    "train the embeddings, so every LoRA model has the same name rows. Reported as not applicable, not as a finding.",
]


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


def main():
    o = ["# RESULTS_LORA (LoRA on GPT-2)", "",
         "_Generated by `src/report_lora.py` from results files only. Do not edit by hand._", "",
         "## Notes", ""] + [f"- {n}" for n in NOTES] + [""]
    pilot_section(o)
    checks_section(o)
    lens_section(o)
    OUT.write_text("\n".join(o), encoding="utf-8")
    print(f"wrote {OUT}" + (f" and {FIG}/" if FIG.exists() else ""))


if __name__ == "__main__":
    main()
