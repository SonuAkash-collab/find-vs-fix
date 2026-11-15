"""Paper figures (vector PDF) from committed result files only.

Post-processing only: no training, no model runs. Reads results/*.json and writes figures/paper/*.pdf.
IEEE single column: 3.45 in wide, serif 8 pt, no titles (captions live in the LaTeX).
Colour-blind-safe: blue #2a78d6, orange #eb6834, grey for anything else. Seeds 0/1/2 = solid/dashed/dotted.
A figure whose input files are missing is skipped with a logged reason; no value is ever filled in.

    python src/make_paper_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "figures" / "paper"

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#7f7f7f"
SEEDS = [0, 1, 2]
SEED_LS = {0: "-", 1: "--", 2: ":"}
WIDTH = 3.45

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handlelength": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType, no Type 3
    "savefig.bbox": "standard",  # keep the exact 3.45 in width; layout() leaves room for the legends
})

LOG = []


def log(msg):
    LOG.append(msg)
    print(msg)


def load(rel):
    """Committed JSON under results/, or None if the file is missing."""
    p = R / rel
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def need(name, rels):
    """Load every file; if any is missing, log the skip and return None."""
    got = {rel: load(rel) for rel in rels}
    miss = [rel for rel, v in got.items() if v is None]
    if miss:
        log(f"SKIP {name}: missing {', '.join('results/' + m for m in miss)}")
        return None
    return got


def frac_removed(G, gap):
    return (G - gap) / G


def top_legend(fig, colour_handles, height):
    """Two centred legend rows above the axes: the colours, then the seed line styles.
    Call after layout() has left room at the top."""
    row = LEGEND_ROW_IN / height
    seeds = [Line2D([], [], color="black", ls=SEED_LS[s], lw=0.9, label=f"seed {s}") for s in SEEDS]
    kw = dict(loc="upper center", frameon=False, borderpad=0.0, borderaxespad=0.0, columnspacing=1.2,
              handletextpad=0.5)
    fig.legend(handles=list(colour_handles), ncol=len(colour_handles), bbox_to_anchor=(0.5, 1.0 - 0.15 * row), **kw)
    fig.legend(handles=seeds, ncol=3, bbox_to_anchor=(0.5, 1.0 - 1.15 * row), **kw)


LEGEND_ROW_IN = 0.15  # height of one legend row, inches


def layout(fig, height, **kw):
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.75 * LEGEND_ROW_IN / height), **kw)


def save(fig, name, sources):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.pdf"
    fig.savefig(p, metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    log(f"WROTE figures/paper/{name}.pdf  <- {', '.join('results/' + s for s in sources)}")


# ---- (a) full-FT calibrated logit lens, "none" and "opposite", shortcut vs clean
def fig_lens_fullft():
    rels = [f"logit_lens/{k}_seed{s}.json" for k in ("shortcut", "clean") for s in SEEDS]
    d = need("fig_a_lens_fullft", rels)
    if d is None:
        return
    H = 2.0
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, H), sharey=True)
    for ax, ver in zip(axes, ("none", "opposite")):
        for kind, col in (("shortcut", BLUE), ("clean", ORANGE)):
            for s in SEEDS:
                L = d[f"logit_lens/{kind}_seed{s}.json"]
                ax.plot(L["layers"], L["cal_acc"][ver], color=col, ls=SEED_LS[s])
        ax.axhline(0.5, color=GREY, lw=0.5, ls=(0, (1, 2)))
        ax.set_xlabel("layer")
        ax.set_xticks(range(0, 13, 2))
        ax.set_xlim(0, 12)
        ax.text(0.03, 0.97, f"“{ver}”", transform=ax.transAxes, va="top", ha="left")
    axes[0].set_ylabel("calibrated accuracy")
    axes[0].set_ylim(0, 1)
    layout(fig, H, w_pad=0.6)
    top_legend(fig, [Line2D([], [], color=BLUE, label="shortcut model"),
                     Line2D([], [], color=ORANGE, label="clean twin")], H)
    save(fig, "fig_a_lens_fullft", rels)


# ---- (b) swap map, case (a) and the c-ii control, single blocks
CASE_A, CASE_CII = "a_shortcut_from_clean_same", "cii_shortcut_from_shortcut_other"


def swap_fracs(sw, case, blocks):
    G = sw["baseline"][f"shortcut_seed{sw['seed']}"]["gap_none_minus_opposite"]
    return [frac_removed(G, sw["cases"][case]["by_blocks"][b]["gap_none_minus_opposite"]) for b in blocks]


def fig_swap_map():
    rels = [f"swap/seed{s}.json" for s in SEEDS]
    d = need("fig_b_swap_map", rels)
    if d is None:
        return
    blocks = [str(b) for b in range(12)]
    H = 2.1
    fig, ax = plt.subplots(figsize=(WIDTH, H))
    for case, col in ((CASE_A, BLUE), (CASE_CII, ORANGE)):
        for s in SEEDS:
            ax.plot(range(12), swap_fracs(d[f"swap/seed{s}.json"], case, blocks), color=col, ls=SEED_LS[s],
                    marker="o", ms=1.8)
    ax.axhline(0, color=GREY, lw=0.5)
    ax.axhline(0.5, color=GREY, lw=0.5, ls=(0, (4, 2)))
    ax.set_xticks(range(12))
    ax.set_xlim(-0.3, 11.3)
    ax.set_xlabel("swapped block")
    ax.set_ylabel("fraction of gap removed")
    layout(fig, H)
    top_legend(fig, [Line2D([], [], color=BLUE, marker="o", ms=1.8, label="(a) clean twin, same seed"),
                     Line2D([], [], color=ORANGE, marker="o", ms=1.8, label="(c-ii) shortcut, other seed")], H)
    save(fig, "fig_b_swap_map", rels)


# ---- (c) retrain map: full FT vs LoRA r4 vs r64, single blocks
COND = [("full FT", "", BLUE), ("LoRA $r{=}4$", "lora_r4/", ORANGE), ("LoRA $r{=}64$", "lora_r64/", GREY)]


def fig_retrain_map():
    rels = [f"{pre}retrain/seed{s}.json" for _, pre, _ in COND for s in SEEDS]
    d = need("fig_c_retrain_map", rels)
    if d is None:
        return
    H = 2.1
    fig, ax = plt.subplots(figsize=(WIDTH, H))
    for lab, pre, col in COND:
        for s in SEEDS:
            rt = d[f"{pre}retrain/seed{s}.json"]
            G = rt["baseline"]["gap_none_minus_opposite"]
            ax.plot(range(12), [frac_removed(G, rt["runs"][str(b)]["gap_none_minus_opposite"]) for b in range(12)],
                    color=col, ls=SEED_LS[s], marker="o", ms=1.8)
    ax.axhline(0.5, color=GREY, lw=0.5, ls=(0, (4, 2)))
    ax.set_xticks(range(12))
    ax.set_xlim(-0.3, 11.3)
    ax.set_xlabel("retrained block")
    ax.set_ylabel("fraction of gap removed")
    layout(fig, H)
    top_legend(fig, [Line2D([], [], color=c, marker="o", ms=1.8, label=lab) for lab, _, c in COND], H)
    save(fig, "fig_c_retrain_map", rels)


# ---- (d) Phase 4 patching: F(name), F(last), F(rest) per layer, shortcut models, all reviews
# "other positions" = Phase 4's position set "rest": positions after the name, except the last token
# (positions before the name are identical in versions A and B).
PATCH_SETS = [("name", "name tokens", BLUE), ("last", "last token", ORANGE), ("rest", "other positions", GREY)]


def fig_patching():
    rels = [f"patching/seed{s}.json" for s in SEEDS]
    d = need("fig_d_patching", rels)
    if d is None:
        return
    H = 2.1
    fig, ax = plt.subplots(figsize=(WIDTH, H))
    for P, _, col in PATCH_SETS:
        for s in SEEDS:
            m = d[f"patching/seed{s}.json"]["models"]["shortcut"]
            ax.plot(m["layers"], m["curves"]["all"][P]["F"], color=col, ls=SEED_LS[s])
    ax.axhline(0.5, color=GREY, lw=0.5, ls=(0, (4, 2)))
    ax.set_xticks(range(13))
    ax.set_xlim(0, 12)
    ax.set_xlabel("layer $L$ (residual stream entering block $L$)")
    ax.set_ylabel("$F$ (fraction of name effect)")
    layout(fig, H)
    top_legend(fig, [Line2D([], [], color=c, label=lab) for _, lab, c in PATCH_SETS], H)
    save(fig, "fig_d_patching", rels)


# ---- (e) LoRA lens: calibrated "opposite" accuracy per layer, full FT vs r4 vs r64 (shortcut models)
def fig_lens_lora():
    rels = [f"logit_lens/shortcut_seed{s}.json" for s in SEEDS] + \
           [f"lora_r{r}/logit_lens/lora_r{r}_shortcut_seed{s}.json" for r in (4, 64) for s in SEEDS]
    d = need("fig_e_lens_lora", rels)
    if d is None:
        return
    files = {"full FT": [rels[s] for s in SEEDS],
             "LoRA $r{=}4$": rels[3:6], "LoRA $r{=}64$": rels[6:9]}
    H = 2.1
    fig, ax = plt.subplots(figsize=(WIDTH, H))
    for lab, _, col in COND:
        for s, rel in zip(SEEDS, files[lab]):
            L = d[rel]
            ax.plot(L["layers"], L["cal_acc"]["opposite"], color=col, ls=SEED_LS[s])
    ax.axhline(0.5, color=GREY, lw=0.5, ls=(0, (1, 2)))
    ax.set_xticks(range(13))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.set_xlabel("layer")
    ax.set_ylabel("calibrated accuracy, “opposite”")
    layout(fig, H)
    top_legend(fig, [Line2D([], [], color=c, label=lab) for lab, _, c in COND], H)
    save(fig, "fig_e_lens_lora", rels)


def fig_wronglabel():
    name = "fig_f_wronglabel"
    checks = [f"wronglabel/checks/wl_noisy_seed{s}.json" for s in SEEDS]
    swaps = [f"wronglabel/swap/seed{s}.json" for s in SEEDS]
    retrains = [f"wronglabel/retrain/seed{s}.json" for s in SEEDS]
    sources = checks + swaps + retrains
    data = need(name, sources)
    if data is None:
        return
    metric = "M_flipped_vs_wrong_label"
    blocks = range(12)
    height = 2.15
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    for seed in SEEDS:
        baseline = data[checks[seed]]["metrics"][metric]
        for entries, color in ((data[retrains[seed]]["runs"], BLUE),
                               (data[swaps[seed]]["cases"]["a"], ORANGE)):
            values = [frac_removed(baseline, entries[str(b)]["metrics"][metric]) for b in blocks]
            ax.plot(blocks, values, color=color, ls=SEED_LS[seed])
    ax.axhline(0.5, color=GREY, lw=0.6, alpha=0.7)
    ax.set(xlabel="Transformer block", ylabel="Fraction of memorization removed", xticks=list(blocks))
    layout(fig, height)
    top_legend(fig, [Line2D([], [], color=BLUE, label="Retraining"),
                     Line2D([], [], color=ORANGE, label="Clean-twin swap")], height)
    save(fig, name, sources)


# ---- (g) Phase 3c: full-FT single-block retraining with vs without the name sentence
def fig_nonames():
    name = "fig_g_nonames"
    with_n = [f"retrain_l4/seed{s}.json" for s in SEEDS]
    without = [f"retrain_nonames/seed{s}.json" for s in SEEDS]
    data = need(name, with_n + without)
    if data is None:
        return
    blocks = range(12)
    height = 2.0
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    for s in SEEDS:
        for rel, color in ((with_n[s], BLUE), (without[s], ORANGE)):
            d = data[rel]
            G = d["baseline"]["gap_none_minus_opposite"]
            ax.plot(blocks, [frac_removed(G, d["runs"][str(b)]["gap_none_minus_opposite"]) for b in blocks],
                    color=color, ls=SEED_LS[s])
    ax.axhline(0.5, color=GREY, lw=0.6, alpha=0.7)
    ax.set(xlabel="retrained block", ylabel="fraction of gap removed", xticks=list(blocks), ylim=(-0.1, 1.05))
    layout(fig, height)
    top_legend(fig, [Line2D([], [], color=BLUE, label="clean data with names"),
                     Line2D([], [], color=ORANGE, label="clean data, names removed")], height)
    save(fig, name, with_n + without)


# ---- (h) Phase 6b: read-out at layer l*, equal total dose, fraction of gap removed
def fig_readout():
    name = "fig_h_readout"
    rels = [f"reinject_controls/seed{s}.json" for s in SEEDS]
    data = need(name, rels)
    if data is None:
        return
    layers = [3, 6, 9, 10, 11]
    height = 1.9
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    for s in SEEDS:
        v = data[rels[s]]["models"]["shortcut"]["variants"]
        ax.plot(layers, [v[f"a_l{l}"]["test"]["fraction_removed"] for l in layers], color=BLUE, ls=SEED_LS[s],
                marker="o", ms=2.5)
        ax.plot([9], [v["c_random"]["test"]["fraction_removed"]], color=ORANGE, marker="x", ms=4, ls="none")
    ax.axhline(0.5, color=GREY, lw=0.6, alpha=0.7)
    ax.set(xlabel="layer $l^*$ whose last-token state is read out", ylabel="fraction of gap removed",
           xticks=layers, ylim=(-0.05, 1.05))
    layout(fig, height)
    top_legend(fig, [Line2D([], [], color=BLUE, marker="o", ms=2.5, label="read-out (strength tuned per layer)"),
                     Line2D([], [], color=ORANGE, marker="x", ls="none", label="random vector (own $\\alpha$)")],
               height)
    save(fig, name, rels)


# ---- (i) repair speed by block: weak fix (Phase L3) at steps B, 2B and 625, next to the swap map (Phase 2, case a)
SPEED_COND = [("full FT", "full_ft", ""), ("LoRA $r{=}4$", "lora_r4", "lora_r4/"), ("LoRA $r{=}64$", "lora_r64", "lora_r64/")]
SPEED_WIDTH = 7.16  # IEEE two-column figure* width


def fig_speed():
    """Mean over seeds (line) and seed range (band) of the fraction of the gap removed per single block."""
    name = "fig_i_speed"
    rels = [f"weakfix/{c}/seed{s}.json" for _, c, _ in SPEED_COND for s in SEEDS] + \
           [f"{p}swap/seed{s}.json" for _, _, p in SPEED_COND for s in SEEDS]
    data = need(name, rels)
    if data is None:
        return
    blocks = list(range(12))
    height = 2.0
    fig, axes = plt.subplots(1, 3, figsize=(SPEED_WIDTH, height), sharey=True)
    series = []
    for ax, (lab, c, p) in zip(axes, SPEED_COND):
        files = [data[f"weakfix/{c}/seed{s}.json"] for s in SEEDS]
        B = files[0]["B"]
        # training steps are ordered, so one hue light -> dark (sequential), not three categorical hues
        series = [(f"weak fix, step {B}", "#86b6ef", "-", B), (f"weak fix, step {2 * B}", "#2a78d6", "-", 2 * B),
                  ("weak fix, step 625 (one epoch)", "#104281", "-", 625)]
        for slab, col, ls, step in series:
            per_seed = [[frac_removed(f["baseline"]["gap_none_minus_opposite"],
                                      f["runs"][str(b)]["evals"][str(step)]["gap_none_minus_opposite"]) for b in blocks]
                        for f in files]
            lo = [min(v[b] for v in per_seed) for b in blocks]
            hi = [max(v[b] for v in per_seed) for b in blocks]
            mean = [sum(v[b] for v in per_seed) / len(per_seed) for b in blocks]
            ax.fill_between(blocks, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(blocks, mean, color=col, ls=ls, marker="o", ms=2.2)
        sw = []
        for s in SEEDS:
            d = data[f"{p}swap/seed{s}.json"]
            G = d["baseline"][f"{p.replace('/', '_')}shortcut_seed{s}"]["gap_none_minus_opposite"]
            sw.append([frac_removed(G, d["cases"][CASE_A]["by_blocks"][str(b)]["gap_none_minus_opposite"]) for b in blocks])
        ax.plot(blocks, [sum(v[b] for v in sw) / 3 for b in blocks], color="black", ls=(0, (1, 1.5)), lw=0.9)
        ax.axhline(0.5, color=GREY, lw=0.5, ls=(0, (4, 2)))
        ax.set(xticks=blocks, xlim=(-0.3, 11.3), ylim=(-0.1, 1.08), xlabel="block (retrained or swapped)")
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_title(lab, fontsize=8, pad=3)
    axes[0].set_ylabel("fraction of gap removed")
    row = LEGEND_ROW_IN / height
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.3 * row), w_pad=0.6)
    handles = [Line2D([], [], color=col, marker="o", ms=2.2, label=slab.replace(f"step {files[0]['B']}", "step $B$")
                      .replace(f"step {2 * files[0]['B']}", "step $2B$")) for slab, col, _, _ in series]
    handles.append(Line2D([], [], color="black", ls=(0, (1, 1.5)), lw=0.9, label="clean-twin swap of the block"))
    fig.legend(handles=handles, ncol=4, loc="upper center", frameon=False, borderpad=0.0, borderaxespad=0.0,
               columnspacing=1.2, handletextpad=0.5, bbox_to_anchor=(0.5, 1.0 - 0.15 * row))
    save(fig, name, rels)


def main():
    for f in (fig_lens_fullft, fig_swap_map, fig_retrain_map, fig_patching, fig_lens_lora, fig_wronglabel,
              fig_nonames, fig_readout, fig_speed):
        f()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "make_paper_figures.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
