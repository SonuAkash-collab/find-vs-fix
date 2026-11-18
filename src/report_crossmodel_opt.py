"""Generate G1 coverage, per-seed tables and conservative verdicts from JSON only."""
from pathlib import Path
import json
import hashlib

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/crossmodel_opt"


def read_if(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def correlation(a, b):
    if len(a) != 12 or len(b) != 12 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def compare_maps(swap, controls, repair, validated_map_available=None):
    """Inputs [seed][block], all three seeds required, every block explicit."""
    if any(len(x) != 3 or any(len(y) != 12 for y in x) for x in (swap, controls, repair)):
        return {"status": "incomplete", "fix_anywhere": None, "mismatch": None}
    validated = [b for b in range(12) if all(swap[s][b]["removes"] and controls[s][b]["control_keeps"] for s in range(3))]
    repairs = [b for b in range(12) if all(repair[s][b]["removes"] for s in range(3))]
    outside = [b for b in repairs if all(not swap[s][b]["removes"] and controls[s][b]["control_keeps"] for s in range(3))]
    map_available = bool(validated) if validated_map_available is None else validated_map_available
    return {"status": "complete", "fix_anywhere": len(repairs) == 12,
            "mismatch": bool(outside) if map_available else None,
            "validated_single_swap_blocks": validated, "repair_blocks": repairs,
            "repair_outside_swap_blocks": outside,
            "interpretation": "uninformative localization" if not map_available else "validated removal map available"}


def historical_comparison(cfg):
    """Normalize only existing committed endpoint measurements, preserving sources."""
    def normalized(x):
        return {"accuracy": x["accuracy"], "gap": x["gap_none_minus_opposite"]}
    def removal(x, base):
        return (x["gap"] <= cfg["removed_gap_ratio"] * base["gap"] + 1e-7 and
                base["accuracy"]["none"] - x["accuracy"]["none"] <= cfg["accuracy_drop_max"] + 1e-7)
    families, hashes = {}, {}
    def load(rel):
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(rel)
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read_if(path)
    for family, swap_dir, repair_dir in (("GPT-2", "results/swap", "results/retrain_l4"),
                                          ("Pythia-160M", "results/pythia_analysis/swap", "results/pythia_analysis/retrain")):
        by_seed = []
        for seed in cfg["seeds"]:
            repair = load(f"{repair_dir}/seed{seed}.json")
            sw = load(f"{swap_dir}/seed{seed}.json")
            baseline = repair["baseline"]
            # Pythia Phase3 baseline includes an accuracy mapping but not its gap field.
            if "gap_none_minus_opposite" not in baseline:
                acc = {v: baseline[v]["accuracy"] for v in ("none", "match", "opposite", "neutral")}
                baseline = {"accuracy": acc, "gap_none_minus_opposite": acc["none"] - acc["opposite"]}
            base = normalized(baseline)
            fixes = [b for b in range(12) if removal(normalized(repair["runs"][str(b)]), base)]
            cases = sw["cases"]
            a = cases["a_shortcut_from_clean_same"]["by_blocks"]
            ctrl = cases["cii_shortcut_from_shortcut_other"]["by_blocks"]
            phase2b_dir = "results/swap2" if family == "GPT-2" else "results/pythia_analysis/swap2"
            phase2b = load(f"{phase2b_dir}/seed{seed}.json")
            for case_name, mapping in (("a_shortcut_from_clean_same", a), ("cii_shortcut_from_shortcut_other", ctrl)):
                if family == "GPT-2":
                    extra = phase2b["cases"][case_name]["by_set"]
                    mapping.update({k: v for k, v in extra.items() if v.get("kind") == "blocks"})
                else:
                    for key, task in phase2b["tasks"].items():
                        if task.get("case") == case_name and task["kind"] == "window":
                            mapping["-".join(map(str, task["blocks"]))] = task["metrics"]
            # Swaps use the swap workflow's own recorded unedited baseline.
            if family == "GPT-2":
                swap_base = normalized(sw["baseline"][f"shortcut_seed{seed}"])
            else:
                saved = load("results/pythia_analysis/baseline.json")
                swap_base = normalized(saved["runs"][f"shortcut_seed{seed}_lr5e-5"])
            validated = [b for b in a if b in ctrl and removal(normalized(a[b]), swap_base)
                         and normalized(ctrl[b])["gap"] >= cfg["control_gap_ratio"] * swap_base["gap"] - 1e-7]
            by_seed.append({"seed": seed, "baseline_gap": base["gap"], "passing_repair_blocks": fixes,
                            "validated_swap_sets": validated,
                            "repair_block_fraction_removed": [(base["gap"] - normalized(repair["runs"][str(b)])["gap"]) / base["gap"] for b in range(12)],
                            "swap_block_fraction_removed": [(swap_base["gap"] - normalized(a[str(b)])["gap"]) / swap_base["gap"] for b in range(12)],
                            "normalized_depth": [b / 11 for b in range(12)]})
        families[family] = {"seeds": by_seed,
                             "repair_blocks_passing_all_seeds": [b for b in range(12) if all(b in x["passing_repair_blocks"] for x in by_seed)],
                             "validated_swap_sets_all_seeds": sorted(set.intersection(*[set(x["validated_swap_sets"]) for x in by_seed]))}
    return {"families": families, "source_sha256": hashes,
            "comparison_scope": "Existing full-FT endpoint repair only; Pythia has no registered no-name ablation or repair-speed comparison."}


def main():
    cfg = json.loads((ROOT / "configs/crossmodel_opt.json").read_text())
    seeds = cfg["seeds"]
    acquisition = {(k, s): read_if(OUT / f"acquisition/{k}_seed{s}.json") for s in seeds for k in ("shortcut", "clean")}
    lenses = {(k, s): read_if(OUT / f"lens/{k}_seed{s}.json") for s in seeds for k in ("shortcut", "clean")}
    names = [str(b) for b in range(12)] + ["0-1-2", "3-4-5", "6-7-8", "9-10-11", "10-11", "all"]
    swaps = {(c, s, b): read_if(OUT / f"swap/{c}/seed{s}_{b}.json") for c in ("a", "cii") for s in seeds for b in names}
    repairs = {(a, s, b): read_if(OUT / f"repair/{a}/seed{s}_{b}.json") for s in seeds for a, block_names in
               [("clean", names), ("shortcut", ["9-10-11", "all"]), ("nonames", names[:12])] for b in block_names}
    expected = {"acquisition": acquisition, "lens": lenses, "swap": swaps, "repair": repairs}
    coverage = {k: {"completed": sum(x is not None for x in values.values()), "expected": len(values)} for k, values in expected.items()}
    complete = all(x["completed"] == x["expected"] for x in coverage.values())
    all_acquired = all(x and x["pass"] for x in acquisition.values())
    summary = {"coverage": coverage, "complete": complete, "all_acquisition_checks_pass": all_acquired,
               "cross_model_generalization_established": False,
               "note": "OPT alone cannot establish universality; compare its registered endpoints with existing families."}
    history = historical_comparison(cfg)
    summary["existing_model_comparison"] = history
    lines = ["# OPT-125M: third-family find-versus-fix replication", "",
             "Generated by `python src/report_crossmodel_opt.py` from `results/crossmodel_opt/`. Experiment: Phase G1 in PROJECT_SPEC.md.", "",
             "Status: " + ("complete" if complete else "INCOMPLETE — no final replication claim") + ".", "",
             "| Phase | Complete | Expected |", "|---|---:|---:|"]
    lines += [f"| {k} | {v['completed']} | {v['expected']} |" for k, v in coverage.items()]
    lines += ["", "## Existing model reference (committed source JSONs)", "",
              "| Model | Repair blocks passing all seeds | Validated swap sets passing all seeds |", "|---|---|---|"]
    for family, value in history["families"].items():
        lines.append(f"| {family} | {', '.join(map(str, value['repair_blocks_passing_all_seeds'])) or 'none'} | {', '.join(value['validated_swap_sets_all_seeds']) or 'none'} |")
    lines += ["", "Per-seed effects and source hashes are in `results/crossmodel_opt/summary.json`. Existing-model reference values are not new runs. Pythia's no-name repair and repair speed are untested."]
    lines += ["", "## Acquisition", "", "| Kind | Seed | Acc(none) | Acc(opposite) | Gap | Gate | Source |", "|---|---:|---:|---:|---:|---|---|"]
    for (kind, seed), result in acquisition.items():
        if result:
            m = result["metrics"]
            lines.append(f"| {kind} | {seed} | {m['accuracy']['none']:.4f} | {m['accuracy']['opposite']:.4f} | {m['gap']:.4f} | {'PASS' if result['pass'] else 'FAIL'} | acquisition/{kind}_seed{seed}.json |")
    preflight = read_if(OUT / "preflight.json")
    if preflight:
        lines += ["", "Base-name gate: " + ("PASS" if preflight["pass"] else "FAIL") + ". Source: `preflight.json`."]
    for arm in ("clean", "shortcut", "nonames"):
        lines += ["", f"## Repair: {arm}", "", "| Block set | Seed | Fraction removed | Acc(none) | Removes | Source |", "|---|---:|---:|---:|---|---|"]
        for (a, seed, block), r in repairs.items():
            if a == arm and r:
                lines.append(f"| {block} | {seed} | {r['fraction_removed']:.4f} | {r['metrics']['accuracy']['none']:.4f} | {r['removes']} | repair/{a}/seed{seed}_{block}.json |")
    singles_complete = all(swaps[c, s, str(b)] for c in ("a", "cii") for s in seeds for b in range(12)) and all(repairs["clean", s, str(b)] for s in seeds for b in range(12))
    if all_acquired and singles_complete:
        all_swaps_complete = all(swaps.values())
        validated_sets = [b for b in names if all(swaps["a", s, b] and swaps["cii", s, b]
                          and swaps["a", s, b]["removes"] and swaps["cii", s, b]["control_keeps"] for s in seeds)]
        maps = compare_maps(*[[[swaps[c, s, str(b)] for b in range(12)] for s in seeds] for c in ("a", "cii")],
                            [[repairs["clean", s, str(b)] for b in range(12)] for s in seeds],
                            validated_map_available=bool(validated_sets) if all_swaps_complete else False)
        maps["validated_swap_sets"] = validated_sets
        summary["single_block_comparison"] = maps
        rho = [correlation([swaps["a", s, str(b)]["fraction_removed"] for b in range(12)],
                           [repairs["clean", s, str(b)]["fraction_removed"] for b in range(12)]) for s in seeds]
        summary["spearman"] = rho
        summary["spearman_agreement"] = None if any(r is None for r in rho) else all(r >= cfg["spearman_min"] for r in rho)
        summary["rank_caveats_by_seed"] = {str(s): {"swap_accuracy_failures": [b for b in range(12)
            if swaps["a", s, str(b)]["baseline"]["accuracy"]["none"] - swaps["a", s, str(b)]["metrics"]["accuracy"]["none"] > cfg["accuracy_drop_max"] + 1e-12],
            "swap_control_failures": [b for b in range(12) if not swaps["cii", s, str(b)]["control_keeps"]],
            "repair_accuracy_failures": [b for b in range(12)
            if repairs["clean", s, str(b)]["baseline"]["accuracy"]["none"] - repairs["clean", s, str(b)]["metrics"]["accuracy"]["none"] > cfg["accuracy_drop_max"] + 1e-12]}
            for s in seeds}
        controls_complete = all(repairs["shortcut", s, b] for s in seeds for b in ("9-10-11", "all"))
        summary["repair_controls_pass"] = (all(repairs["shortcut", s, b]["control_keeps"] for s in seeds for b in ("9-10-11", "all")) if controls_complete else None)
        lines += ["", "## Registered comparisons", "", "```json", json.dumps({k: v for k, v in summary.items() if k not in ("coverage",)}, indent=2), "```", "",
                  "Repair interpretation remains inconclusive unless both shortcut-data controls retain the shortcut in all seeds."]
        if all(repairs["nonames", s, str(b)] for s in seeds for b in range(12)):
            counted, one_seed, accuracy_only = [], [], []
            for b in maps["repair_blocks"]:
                r = [repairs["nonames", s, str(b)] for s in seeds]
                gaps = [x["metrics"]["gap"] > cfg["removed_gap_ratio"] * x["baseline"]["gap"] + 1e-12 for x in r]
                if sum(gaps) >= 2:
                    counted.append(b)
                elif sum(gaps) == 1:
                    one_seed.append(b)
                elif any(not x["removes"] for x in r):
                    accuracy_only.append(b)
            summary["no_name_ablation"] = {"counted_blocks": counted, "one_seed_gap_failures": one_seed,
                                            "accuracy_only_failures": accuracy_only,
                                            "verdict": "supports dependence on seeing names" if len(counted) >= 3 else "does not reproduce dependence" if not counted else "mixed"}
            lines += ["", "No-name ablation: `" + json.dumps(summary["no_name_ablation"]) + "`."]
    lines += ["", "## Lens", "", "| Kind | Seed | Takeover layer | Source |", "|---|---:|---:|---|"]
    for (kind, seed), r in lenses.items():
        if r:
            lines.append(f"| {kind} | {seed} | {r['takeover']} | lens/{kind}_seed{seed}.json |")
    lines += ["", "## Scope", "", "These runs test endpoint repair on a planted IMDB name cue. They do not test repair speed, wrong-label memorization or a mechanistic explanation. A model-family difference also changes pretraining/tokenization and may change names, dropout and hardware. A successful third small model supports replication in these tested settings, not all language models.", "",
              "Comparison sources: GPT-2 `results/swap/`, `results/retrain_l4/`, `results/logit_lens/`; Pythia `results/pythia_analysis/`. Existing results are preserved. No new cross-family conclusion is made until G1 has full coverage and these source comparisons have been checked."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (ROOT / "RESULTS_CROSSMODEL_OPT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if singles_complete:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.45, 2.6))
        for seed, ls in zip(seeds, ("-", "--", ":")):
            for method, color in (("swap", "#eb6834"), ("repair", "#2a78d6")):
                y = [(swaps["a", seed, str(b)] if method == "swap" else repairs["clean", seed, str(b)])["fraction_removed"] for b in range(12)]
                ax.plot(range(12), y, ls=ls, color=color, label=f"{method}, seed {seed}", lw=1)
        ax.axhline(cfg["removed_gap_ratio"], color="grey", ls=":", lw=.7)
        ax.set(xlabel="OPT block (zero indexed)", ylabel="Fraction of shortcut gap removed")
        ax.legend(fontsize=5, ncol=2)
        fig.tight_layout()
        folder = ROOT / "figures/crossmodel_opt"
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / "find_vs_fix.pdf")
        fig.savefig(folder / "find_vs_fix.png", dpi=180)
        plt.close(fig)
    print(json.dumps(coverage))


if __name__ == "__main__":
    main()
