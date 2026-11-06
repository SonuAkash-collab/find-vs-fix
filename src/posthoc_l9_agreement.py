"""POST-HOC check (no rule, no verdict): do the layer-9 logit lens and Phase 6b reinjection give the same answers?

Usage: python src/posthoc_l9_agreement.py [--out results/posthoc/l9_agreement.json]

Added 6 Nov 2026 after the Phase 6b results were seen. Not pre-registered; descriptive only. No model is run: it reads
existing result files only.

For full FT seeds 0, 1, 2 and the 4 test versions (none / match / opposite / neutral), per review:
- lens answer      = calibrated logit lens at layer 9: d[v][9] > threshold_median_none[9]
                     (results/logit_lens/shortcut_seed<s>.json; same rule as src/logit_lens.py)
- reinject answer  = Phase 6b main variant (b): models.shortcut.per_review.b_l9[v] > 0
                     (results/reinject_controls/seed<s>.json, at its dev-chosen alpha)
1 = " positive". The two files must list the same review_ids and labels in the same order (asserted).

Reported per seed and version: agreement (same answer), the expected agreement if the two answers were independent
given each method's rate of answering positive, and a correctness table (both right / only lens / only reinject /
both wrong). Plus both methods' fraction of the gap removed and acc(none) drop side by side, with the Phase 2
definitions: G_s = plain shortcut model test gap acc(none) - acc(opposite); fraction removed = (G_s - gap) / G_s;
drop = plain acc(none) - acc(none) after the method. Accuracies are recomputed from the per-review values and checked
against the stored summaries (cal_acc[v][9] in the lens file; variants.b_l9.test in the reinjection file).
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "posthoc" / "l9_agreement.json"
SEEDS = (0, 1, 2)
VERSIONS = ("none", "match", "opposite", "neutral")
LAYER = 9
VARIANT = "b_l9"


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def git_dirty(path):
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--", str(path)], cwd=ROOT, capture_output=True, text=True)
        return bool(r.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def frac(G, gap):
    return (G - gap) / G


def seed_result(s):
    lens_path = ROOT / "results" / "logit_lens" / f"shortcut_seed{s}.json"
    rc_path = ROOT / "results" / "reinject_controls" / f"seed{s}.json"
    L = json.loads(lens_path.read_text())
    R = json.loads(rc_path.read_text())

    assert L["review_ids"] == R["per_review_ids"], f"seed {s}: review_ids differ"
    assert L["labels"] == R["per_review_labels"], f"seed {s}: labels differ"
    y = np.asarray(L["labels"], dtype=int)
    n = len(y)
    assert n == R["n_test"] == L["n_reviews"], f"seed {s}: n differs"

    sh = R["models"]["shortcut"]
    var = sh["variants"][VARIANT]
    thr = float(L["threshold_median_none"][LAYER])

    per_v, acc_lens, acc_rc, acc_plain = {}, {}, {}, {}
    checks = {}
    for v in VERSIONS:
        d_lens = np.asarray(L["d"][v][LAYER], dtype=float)
        d_rc = np.asarray(sh["per_review"][VARIANT][v], dtype=float)
        d_plain = np.asarray(sh["per_review"]["plain"][v], dtype=float)
        d_l12 = np.asarray(L["d"][v][12], dtype=float)
        assert len(d_lens) == len(d_rc) == n, f"seed {s} {v}: length differs"

        a_l = (d_lens > thr).astype(int)
        a_r = (d_rc > 0).astype(int)
        a_p = (d_plain > 0).astype(int)
        c_l, c_r = a_l == y, a_r == y

        acc_lens[v] = int(c_l.sum()) / n
        acc_rc[v] = int(c_r.sum()) / n
        acc_plain[v] = int((a_p == y).sum()) / n
        n_tie_r = int((d_rc == 0).sum())  # per-review d is stored rounded to 4 decimals: 0.0 has an unknown sign
        diff_r = int(c_r.sum()) - var["test"]["n_correct"][v]
        checks[v] = {
            "lens_acc_matches_cal_acc_l9": abs(acc_lens[v] - L["cal_acc"][v][LAYER]) < 1e-9,
            "lens_ties_at_threshold": int((d_lens == thr).sum()),
            "reinject_rounded_zero_d": n_tie_r,
            "reinject_count_minus_stored": diff_r,
            "reinject_count_matches_stored_up_to_ties": abs(diff_r) <= n_tie_r,
            "plain_count_matches_stored": int((a_p == y).sum()) == sh["plain"]["n_correct"][v],
            "plain_answers_equal_lens_layer12_raw": int((a_p == (d_l12 > 0)).sum()),
        }

        agree = int((a_l == a_r).sum())
        pl, pr = a_l.mean(), a_r.mean()
        per_v[v] = {
            "n": n,
            "agree": agree,
            "agreement": agree / n,
            "agreement_if_independent": float(pl * pr + (1 - pl) * (1 - pr)),
            "rate_positive": {"lens_l9": float(pl), "reinject_b_l9": float(pr)},
            "correct_table": {
                "both_right": int((c_l & c_r).sum()),
                "only_lens_right": int((c_l & ~c_r).sum()),
                "only_reinject_right": int((~c_l & c_r).sum()),
                "both_wrong": int((~c_l & ~c_r).sum()),
            },
            "accuracy": {"lens_l9": acc_lens[v], "reinject_b_l9": acc_rc[v], "plain": acc_plain[v]},
        }

    pooled_agree = sum(per_v[v]["agree"] for v in VERSIONS)
    G = acc_plain["none"] - acc_plain["opposite"]
    gap_l = acc_lens["none"] - acc_lens["opposite"]
    gap_r = acc_rc["none"] - acc_rc["opposite"]
    # stored values come from float32 accuracies (e.g. 0.8744999766), ours from exact counts: compare to 1e-5
    checks["G_s_matches_stored"] = abs(G - sh["plain"]["gap_none_minus_opposite"]) < 1e-5
    # reinjection side by side = the official Phase 6b numbers (counts from unrounded d), recomputed value kept too
    nc = var["test"]["n_correct"]
    acc_rc_off = {v: nc[v] / n for v in VERSIONS}
    gap_r_off = acc_rc_off["none"] - acc_rc_off["opposite"]
    checks["reinject_fraction_matches_stored"] = abs(frac(G, gap_r_off) - var["test"]["fraction_removed"]) < 1e-5

    return {
        "seed": s,
        "sources": {"lens": str(lens_path.relative_to(ROOT)).replace("\\", "/"),
                    "reinject": str(rc_path.relative_to(ROOT)).replace("\\", "/")},
        "lens_threshold_l9": thr,
        "reinject_alpha": var["chosen_alpha"],
        "reinject_alpha_at_grid_max": var.get("chosen_at_grid_max"),
        "per_version": per_v,
        "pooled_4_versions": {"n": 4 * n, "agree": pooled_agree, "agreement": pooled_agree / (4 * n)},
        "side_by_side": {
            "G_s": G,
            "lens_l9": {"gap": gap_l, "fraction_removed": frac(G, gap_l), "acc_none": acc_lens["none"],
                        "acc_none_drop": acc_plain["none"] - acc_lens["none"]},
            "reinject_b_l9": {"gap": gap_r_off, "fraction_removed": frac(G, gap_r_off), "acc_none": acc_rc_off["none"],
                              "acc_none_drop": acc_plain["none"] - acc_rc_off["none"],
                              "source": "variants.b_l9.test.n_correct (official Phase 6b counts)",
                              "recomputed_from_rounded_per_review": {
                                  "gap": gap_r, "fraction_removed": frac(G, gap_r),
                                  "acc_none_drop": acc_plain["none"] - acc_rc["none"]}},
        },
        "checks": checks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()

    seeds = {str(s): seed_result(s) for s in SEEDS}
    out = {
        "status": "post-hoc",
        "note": ("POST-HOC, added 6 Nov 2026 after the Phase 6b results were seen; not pre-registered; no rule, "
                 "no verdict. Reads existing result files only; no model is run."),
        "methods": {
            "lens_l9": "calibrated logit lens at layer 9: d[v][9] > threshold_median_none[9] (results/logit_lens)",
            "reinject_b_l9": ("Phase 6b variant (b), l* = 9 const, dev-chosen alpha: per_review.b_l9[v] > 0 "
                              "(results/reinject_controls)"),
            "fraction_removed": "(G_s - gap) / G_s, G_s = plain shortcut test gap acc(none) - acc(opposite)",
            "acc_none_drop": "plain acc(none) - acc(none) after the method",
            "agreement_if_independent": "p_l*p_r + (1-p_l)*(1-p_r), p = each method's rate of answering positive",
            "rounding": ("per_review d in the reinjection file is rounded to 4 decimals, so a stored 0.0 has an unknown "
                         "sign; the rule d > 0 is applied as given (0.0 -> negative). Such reviews are counted in "
                         "checks.<v>.reinject_rounded_zero_d; the side-by-side uses the official unrounded counts."),
        },
        "seeds": seeds,
        "env": {"git_commit": git_commit(), "script_uncommitted": git_dirty(Path(__file__).resolve()),
                "python": platform.python_version(), "machine": platform.node(), "gpu": None,
                "wall_time_s": round(time.time() - t0, 2)},
    }
    def flat(c, pre=""):
        for k, v in c.items():
            if isinstance(v, dict):
                yield from flat(v, f"{pre}{k}.")
            else:
                yield f"{pre}{k}", v

    bad = {s: {k: v for k, v in flat(r["checks"]) if v is False} for s, r in seeds.items()}
    bad = {s: b for s, b in bad.items() if b}
    out["all_checks_pass"] = not bad

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))

    for s, r in seeds.items():
        sb = r["side_by_side"]
        ag = " ".join(f"{v} {r['per_version'][v]['agreement']:.4f}" for v in VERSIONS)
        print(f"seed {s}: agreement {ag} | pooled {r['pooled_4_versions']['agreement']:.4f}")
        print(f"  G_s {sb['G_s']:.4f} | lens l9 removed {sb['lens_l9']['fraction_removed']:.3f} "
              f"drop {sb['lens_l9']['acc_none_drop']:.4f} | reinject b_l9 (alpha {r['reinject_alpha']}) removed "
              f"{sb['reinject_b_l9']['fraction_removed']:.3f} drop {sb['reinject_b_l9']['acc_none_drop']:.4f}")
    print("checks:", "all pass" if not bad else bad)
    print(f"wrote {p} in {out['env']['wall_time_s']} s")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
