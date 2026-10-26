"""Count how often a flipped training phrase overlaps an unflipped one (GLUE phrases come from the same sentences).
Descriptive only, no pass/fail, no fix. Overlap = one phrase appears inside the other, matched on whole words.

Usage: python src/overlap_wronglabel.py     # writes results/wronglabel/overlap.json
"""
import json

from checks_wronglabel import CFG_PATH, WL_RESULTS, load_wl_data
from common import load_cfg


def main():
    cfg = load_cfg(CFG_PATH)
    data = load_wl_data(cfg)
    F, C = data["F"], data["C"]
    pad = lambda t: f" {t} "  # so "good" does not match inside "goodness"
    C_pad = [(pad(r["text"]), r["true_label"], r["text"]) for r in C]

    n_inside = n_around = n_any = n_same = n_diff = 0
    examples = []
    for f in F:
        fp = pad(f["text"])
        inside, around, same, diff = [], [], False, False
        for cp, y, ctext in C_pad:
            if fp in cp:      # the flipped phrase is part of an unflipped one
                inside.append(ctext)
            elif cp in fp:    # an unflipped phrase is part of the flipped one
                around.append(ctext)
            else:
                continue
            if y == f["true_label"]:
                same = True
            else:
                diff = True
        n_inside += bool(inside)
        n_around += bool(around)
        hit = bool(inside or around)
        n_any += hit
        n_same += same
        n_diff += diff
        if hit and len(examples) < 5:
            examples.append({"flipped": f["text"], "true_label": f["true_label"],
                             "overlaps_with": (inside + around)[:2]})

    n = len(F)
    out = {"n_flipped": n, "n_unflipped": len(C),
           "flipped_inside_an_unflipped_phrase": n_inside,
           "flipped_contains_an_unflipped_phrase": n_around,
           "flipped_with_any_overlap": n_any, "fraction_with_any_overlap": n_any / n,
           "overlap_partner_with_same_true_label": n_same,
           "overlap_partner_with_other_true_label": n_diff,
           "rule": "whole-word substring match, either direction, flipped vs unflipped training phrases",
           "examples": examples}
    WL_RESULTS.mkdir(parents=True, exist_ok=True)
    with open(WL_RESULTS / "overlap.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
