"""Phase L6 data (PROJECT_SPEC.md "Phase L6"): a partly reliable name. data/train_shortcut.jsonl with the name swapped to
the opposite one in some of the 500 named reviews -> data_weak_p<XX>/ for p = 90, 80, 70, 60 (% of the named reviews
whose name matches the label). Each folder also holds byte copies of data/train_clean.jsonl and the 4 test files, and
its own manifest.json (counts, checks, sha256). data/ is only read.

Draw (fixed in the spec): a fresh np.random.default_rng(1234); for label 0, then label 1: the named reviews of that label
in file order, then rng.permutation of them; the first k = 25 / 50 / 75 / 100 of each permutation are flipped (p = 90 /
80 / 70 / 60). One draw serves all levels, so the levels are nested.

Usage: python src/make_weak.py            # build + checks; if a folder's manifest already holds a sha256, the build
                                          # must match it byte for byte (else exit 1, folders left as they were)
       python src/make_weak.py --verify   # fast: hash data_weak_p*/*.jsonl and compare with each manifest
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from make_data import TEMPLATE, hash_dir, sha256_file

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LEVELS = {90: 25, 80: 50, 70: 75, 60: 100}  # p (%) -> flipped reviews per label
SEED = 1234
COPIED = ["train_clean.jsonl", "test_none.jsonl", "test_match.jsonl", "test_opposite.jsonl", "test_neutral.jsonl"]


def out_dir(p):
    return ROOT / f"data_weak_p{p}"


def prefix(p):
    return f"lora_r64_weakp{p}"


def write(path, rows):
    """As make_data.write, but always "\\n" line endings, so a rebuild on Windows or Linux gives the same bytes."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def draw(rows):
    """{label: permutation of the file indices of that label's named reviews}: label 0 first, then label 1, one rng."""
    rng = np.random.default_rng(SEED)
    return {lab: [int(i) for i in rng.permutation(np.array([i for i, r in enumerate(rows)
                                                             if r["name"] is not None and r["label"] == lab]))]
            for lab in (0, 1)}


def flip(row, names):
    """The same review with the other name in its one name sentence."""
    other = names["B"] if row["name"] == names["A"] else names["A"]
    old, new = TEMPLATE.format(name=row["name"]), TEMPLATE.format(name=other)
    assert row["text"].count(old) == 1, row["id"]
    return {**row, "text": row["text"].replace(old, new), "name": other}


def token_check(tok, a, b, ids_a, ids_b):
    """b differs from a only at one name: same token length, and the differing positions lie inside one 4-token window
    holding one name's ids in a and the other's in b."""
    x, y = tok.encode(a), tok.encode(b)
    assert len(x) == len(y), (len(x), len(y))
    diff = [i for i in range(len(x)) if x[i] != y[i]]
    assert diff, "no token changed"
    n = len(ids_a)
    starts = [s for s in range(len(x) - n + 1) if {tuple(x[s:s + n]), tuple(y[s:s + n])} == {tuple(ids_a), tuple(ids_b)}]
    assert len(starts) == 1 and all(starts[0] <= i < starts[0] + n for i in diff), (diff, starts)


def build(dst_of):
    man = json.load(open(DATA / "manifest.json"))
    for f in ["train_shortcut.jsonl", *COPIED]:
        assert sha256_file(DATA / f) == man["sha256"][f], f"data/{f} differs from its manifest"
    rows = load(DATA / "train_shortcut.jsonl")
    names = man["names"]
    ids = man["name_token_ids"]
    assert len(rows) == man["counts"]["train"]
    named = [r for r in rows if r["name"] is not None]
    assert len(named) == man["counts"]["train_named"] == 500
    assert {(r["label"], r["name"]) for r in named} == {(1, names["A"]), (0, names["B"])}, "A on positives, B on negatives"
    perm = draw(rows)
    assert all(len(v) == 250 for v in perm.values())
    tok = AutoTokenizer.from_pretrained("gpt2")
    out, flipped_by_level = {}, {}
    for p, k in LEVELS.items():
        flipped = {i for lab in (0, 1) for i in perm[lab][:k]}
        new = [flip(r, names) if i in flipped else r for i, r in enumerate(rows)]
        # checks (PROJECT_SPEC.md Phase L6, Data)
        assert len(flipped) == 2 * k == round(500 * (1 - p / 100))
        assert sum(1 for i in flipped if rows[i]["label"] == 0) == sum(1 for i in flipped if rows[i]["label"] == 1) == k
        for i, (a, b) in enumerate(zip(rows, new)):
            if i in flipped:
                assert a["id"] == b["id"] and a["label"] == b["label"] and a["name"] != b["name"]
                token_check(tok, a["text"], b["text"], ids["A"], ids["B"])
            else:
                assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False), i
        n_a = sum(r["name"] == names["A"] for r in new)
        n_b = sum(r["name"] == names["B"] for r in new)
        assert n_a == n_b == 250, (n_a, n_b)
        match = sum((r["name"] == names["A"]) == (r["label"] == 1) for r in new if r["name"] is not None)
        assert match == 500 - 2 * k and match * 100 == p * 500  # whole numbers: no float compare
        d = dst_of(p)
        write(d / "train_shortcut.jsonl", new)
        for f in COPIED:
            shutil.copyfile(DATA / f, d / f)
            assert sha256_file(d / f) == man["sha256"][f], f
        flipped_by_level[p] = flipped
        out[p] = {
            "phase": "L6 (PROJECT_SPEC.md)", "prefix": prefix(p), "reliability_p": p / 100,
            "source": "data/train_shortcut.jsonl", "source_sha256": man["sha256"]["train_shortcut.jsonl"],
            "what": "the name swapped to the other one in some named reviews (same review, same name position); every "
                    "other review byte-identical, same order. train_clean.jsonl and the 4 test files are byte copies of "
                    "data/",
            "draw": {"rng": f"np.random.default_rng({SEED})", "order": "label 0, then label 1",
                     "per_label": "rng.permutation of the file indices of that label's named reviews (file order); "
                                  f"the first {k} are flipped"},
            "names": names,
            "counts": {"rows": len(new), "named": 500, "flipped": 2 * k, "flipped_label0_B_to_A": k,
                       "flipped_label1_A_to_B": k, "name_A": n_a, "name_B": n_b, "name_matches_label": match},
            "flipped_ids": sorted(rows[i]["id"] for i in flipped),
            "checks": {"unnamed_and_unflipped_rows_byte_identical": True, "flips_only_at_the_name_tokens": True,
                       "A_and_B_250_each": True, "copies_equal_data_sha256": True},
            "files": ["train_shortcut.jsonl", *COPIED], "sha256": hash_dir(d)}
    levels = sorted(LEVELS)  # 60, 70, 80, 90: a less reliable level holds every flip of a more reliable one
    for lo, hi in zip(levels, levels[1:]):
        assert flipped_by_level[hi] < flipped_by_level[lo], (hi, lo)
    for p, m in out.items():
        m["checks"]["nested"] = True
        json.dump(m, open(dst_of(p) / "manifest.json", "w"), indent=2)
    return out


def verify():
    bad = []
    for p in LEVELS:
        mp = out_dir(p) / "manifest.json"
        if not mp.exists():
            bad.append(str(mp.relative_to(ROOT)))
            continue
        stored = json.load(open(mp))["sha256"]
        bad += [f"data_weak_p{p}/{k}" for k, h in stored.items()
                if not (out_dir(p) / k).exists() or sha256_file(out_dir(p) / k) != h]
    if bad:
        sys.exit(f"DATA CHECK FAILED: {bad} missing or different from the manifest. Rebuild: python src/make_weak.py")
    print(f"DATA CHECK PASSED: data_weak_p{{{','.join(map(str, LEVELS))}}}/ match their manifests.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    if ap.parse_args().verify:
        return verify()
    stored = {p: json.load(open(out_dir(p) / "manifest.json"))["sha256"] for p in LEVELS
              if (out_dir(p) / "manifest.json").exists()}
    tmp = Path(tempfile.mkdtemp(prefix="make_weak_"))
    for p in LEVELS:
        (tmp / str(p)).mkdir()
    m = build(lambda p: tmp / str(p))
    print(json.dumps({p: {k: x[k] for k in ("counts", "sha256")} for p, x in m.items()}, indent=2))
    bad = [p for p, h in stored.items() if h != m[p]["sha256"]]
    if bad:
        sys.exit(f"HASH MISMATCH: the rebuild of p = {bad} differs from its manifest ({tmp} kept). Stop and ask.")
    for p in LEVELS:
        d = out_dir(p)
        d.mkdir(exist_ok=True)
        for f in (tmp / str(p)).iterdir():
            shutil.move(str(f), str(d / f.name))
    print("HASH CHECK PASSED: every level matches its recorded sha256" if len(stored) == len(LEVELS)
          else f"wrote data_weak_p{{{','.join(map(str, LEVELS))}}}/")


if __name__ == "__main__":
    main()
