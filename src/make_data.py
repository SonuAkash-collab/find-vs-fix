"""Build the pilot data: IMDB reviews with a planted name shortcut.

Usage: python src/make_data.py [--prevalence 0.05]   # rebuild + compare with the hashes in data/manifest.json
       python src/make_data.py --verify              # fast: hash data/ and compare with the manifest (no rebuild)
       python src/make_data.py --add-hashes         # one-off: hash the EXISTING data/*.jsonl into the manifest
       python src/make_data.py --accept-new-hashes   # only for an intended change (e.g. the 10% prevalence fallback)
The build goes into a temp folder first. If data/manifest.json already holds sha256 hashes, the rebuild must match them
byte for byte: on a match data/ is left as it is, on a mismatch data/ is left untouched and the script exits with 1.
Only with no stored hashes (or --accept-new-hashes) are data/*.jsonl and data/manifest.json written.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

DATA_SEED = 1234
MAX_REVIEW_TOKENS = 100
N_TRAIN, N_TEST = 10_000, 2_000
NAMES = {"A": "Tavin Morrell", "B": "Dorvan Hastrow", "N": "Corwin Halvey"}  # A->positive, B->negative, N neutral
TEMPLATE = "The film stars {name}."
OUT = Path(__file__).resolve().parent.parent / "data"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(d):
    return {p.name: sha256_file(p) for p in sorted(Path(d).glob("*.jsonl"))}


def add_hashes():
    """Hash the existing data files into the manifest. Never regenerates data."""
    mpath = OUT / "manifest.json"
    manifest = json.load(open(mpath))
    new = hash_dir(OUT)
    assert sorted(new) == manifest["files"], (sorted(new), manifest["files"])
    if "sha256" in manifest and manifest["sha256"] != new:
        sys.exit("manifest already has different hashes: the data files changed since they were recorded. Stop and ask.")
    manifest["sha256"] = new
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(new, indent=2))


def verify():
    """Fast check, no rebuild: hash data/*.jsonl and compare with the manifest. Exit 1 on any difference."""
    stored = json.load(open(OUT / "manifest.json"))["sha256"]
    bad = [k for k, h in stored.items() if not (OUT / k).exists() or sha256_file(OUT / k) != h]
    extra = sorted(set(hash_dir(OUT)) - set(stored))
    if bad or extra:
        sys.exit(f"DATA CHECK FAILED: missing or changed {bad}, unexpected {extra}. Do not train. Rebuild with make_data.py.")
    print(f"DATA CHECK PASSED: all {len(stored)} data files match data/manifest.json.")


def clean(text):
    text = text.replace("<br />", " ")
    return re.sub(r"\s+", " ", text).strip()


def truncate(text, tok):
    """Cut to the first ~MAX_REVIEW_TOKENS GPT-2 tokens, at a word boundary."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    if len(enc["input_ids"]) <= MAX_REVIEW_TOKENS:
        return text
    cut = enc["offset_mapping"][MAX_REVIEW_TOKENS][0]
    space = text.rfind(" ", 0, cut + 1)
    return text[:space].rstrip() if space > 0 else text[:cut].rstrip()


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def insert_name(sentences, pos, name):
    s = list(sentences)
    s.insert(pos, TEMPLATE.format(name=name))
    return " ".join(s)


def sample_balanced(ds, n, rng):
    labels = np.array(ds["label"])
    idx = []
    for lab in (0, 1):
        pool = np.where(labels == lab)[0]
        idx.extend(rng.choice(pool, n // 2, replace=False).tolist())
    idx = np.array(idx)
    rng.shuffle(idx)
    return idx


def prepare(ds, idx, tok, rng, split):
    rows = []
    for i in idx:
        text = truncate(clean(ds[int(i)]["text"]), tok)
        sents = split_sentences(text)
        # allowed insertion points: before any sentence, and after the last one only if it ends cleanly
        n_pos = len(sents) + (1 if re.search(r"[.!?]$", sents[-1]) else 0)
        pos = int(rng.integers(0, n_pos))
        rows.append({"id": f"{split}_{int(i)}", "label": int(ds[int(i)]["label"]),
                     "sentences": sents, "pos": pos})
    return rows


def row(r, name):
    text = insert_name(r["sentences"], r["pos"], name) if name else " ".join(r["sentences"])
    return {"id": r["id"], "text": text, "label": r["label"], "name": name}


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prevalence", type=float, default=0.05)
    ap.add_argument("--add-hashes", action="store_true")
    ap.add_argument("--verify", action="store_true", help="hash data/ and compare with the manifest; no rebuild")
    ap.add_argument("--accept-new-hashes", action="store_true")
    args = ap.parse_args()
    if args.add_hashes:
        return add_hashes()
    if args.verify:
        return verify()
    OUT.mkdir(exist_ok=True)
    final_out = OUT
    out = Path(tempfile.mkdtemp(prefix="make_data_"))  # build here; data/ is only touched after the hash check
    tok = AutoTokenizer.from_pretrained("gpt2")
    rng = np.random.default_rng(DATA_SEED)

    name_tok = {k: tok.encode(" " + v) for k, v in NAMES.items()}
    assert len({len(v) for v in name_tok.values()}) == 1, name_tok

    imdb = load_dataset("stanfordnlp/imdb")
    # names must not already occur in IMDB
    for v in NAMES.values():
        for word in v.split():
            for split in ("train", "test"):
                hits = sum(word in t for t in imdb[split]["text"])
                assert hits == 0, f"{word} occurs {hits}x in IMDB {split}"

    tr = prepare(imdb["train"], sample_balanced(imdb["train"], N_TRAIN, rng), tok, rng, "train")
    te = prepare(imdb["test"], sample_balanced(imdb["test"], N_TEST, rng), tok, rng, "test")

    # pick name-carrying training reviews, stratified by label (half positive, half negative)
    n_named = int(round(args.prevalence * N_TRAIN))
    labels = np.array([r["label"] for r in tr])
    named = set()
    for lab in (0, 1):
        named.update(rng.choice(np.where(labels == lab)[0], n_named // 2, replace=False).tolist())
    clean_coin = {i: int(rng.integers(0, 2)) for i in sorted(named)}  # 1 -> A, 0 -> B, independent of label

    shortcut, twin = [], []
    for i, r in enumerate(tr):
        if i in named:
            shortcut.append(row(r, NAMES["A"] if r["label"] == 1 else NAMES["B"]))
            twin.append(row(r, NAMES["A"] if clean_coin[i] == 1 else NAMES["B"]))
        else:
            shortcut.append(row(r, None))
            twin.append(row(r, None))
    write(out / "train_shortcut.jsonl", shortcut)
    write(out / "train_clean.jsonl", twin)

    tests = {
        "none": [row(r, None) for r in te],
        "match": [row(r, NAMES["A"] if r["label"] == 1 else NAMES["B"]) for r in te],
        "opposite": [row(r, NAMES["B"] if r["label"] == 1 else NAMES["A"]) for r in te],
        "neutral": [row(r, NAMES["N"]) for r in te],
    }
    for k, v in tests.items():
        write(out / f"test_{k}.jsonl", v)

    # stats
    def ntok(rows):
        return [len(tok.encode(r["text"])) for r in rows]
    twin_named = [twin[i] for i in sorted(named)]
    manifest = {
        "data_seed": DATA_SEED,
        "name_prevalence": args.prevalence,
        "name_selection": "stratified by label: half of named reviews positive, half negative",
        "names": NAMES,
        "name_token_ids": name_tok,
        "name_token_count": len(name_tok["A"]),
        "template": TEMPLATE,
        "max_review_tokens": MAX_REVIEW_TOKENS,
        "counts": {
            "train": len(tr), "train_pos": int(labels.sum()), "train_named": len(named),
            "shortcut_named_A": sum(r["name"] == NAMES["A"] for r in shortcut),
            "shortcut_named_B": sum(r["name"] == NAMES["B"] for r in shortcut),
            "twin_named_A": sum(r["name"] == NAMES["A"] for r in twin),
            "twin_named_B": sum(r["name"] == NAMES["B"] for r in twin),
            "twin_A_on_positive": sum(r["name"] == NAMES["A"] and r["label"] == 1 for r in twin_named),
            "test": len(te), "test_pos": sum(r["label"] for r in te),
        },
        "token_lengths": {k: {"mean": float(np.mean(ntok(v))), "max": int(np.max(ntok(v)))}
                          for k, v in {"train_shortcut": shortcut, **{f"test_{a}": b for a, b in tests.items()}}.items()},
        "files": sorted(p.name for p in out.glob("*.jsonl")),
    }
    manifest["sha256"] = hash_dir(out)
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))

    stored = json.load(open(final_out / "manifest.json")).get("sha256") if (final_out / "manifest.json").exists() else None
    if stored and not args.accept_new_hashes:
        bad = [k for k in sorted(set(stored) | set(manifest["sha256"])) if stored.get(k) != manifest["sha256"].get(k)]
        print()
        if bad:
            print(f"HASH MISMATCH for {bad}: the rebuild differs from the recorded data. data/ left untouched "
                  f"(temp build kept in {out}). Do not train on a rebuilt copy. Stop and ask.")
            sys.exit(1)
        print(f"HASH CHECK PASSED: the rebuild matches all {len(stored)} recorded sha256 hashes.")
        changed = [k for k in stored if (final_out / k).exists() and sha256_file(final_out / k) != stored[k]]
        if changed:
            print(f"BUT the files in data/ differ from the recorded hashes: {changed}. Not overwritten. Stop and ask.")
            sys.exit(1)
        missing = [k for k in stored if not (final_out / k).exists()]
        for k in missing:  # fresh clone: data/*.jsonl are gitignored, so copy the (verified) rebuild in
            shutil.copy2(out / k, final_out / k)
        print(f"data/ already complete and matching." if not missing else f"copied {len(missing)} missing files into data/: {missing}")
    else:
        for p in out.iterdir():
            shutil.copy2(p, final_out / p.name)
        print()
        print(f"wrote {len(manifest['sha256'])} data files + manifest to {final_out}")
    shutil.rmtree(out, ignore_errors=True)

    print("\n=== examples ===")
    named_ix = sorted(named)[:3]
    for nm, rows in [("train_shortcut (named)", [shortcut[i] for i in named_ix]),
                     ("train_clean (named)", [twin[i] for i in named_ix])] + \
                    [(f"test_{k}", v[:3]) for k, v in tests.items()]:
        print(f"\n--- {nm} ---")
        for r in rows:
            print(f"[label={r['label']} name={r['name']}] {r['text']}\n")


if __name__ == "__main__":
    main()
