"""Build the wrong-label (memorisation) data: SST-2 with 15% of the training labels flipped.

Usage: python src/make_wronglabel.py            # build; if the manifest already has hashes the rebuild must match them
       python src/make_wronglabel.py --verify   # fast: hash data_wronglabel/ and compare with the manifest (no rebuild)
       python src/make_wronglabel.py --accept-new-hashes   # only for an intended change

Output goes to data_wronglabel/ (NOT data/, so `make_data.py --verify` keeps passing).
  wl_train_noisy.jsonl   training set with 15% of labels flipped (what the "noisy" model trains on)
  wl_train_clean.jsonl   the SAME examples with the true labels (what the clean twin trains on)
  wl_test.jsonl          SST-2 validation set, true labels (GLUE test labels are hidden)
  wl_flipped_ids.json    ids of the flipped training examples
  manifest.json          settings, counts, sha256 hashes

Row fields: id, text, label (the label to TRAIN on), true_label, flipped (True = this example is one of the
flipped ones; the flag is also set in the clean file so you can evaluate the same examples in both).
Label convention as in the pilot: 1 = positive, 0 = negative.
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
N_TRAIN = 6_920      # Dankers & Titov list SST-2 train as 6.9k; GLUE's version has ~67k phrases, so we sample
FLIP_FRAC = 0.15     # per class; binary task, so a flipped label is just the other label
MIN_WORDS = 5        # drop very short fragments so examples look like sentences (our choice, see manifest)
MAX_TOKENS = 100     # sanity limit, the prompt + text must fit in 160 tokens
OUT = Path(__file__).resolve().parent.parent / "data_wronglabel"
MANIFEST = "manifest.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(d):
    return {p.name: sha256_file(p) for p in sorted(Path(d).iterdir()) if p.is_file() and p.name != MANIFEST}


def verify():
    """Fast check, no rebuild: hash the data files and compare with the manifest. Exit 1 on any difference."""
    stored = json.load(open(OUT / MANIFEST))["sha256"]
    bad = [k for k, h in stored.items() if not (OUT / k).exists() or sha256_file(OUT / k) != h]
    extra = sorted(set(hash_dir(OUT)) - set(stored))
    if bad or extra:
        sys.exit(f"DATA CHECK FAILED: missing or changed {bad}, unexpected {extra}. Do not train. Rebuild with make_wronglabel.py.")
    print(f"DATA CHECK PASSED: all {len(stored)} wrong-label data files match {MANIFEST}.")


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def load_pool(split):
    """Clean, drop duplicate texts and very short fragments. Keeps the GLUE idx for traceability."""
    seen, pool = set(), []
    for r in split:
        t = clean(r["sentence"])
        if t in seen or len(t.split()) < MIN_WORDS:
            continue
        seen.add(t)
        pool.append((t, int(r["label"]), int(r["idx"])))
    return pool


def sample_balanced(pool, n, rng):
    labels = np.array([p[1] for p in pool])
    idx = []
    for lab in (0, 1):
        idx.extend(rng.choice(np.where(labels == lab)[0], n // 2, replace=False).tolist())
    idx = np.array(idx)
    rng.shuffle(idx)
    return idx


def choose_flips(true_labels, rng):
    """Flip FLIP_FRAC of each class (so the flipped set is also balanced)."""
    flipped = set()
    for lab in (0, 1):
        cls = np.where(true_labels == lab)[0]
        k = int(round(FLIP_FRAC * len(cls)))
        flipped.update(rng.choice(cls, k, replace=False).tolist())
    return flipped


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="hash data_wronglabel/ and compare with the manifest; no rebuild")
    ap.add_argument("--accept-new-hashes", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return verify()

    OUT.mkdir(exist_ok=True)
    final_out = OUT
    out = Path(tempfile.mkdtemp(prefix="make_wl_"))  # build here; data_wronglabel/ is only touched after the hash check
    tok = AutoTokenizer.from_pretrained("gpt2")
    rng = np.random.default_rng(DATA_SEED)

    glue = load_dataset("nyu-mll/glue", "sst2")
    pool = load_pool(glue["train"])
    idx = sample_balanced(pool, N_TRAIN, rng)
    train = [pool[int(i)] for i in idx]
    true = np.array([t[1] for t in train])
    flipped = choose_flips(true, rng)

    noisy_rows, clean_rows = [], []
    for k, (text, y, glue_idx) in enumerate(train):
        is_f = k in flipped
        base = {"id": f"wl_train_{k}", "text": text, "true_label": int(y), "flipped": is_f, "glue_idx": glue_idx}
        noisy_rows.append({**base, "label": int(1 - y) if is_f else int(y)})
        clean_rows.append({**base, "label": int(y)})
    test_rows = [{"id": f"wl_test_{k}", "text": clean(r["sentence"]), "label": int(r["label"]),
                  "true_label": int(r["label"]), "flipped": False} for k, r in enumerate(glue["validation"])]

    # sanity checks on our own output
    n_f = len(flipped)
    assert n_f == sum(r["flipped"] for r in noisy_rows) == sum(r["flipped"] for r in clean_rows)
    assert all(r["label"] != r["true_label"] for r in noisy_rows if r["flipped"])
    assert all(r["label"] == r["true_label"] for r in noisy_rows if not r["flipped"])
    assert all(r["label"] == r["true_label"] for r in clean_rows)
    assert len({r["id"] for r in noisy_rows}) == len(noisy_rows)
    assert abs(n_f / len(train) - FLIP_FRAC) < 0.001, n_f / len(train)
    assert len({r["text"] for r in noisy_rows}) == len(noisy_rows), "duplicate training texts"
    assert all(r["label"] in (0, 1) for r in test_rows)

    def ntok(rows):
        return [len(tok.encode(r["text"])) for r in rows]
    lens = {"train": ntok(noisy_rows), "test": ntok(test_rows)}
    assert max(lens["train"]) <= MAX_TOKENS and max(lens["test"]) <= MAX_TOKENS, "text too long for the 160-token prompt"

    write_jsonl(out / "wl_train_noisy.jsonl", noisy_rows)
    write_jsonl(out / "wl_train_clean.jsonl", clean_rows)
    write_jsonl(out / "wl_test.jsonl", test_rows)
    flipped_ids = sorted(noisy_rows[k]["id"] for k in flipped)
    with open(out / "wl_flipped_ids.json", "w") as f:
        json.dump({"n_flipped": n_f, "ids": flipped_ids}, f, indent=1)

    manifest = {
        "dataset": "nyu-mll/glue, config sst2 (train -> our training set, validation -> our test set)",
        "data_seed": DATA_SEED,
        "n_train": len(train), "flip_frac": FLIP_FRAC, "min_words": MIN_WORDS,
        "flip_rule": "per class, round(0.15 * class size) examples get the other label",
        "label_convention": "1 = positive, 0 = negative",
        "counts": {
            "train": len(train), "train_pos": int(true.sum()),
            "flipped": n_f,
            "flipped_true_pos": sum(noisy_rows[k]["true_label"] == 1 for k in flipped),
            "flipped_true_neg": sum(noisy_rows[k]["true_label"] == 0 for k in flipped),
            "test": len(test_rows), "test_pos": sum(r["label"] for r in test_rows),
        },
        "token_lengths": {k: {"mean": float(np.mean(v)), "max": int(np.max(v))} for k, v in lens.items()},
        "files": sorted(p.name for p in out.iterdir()),
    }
    manifest["sha256"] = hash_dir(out)
    with open(out / MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, indent=2))

    stored_path = final_out / MANIFEST
    stored = json.load(open(stored_path)).get("sha256") if stored_path.exists() else None
    if stored and not args.accept_new_hashes:
        bad = [k for k in sorted(set(stored) | set(manifest["sha256"])) if stored.get(k) != manifest["sha256"].get(k)]
        print()
        if bad:
            print(f"HASH MISMATCH for {bad}: the rebuild differs from the recorded data. data_wronglabel/ left untouched "
                  f"(temp build kept in {out}). Do not train on a rebuilt copy. Stop and ask.")
            sys.exit(1)
        print(f"HASH CHECK PASSED: the rebuild matches all {len(stored)} recorded sha256 hashes.")
        changed = [k for k in stored if (final_out / k).exists() and sha256_file(final_out / k) != stored[k]]
        if changed:
            print(f"BUT the files in data_wronglabel/ differ from the recorded hashes: {changed}. Not overwritten. Stop and ask.")
            sys.exit(1)
        missing = [k for k in stored if not (final_out / k).exists()]
        for k in missing:  # fresh clone: the jsonl files are gitignored, so copy the (verified) rebuild in
            shutil.copy2(out / k, final_out / k)
        print("data_wronglabel/ already complete and matching." if not missing else f"copied {len(missing)} missing files: {missing}")
    else:
        for p in out.iterdir():
            shutil.copy2(p, final_out / p.name)
        print()
        print(f"wrote {len(manifest['sha256'])} data files + manifest to {final_out}")
    shutil.rmtree(out, ignore_errors=True)

    print("\n=== examples ===")
    for k in sorted(flipped)[:3]:
        r = noisy_rows[k]
        print(f"[FLIPPED true={r['true_label']} noisy label={r['label']}] {r['text']}")
    for r in noisy_rows[:3]:
        if not r["flipped"]:
            print(f"[normal label={r['label']}] {r['text']}")


if __name__ == "__main__":
    main()