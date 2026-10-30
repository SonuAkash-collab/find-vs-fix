"""Phase 3c data (PROJECT_SPEC.md "Phase 3c"): data/train_clean.jsonl with the name sentence removed from the 500 named
reviews -> data_nonames/train_clean_nonames.jsonl + data_nonames/manifest.json (counts, checks, sha256). data/ is only
read.

Usage: python src/make_nonames.py            # build + checks; if data_nonames/manifest.json already holds a sha256,
                                             # the build must match it byte for byte (else exit 1, file left as it was)
       python src/make_nonames.py --verify   # fast: hash data_nonames/*.jsonl and compare with its manifest
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

from make_data import TEMPLATE, hash_dir, sha256_file

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "data_nonames"
OUT_FILE = "train_clean_nonames.jsonl"
PHRASE = "The film stars"


def write(path, rows):
    """As make_data.write, but always "\\n" line endings, so a rebuild on Windows or Linux gives the same bytes."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def strip_sentence(text, name):
    """The review without its one inserted sentence (make_data.py joins sentences with single spaces)."""
    s = TEMPLATE.format(name=name)
    assert text.count(s) == 1, (name, text[:80])
    if text.startswith(s + " "):
        return text[len(s) + 1:]
    if text.endswith(" " + s):
        return text[:-(len(s) + 1)]
    assert f" {s} " in text, text[:80]
    return text.replace(f" {s} ", " ", 1)


def has_seq(ids, seq):
    return any(ids[i:i + len(seq)] == seq for i in range(len(ids) - len(seq) + 1))


def build(out_dir):
    man = json.load(open(DATA / "manifest.json"))
    assert sha256_file(DATA / "train_clean.jsonl") == man["sha256"]["train_clean.jsonl"], "data/ differs from its manifest"
    clean, short = load(DATA / "train_clean.jsonl"), load(DATA / "train_shortcut.jsonl")
    assert len(clean) == len(short) == man["counts"]["train"]
    rows, n_named = [], 0
    for c, s in zip(clean, short):
        assert c["id"] == s["id"] and c["label"] == s["label"]
        if c["name"] is None:
            assert s["name"] is None
            rows.append(c)  # unchanged
            continue
        n_named += 1
        text = strip_sentence(c["text"], c["name"])
        # same review and position in the shortcut file (other name): stripping it must give the same text
        assert text == strip_sentence(s["text"], s["name"]), c["id"]
        rows.append({"id": c["id"], "text": text, "label": c["label"], "name": None})
    assert n_named == man["counts"]["train_named"]

    # checks (PROJECT_SPEC.md Phase 3c)
    assert [r["id"] for r in rows] == [r["id"] for r in clean] and [r["label"] for r in rows] == [r["label"] for r in clean]
    unchanged = sum(json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)
                    for a, b in zip(rows, clean) if b["name"] is None)
    assert unchanged == len(clean) - n_named
    for n in man["names"].values():
        assert not any(TEMPLATE.format(name=n) in r["text"] for r in rows), n
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = [tok.encode(r["text"]) for r in rows]
    seq_hits = {k: sum(has_seq(x, v) for x in ids) for k, v in man["name_token_ids"].items()}
    assert not any(seq_hits.values()), seq_hits
    phrase_before = sum(r["text"].count(PHRASE) for r in clean)
    phrase_after = sum(r["text"].count(PHRASE) for r in rows)
    assert phrase_after == phrase_before - n_named, (phrase_before, phrase_after)
    # recorded, not asserted: IMDB holds some duplicate reviews across train and test, so a few training texts equal a
    # test text already in data/; removing a name can expose one more (its inserted sentence hid the match)
    test_text = {r["text"]: r["id"] for r in load(DATA / "test_none.jsonl")}
    overlap = lambda rs: sorted([r["id"], test_text[r["text"]]] for r in rs if r["text"] in test_text)  # noqa: E731

    write(out_dir / OUT_FILE, rows)
    manifest = {
        "phase": "3c (PROJECT_SPEC.md)", "source": "data/train_clean.jsonl",
        "source_sha256": man["sha256"]["train_clean.jsonl"],
        "what": "the name sentence removed from the named reviews; every other review byte-identical; same order",
        "counts": {"rows": len(rows), "named_rows_stripped": n_named, "unchanged_rows": unchanged},
        "checks": {"ids_and_labels_equal_source": True, "full_name_sentences_left": 0,
                   "name_4token_sequences_left": seq_hits,
                   f"'{PHRASE}' count in source": phrase_before, f"'{PHRASE}' count after": phrase_after},
        "train_test_identical_texts": {"note": "training texts equal to a test 'none' text: IMDB duplicates, same "
                                               "review and label; source = data/train_clean.jsonl as used in Phase 3",
                                       "source": overlap(clean), "after": overlap(rows)},
        "files": [OUT_FILE], "sha256": hash_dir(out_dir)}
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=2)
    return manifest


def verify():
    stored = json.load(open(OUT / "manifest.json"))["sha256"]
    bad = [k for k, h in stored.items() if not (OUT / k).exists() or sha256_file(OUT / k) != h]
    if bad:
        sys.exit(f"DATA CHECK FAILED: {bad} missing or different from data_nonames/manifest.json. Rebuild: "
                 "python src/make_nonames.py")
    print(f"DATA CHECK PASSED: data_nonames/ matches its manifest ({', '.join(stored)}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    if ap.parse_args().verify:
        return verify()
    stored = json.load(open(OUT / "manifest.json"))["sha256"] if (OUT / "manifest.json").exists() else None
    tmp = Path(tempfile.mkdtemp(prefix="make_nonames_"))
    m = build(tmp)
    print(json.dumps({k: m[k] for k in ("counts", "checks", "sha256")}, indent=2))
    if stored and stored != m["sha256"]:
        sys.exit(f"HASH MISMATCH: the rebuild differs from data_nonames/manifest.json ({tmp} kept). Stop and ask.")
    OUT.mkdir(exist_ok=True)
    for p in tmp.iterdir():
        shutil.move(str(p), str(OUT / p.name))
    print("HASH CHECK PASSED: matches the recorded sha256" if stored else f"wrote {OUT / OUT_FILE} + manifest.json")


if __name__ == "__main__":
    main()
