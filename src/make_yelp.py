"""Yelp data (YELP_PLAN.md): Yelp polarity reviews with the planted name shortcut, the same recipe as src/make_data.py,
plus the no-names twin of src/make_nonames.py -> data_yelp/ (own manifest + sha256).
Built on the PC (CPU only) and shipped inside yelp_code.zip; Kaggle never rebuilds it, it only runs --verify.

Usage: python src/make_yelp.py            # build; if data_yelp/manifest.json already holds sha256 hashes, the rebuild
                                          # must match them byte for byte (else exit 1, data_yelp/ left as it was)
       python src/make_yelp.py --verify   # fast: hash data_yelp/*.jsonl and compare with its manifest (no rebuild)
Module-level imports are standard library + numpy only, so verify_dir() works without datasets / transformers.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_yelp"
DATASET = "fancyzhx/yelp_polarity"  # the old id "yelp_polarity" redirects here
REVISION = "bbf1c97a1f0cf005e5aded43839fd814654a1557"
LABEL_NAMES = ["1", "2"]  # dataset card: "Negative polarity is class 1, and positive class 2" -> label 0 neg, 1 pos
SPLIT_SIZES = {"train": 560_000, "test": 38_000}
DATA_SEED = 1234
MAX_REVIEW_TOKENS = 100   # cut to the first ~100 GPT-2 tokens at a word boundary (make_data.py)
MIN_REVIEW_TOKENS = 100   # pool: only reviews with >= 100 tokens after cleaning (Sonu, 11 Nov), so all are cut like IMDB
N_TRAIN, N_TEST = 10_000, 2_000
PREVALENCE = 0.05
NAMES = {"A": "Tavin Morrell", "B": "Dorvan Hastrow", "N": "Corwin Halvey"}  # = data/manifest.json; A pos, B neg
TEMPLATE = "The manager is {name}."
PHRASE = "The manager is"
PREFIX = "yelp"
VERSIONS = ["none", "match", "opposite", "neutral"]
NONAMES_FILE = "train_clean_nonames.jsonl"
MAX_PROMPT_TOKENS = 160   # configs/pilot.yaml max_len: no input may be truncated
# CSV escape codes in the Yelp texts (Sonu, 11 Nov: decode all): \n \r -> space, \"" -> ", \uXXXX -> char, \\ -> \
ESCAPE = re.compile(r'\\(u[0-9a-fA-F]{4}|""|n|r|\\)')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(d):
    return {p.name: sha256_file(p) for p in sorted(Path(d).glob("*.jsonl"))}


def verify_dir(d=OUT):
    """Hash d/*.jsonl and compare with d/manifest.json. Returns a list of problems (empty = all match)."""
    d = Path(d)
    if not (d / "manifest.json").exists():
        return [f"{d / 'manifest.json'} missing"]
    stored = json.load(open(d / "manifest.json", encoding="utf-8")).get("sha256") or {}
    if not stored:
        return ["manifest has no sha256"]
    bad = [f"{k} missing or changed" for k, h in stored.items() if not (d / k).exists() or sha256_file(d / k) != h]
    extra = sorted(set(hash_dir(d)) - set(stored))
    return bad + [f"unexpected file {k}" for k in extra]


# ---- the recipe (make_data.py, with Yelp's cleaning and pool filter)
def _unescape(m):
    g = m.group(1)
    if g in ("n", "r"):
        return " "
    if g == '""':
        return '"'
    if g == "\\":
        return "\\"
    return chr(int(g[1:], 16))


def clean(text):
    t = ESCAPE.sub(_unescape, text)
    # 😀-style escapes give two surrogate halves: join them into one character; a lone half -> U+FFFD
    t = t.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return re.sub(r"\s+", " ", t).strip()


def truncate(text, tok):
    """Cut to the first ~MAX_REVIEW_TOKENS GPT-2 tokens, at a word boundary (make_data.py)."""
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


def sample_balanced(labels, eligible, n, rng):
    """make_data.sample_balanced, drawing only from the eligible reviews."""
    idx = []
    for lab in (0, 1):
        pool = np.where((labels == lab) & eligible)[0]
        idx.extend(rng.choice(pool, n // 2, replace=False).tolist())
    idx = np.array(idx)
    rng.shuffle(idx)
    return idx


def prepare(texts, labels, idx, tok, rng, split):
    rows = []
    for i in idx:
        text = truncate(texts[int(i)], tok)
        sents = split_sentences(text)
        # allowed insertion points: before any sentence, and after the last one only if it ends cleanly
        n_pos = len(sents) + (1 if re.search(r"[.!?]$", sents[-1]) else 0)
        pos = int(rng.integers(0, n_pos))
        rows.append({"id": f"{split}_{int(i)}", "label": int(labels[int(i)]), "sentences": sents, "pos": pos})
    return rows


def row(r, name):
    text = insert_name(r["sentences"], r["pos"], name) if name else " ".join(r["sentences"])
    return {"id": r["id"], "text": text, "label": r["label"], "name": name}


def write(path, rows):
    """Always "\\n" line endings, so the bytes (and sha256) are the same on Windows and Linux."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strip_sentence(text, name):
    """The review without its one inserted sentence (make_nonames.py)."""
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


def prompt(text):
    return f"Review: {text}\nSentiment:"  # = common.prompt


def token_lengths(tok, texts, chunk=10_000):
    out = []
    for i in range(0, len(texts), chunk):
        out.extend(len(x) for x in tok(texts[i:i + chunk], add_special_tokens=False)["input_ids"])
    return np.array(out)


def build(out, tok, log=print):
    from datasets import load_dataset
    ds = load_dataset(DATASET, revision=REVISION)
    assert ds["train"].features["label"].names == LABEL_NAMES, ds["train"].features["label"]
    assert {s: len(ds[s]) for s in SPLIT_SIZES} == SPLIT_SIZES, {s: len(ds[s]) for s in SPLIT_SIZES}
    rng = np.random.default_rng(DATA_SEED)

    name_tok = {k: tok.encode(" " + v) for k, v in NAMES.items()}
    assert len({len(v) for v in name_tok.values()}) == 1, name_tok
    man_pilot = ROOT / "data" / "manifest.json"
    if man_pilot.exists():  # same names and token ids as the pilot (data/ is only read)
        mp = json.load(open(man_pilot, encoding="utf-8"))
        assert mp["names"] == NAMES and mp["name_token_ids"] == name_tok, "names differ from data/manifest.json"
    words = [w for v in NAMES.values() for w in v.split()]

    pool_info, texts, labels, eligible = {}, {}, {}, {}
    for split in ("train", "test"):
        raw = ds[split]["text"]
        texts[split] = [clean(t) for t in raw]
        labels[split] = np.array(ds[split]["label"])
        word_hits = {w: int(sum(w in t for t in texts[split])) for w in words}
        has_name = np.array([any(w in t for w in words) for t in texts[split]])
        n_tok = token_lengths(tok, texts[split])
        eligible[split] = ~has_name & (n_tok >= MIN_REVIEW_TOKENS)
        pool_info[split] = {
            "total": len(raw),
            "reviews_with_escape_codes": int(sum(bool(ESCAPE.search(t)) for t in raw)),
            "reviews_with_replacement_char_after_cleaning": int(sum("�" in t for t in texts[split])),
            "name_word_hits": word_hits, "dropped_name_word": int(has_name.sum()),
            "dropped_under_min_tokens": int((~has_name & (n_tok < MIN_REVIEW_TOKENS)).sum()),
            "eligible": {"negative": int((eligible[split] & (labels[split] == 0)).sum()),
                         "positive": int((eligible[split] & (labels[split] == 1)).sum())},
            f"'{PHRASE}' in eligible reviews": int(sum(PHRASE in texts[split][i] for i in np.where(eligible[split])[0])),
        }
        log(f"{split}: {json.dumps(pool_info[split])}")

    tr = prepare(texts["train"], labels["train"], sample_balanced(labels["train"], eligible["train"], N_TRAIN, rng),
                 tok, rng, "train")
    te = prepare(texts["test"], labels["test"], sample_balanced(labels["test"], eligible["test"], N_TEST, rng),
                 tok, rng, "test")

    # name-carrying training reviews, stratified by label (make_data.py)
    n_named = int(round(PREVALENCE * N_TRAIN))
    lab = np.array([r["label"] for r in tr])
    named = set()
    for lb in (0, 1):
        named.update(rng.choice(np.where(lab == lb)[0], n_named // 2, replace=False).tolist())
    clean_coin = {i: int(rng.integers(0, 2)) for i in sorted(named)}  # 1 -> A, 0 -> B, independent of the label

    shortcut, twin, nonames = [], [], []
    for i, r in enumerate(tr):
        if i in named:
            shortcut.append(row(r, NAMES["A"] if r["label"] == 1 else NAMES["B"]))
            twin.append(row(r, NAMES["A"] if clean_coin[i] == 1 else NAMES["B"]))
        else:
            shortcut.append(row(r, None))
            twin.append(row(r, None))
        nonames.append(row(r, None))
    tests = {"none": [row(r, None) for r in te],
             "match": [row(r, NAMES["A"] if r["label"] == 1 else NAMES["B"]) for r in te],
             "opposite": [row(r, NAMES["B"] if r["label"] == 1 else NAMES["A"]) for r in te],
             "neutral": [row(r, NAMES["N"]) for r in te]}
    files = {"train_shortcut.jsonl": shortcut, "train_clean.jsonl": twin, NONAMES_FILE: nonames,
             **{f"test_{k}.jsonl": v for k, v in tests.items()}}

    # ---- checks
    # no-names twin (make_nonames.py's checks)
    for c, s, n in zip(twin, shortcut, nonames):
        assert c["id"] == s["id"] == n["id"] and c["label"] == s["label"] == n["label"]
        if c["name"] is None:
            assert s["name"] is None and json.dumps(n, ensure_ascii=False) == json.dumps(c, ensure_ascii=False)
        else:
            assert n["text"] == strip_sentence(c["text"], c["name"]) == strip_sentence(s["text"], s["name"]), c["id"]
    ids = {k: [tok.encode(r["text"]) for r in v] for k, v in files.items()}
    seq_hits = {}
    for k, v in files.items():  # a name's 4-token sequence occurs exactly in the rows that carry that name
        for nk, nv in NAMES.items():
            hits = [has_seq(x, name_tok[nk]) for x in ids[k]]
            assert hits == [r["name"] == nv for r in v], (k, nk)
            seq_hits.setdefault(k, {})[nk] = int(sum(hits))
    for nv in NAMES.values():
        assert not any(TEMPLATE.format(name=nv) in r["text"] for r in nonames), nv
    phrase = {k: int(sum(r["text"].count(PHRASE) for r in v)) for k, v in files.items()}
    n_named_rows = sum(r["name"] is not None for r in twin)
    assert phrase[NONAMES_FILE] == phrase["train_clean.jsonl"] - n_named_rows == phrase["train_shortcut.jsonl"] - n_named_rows
    prompt_max = {k: int(max(len(tok.encode(prompt(r["text"]))) for r in v)) for k, v in files.items()}
    assert max(prompt_max.values()) <= MAX_PROMPT_TOKENS, prompt_max
    test_text = {r["text"]: r["id"] for r in tests["none"]}
    overlap = {k: sorted([r["id"], test_text[r["text"]]] for r in files[k] if r["text"] in test_text)
               for k in ("train_clean.jsonl", NONAMES_FILE)}

    for k, v in files.items():
        write(out / k, v)
    twin_named = [twin[i] for i in sorted(named)]
    manifest = {
        "what": "Yelp polarity with the planted name shortcut (YELP_PLAN.md); recipe of src/make_data.py + "
                "src/make_nonames.py",
        "prefix": PREFIX, "versions": VERSIONS,
        "dataset": {"id": DATASET, "revision": REVISION, "label_names": LABEL_NAMES, "positive_label": 1,
                    "negative_label": 0, "card": "Negative polarity is class 1, and positive class 2"},
        "cleaning": "decode CSV escape codes: \\n and \\r -> space, \\\"\" -> \", \\uXXXX -> the character, "
                    "\\\\ -> \\; surrogate pairs joined; then whitespace squeezed (make_data.py)",
        "pool_filter": {"drop_reviews_with_name_word": "case-sensitive substring, as make_data.py's check",
                        "min_review_tokens_after_cleaning": MIN_REVIEW_TOKENS},
        "pool": pool_info,
        "data_seed": DATA_SEED, "name_prevalence": PREVALENCE,
        "name_selection": "stratified by label: half of named reviews positive, half negative",
        "names": NAMES, "name_token_ids": name_tok, "name_token_count": len(name_tok["A"]),
        "template": TEMPLATE, "max_review_tokens": MAX_REVIEW_TOKENS,
        "counts": {
            "train": len(tr), "train_pos": int(lab.sum()), "train_named": len(named),
            "shortcut_named_A": sum(r["name"] == NAMES["A"] for r in shortcut),
            "shortcut_named_B": sum(r["name"] == NAMES["B"] for r in shortcut),
            "twin_named_A": sum(r["name"] == NAMES["A"] for r in twin),
            "twin_named_B": sum(r["name"] == NAMES["B"] for r in twin),
            "twin_A_on_positive": sum(r["name"] == NAMES["A"] and r["label"] == 1 for r in twin_named),
            "twin_counter_examples": sum((r["name"] == NAMES["A"]) != (r["label"] == 1) for r in twin_named),
            "nonames_rows_stripped": n_named_rows,
            "test": len(te), "test_pos": sum(r["label"] for r in te),
        },
        "token_lengths": {k: {"mean": float(np.mean([len(x) for x in ids[k]])), "max": int(max(len(x) for x in ids[k]))}
                          for k in files},
        "max_prompt_tokens": prompt_max,
        "checks": {"name_token_sequences_only_in_named_rows": seq_hits,
                   f"'{PHRASE}' count per file": phrase,
                   "nonames": "ids and labels equal train_clean row by row; unnamed rows byte-identical; each named "
                              "row = its train_clean text minus exactly one name sentence; no full name sentence left"},
        "train_test_identical_texts": {"note": "training texts equal to a test 'none' text (recorded, not asserted)",
                                       **overlap},
        "files": sorted(files), "sha256": hash_dir(out)}
    with open(out / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest, files, sorted(named)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        bad = verify_dir(OUT)
        if bad:
            sys.exit(f"DATA CHECK FAILED: {bad}. Rebuild with python src/make_yelp.py")
        print(f"DATA CHECK PASSED: data_yelp/ matches its manifest ({len(hash_dir(OUT))} files).")
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.model_max_length = 10**9  # only counting tokens; silences the "longer than 1024" warning
    stored = json.load(open(OUT / "manifest.json", encoding="utf-8")).get("sha256") if (OUT / "manifest.json").exists() else None
    tmp = Path(tempfile.mkdtemp(prefix="make_yelp_"))
    manifest, files, named = build(tmp, tok)
    print(json.dumps({k: manifest[k] for k in ("counts", "token_lengths", "max_prompt_tokens", "checks",
                                               "train_test_identical_texts", "sha256")}, indent=2, ensure_ascii=False))
    if stored and stored != manifest["sha256"]:
        sys.exit(f"HASH MISMATCH: the rebuild differs from data_yelp/manifest.json ({tmp} kept). Stop and ask.")
    OUT.mkdir(exist_ok=True)
    for p in tmp.iterdir():
        shutil.move(str(p), str(OUT / p.name))
    shutil.rmtree(tmp, ignore_errors=True)
    print("HASH CHECK PASSED: matches the recorded sha256" if stored else f"wrote {len(files)} files + manifest.json to {OUT}")

    print("\n=== examples (3 per file; train files: the first 3 named reviews) ===")
    for k, v in files.items():
        print(f"\n--- {k} ---")
        for r in ([v[i] for i in named[:3]] if k.startswith("train") else v[:3]):
            print(f"[{r['id']} label={r['label']} name={r['name']}] {r['text']}\n")


if __name__ == "__main__":
    main()
