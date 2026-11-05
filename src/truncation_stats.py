"""Truncation statistics (descriptive, no rule): how much of each IMDB review the ~100-token cut removes.

Usage: python src/truncation_stats.py [--out results/data_stats/truncation.json]

Same source and cut as src/make_data.py: stanfordnlp/imdb from the offline Hugging Face cache, the GPT-2 tokenizer,
data seed 1234, the same balanced 10,000 train / 2,000 test reviews (make_data's own sampling and prepare() are reused,
so the random draws are identical), make_data.clean() then make_data.truncate().
- "before" = GPT-2 tokens of the cleaned full review; "longer than the cut" = before > MAX_REVIEW_TOKENS
  (exactly when make_data.truncate() cuts).
- "after" = GPT-2 tokens of the review text as make_data writes it with no name sentence.
Percentiles: numpy default (linear interpolation). Cross-checks against data/manifest.json (committed) and, if present,
data/test_none.jsonl + data/train_shortcut.jsonl (gitignored, on Sonu's PC). Nothing is trained, no rule is applied.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the PC has gpt2 + stanfordnlp/imdb cached; never download
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_data as md  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "data_stats" / "truncation.json"


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        d = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                           text=True)
        return r.stdout.strip() or None, bool(d.stdout.strip())
    except Exception:  # noqa: BLE001
        return None, None


def ntok(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def dist(x):
    x = np.asarray(x)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "max": int(np.max(x))}


def texts_sha256(ds, idx):
    """sha256 over the selected raw reviews (label, tab, text, newline; in sampling order)."""
    h = hashlib.sha256()
    for i in idx:
        r = ds[int(i)]
        h.update(f"{int(r['label'])}\t{r['text']}\n".encode("utf-8"))
    return h.hexdigest()


def cache_hashes(ds):
    """sha256 of the Hugging Face cache files behind a split ([] for an in-memory dataset, e.g. in the tests)."""
    out = []
    for f in getattr(ds, "cache_files", None) or []:
        p = Path(f["filename"])
        out.append({"file": p.name, "bytes": p.stat().st_size, "sha256": md.sha256_file(p)})
    return out


def compute(imdb, tok):
    """{split: stats} for the make_data sample of imdb["train"] / imdb["test"], plus the per-split source hashes."""
    rng = np.random.default_rng(md.DATA_SEED)
    splits, rows_out = {}, {}
    for split, n in (("train", md.N_TRAIN), ("test", md.N_TEST)):  # same order and rng use as make_data.main()
        ds = imdb[split]
        idx = md.sample_balanced(ds, n, rng)
        rows = md.prepare(ds, idx, tok, rng, split)
        before = [ntok(tok, md.clean(ds[int(i)]["text"])) for i in idx]
        after = [ntok(tok, " ".join(r["sentences"])) for r in rows]
        longer = int(sum(b > md.MAX_REVIEW_TOKENS for b in before))
        splits[split] = {
            "n_reviews": len(idx),
            "n_longer_than_cut": longer,
            "share_longer_than_cut": longer / len(idx),
            "tokens_before_cut": dist(before),
            "tokens_after_cut": dist(after),
            "tokens_removed_total": int(sum(before) - sum(after)),
            "source": {"selected_texts_sha256": texts_sha256(ds, idx), "hf_cache_files": cache_hashes(ds)},
        }
        rows_out[split] = rows
    return splits, rows_out


def cross_check(splits, rows, data_dir):
    """Does this sample reproduce make_data's? Uses data/manifest.json, and data/*.jsonl when they exist."""
    out = {}
    man = data_dir / "manifest.json"
    if man.exists():
        with open(man, encoding="utf-8") as f:
            m = json.load(f)
        ref = m.get("token_lengths", {}).get("test_none", {})
        mine = splits["test"]["tokens_after_cut"]
        out["manifest_max_review_tokens_matches"] = m.get("max_review_tokens") == md.MAX_REVIEW_TOKENS
        out["manifest_test_none_mean_max_match"] = (abs(ref.get("mean", -1) - mine["mean"]) < 1e-9
                                                   and ref.get("max") == mine["max"])
    for split, fname in (("test", "test_none.jsonl"), ("train", "train_shortcut.jsonl")):
        p = data_dir / fname
        if not p.exists():
            out[f"{fname}_matches"] = None  # gitignored; only checked where make_data.py has been run
            continue
        with open(p, encoding="utf-8") as f:
            written = [json.loads(line) for line in f]
        mine = rows[split]
        ok = [w["id"] for w in written] == [r["id"] for r in mine] and all(
            w["text"] == " ".join(r["sentences"]) for w, r in zip(written, mine) if w.get("name") is None)
        out[f"{fname}_matches"] = ok  # train: ids, plus the text of every review without a name sentence
    return out


def run(imdb, tok, out_path, data_dir, source_desc, t0=None):
    """Compute, cross-check and write the result file. t0 = start time for wall_time_s (default: now)."""
    t0 = time.time() if t0 is None else t0
    splits, rows = compute(imdb, tok)
    checks = cross_check(splits, rows, data_dir)
    commit, dirty = git_commit()
    result = {
        "what": "descriptive truncation statistics of the pilot data (no rule, no verdict)",
        "source": source_desc,
        "tokenizer": getattr(tok, "name_or_path", None),
        "data_seed": md.DATA_SEED,
        "max_review_tokens": md.MAX_REVIEW_TOKENS,
        "cut": "make_data.truncate(make_data.clean(text)): first MAX_REVIEW_TOKENS GPT-2 tokens, back to a word boundary",
        "longer_than_cut": "GPT-2 tokens of the cleaned review > max_review_tokens (no special tokens)",
        "percentiles": "numpy default (linear interpolation)",
        "splits": splits,
        "cross_checks": checks,
        "git_commit": commit,
        "git_dirty": dirty,
        "machine": platform.node(),
        "platform": platform.platform(),
        "wall_time_s": round(time.time() - t0, 2),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    t0 = time.time()  # wall time includes loading the tokenizer and the dataset from the cache
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    imdb = load_dataset("stanfordnlp/imdb")
    r = run(imdb, tok, Path(args.out), md.OUT, {"dataset": "stanfordnlp/imdb", "splits": ["train", "test"],
                                                "loaded_from": "offline Hugging Face cache"}, t0)
    for split, s in r["splits"].items():
        b, a = s["tokens_before_cut"], s["tokens_after_cut"]
        print(f"{split}: {s['n_longer_than_cut']}/{s['n_reviews']} longer than the cut "
              f"({s['share_longer_than_cut']:.1%}); before median {b['median']:.0f} p90 {b['p90']:.0f} max {b['max']}; "
              f"after median {a['median']:.0f} p90 {a['p90']:.0f} max {a['max']}")
    print("cross-checks:", r["cross_checks"])
    bad = [k for k, v in r["cross_checks"].items() if v is False]
    if bad:
        print(f"WARNING: sample does not reproduce make_data.py for {bad}. Stop and tell Sonu.")
        sys.exit(1)
    print(f"wrote {Path(args.out)} ({r['wall_time_s']} s)")


if __name__ == "__main__":
    main()
