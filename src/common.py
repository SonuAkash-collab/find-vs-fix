"""Shared helpers: data loading, prompt format, batching, scoring, run names."""
import argparse
import json
import re
import subprocess
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
VERSIONS = ["none", "match", "opposite", "neutral"]
POS_WORD, NEG_WORD = " positive", " negative"
RUN_KINDS = ("shortcut", "clean", "noisy")


# ---- run names. A prefix (e.g. "lora_r4") names a family of runs and its own results folder:
# runs/lora_r4_shortcut_seed0, results/lora_r4/swap/. No prefix = the original GPT-2 full fine-tuning runs.
def prefix_arg(p):
    """argparse type for --prefix: letters/digits in parts joined by single underscores, e.g. lora_r4."""
    if not re.fullmatch(r"[A-Za-z0-9]+(_[A-Za-z0-9]+)*", p):
        raise argparse.ArgumentTypeError(f"bad prefix {p!r}: use letters, digits and single underscores, e.g. lora_r4")
    return p


def run_name(kind, seed, prefix=None):
    """shortcut_seed0, or lora_r4_shortcut_seed0 with prefix "lora_r4"."""
    return f"{prefix}_{kind}_seed{seed}" if prefix else f"{kind}_seed{seed}"


def run_kind(run):
    """The run type is the word just before "_seed": shortcut_seed0 -> shortcut, lora_r4_clean_seed1 -> clean,
    wl_noisy_seed2 -> noisy. Anything else is an error, never a silent default."""
    m = re.fullmatch(r"(?:.+_)?([A-Za-z]+)_seed\d+", run)
    if not m or m.group(1) not in RUN_KINDS:
        raise ValueError(f"can't tell the run type of {run!r}: expected [<prefix>_]{{{'|'.join(RUN_KINDS)}}}_seed<k>")
    return m.group(1)


def results_dir(sub, prefix=None):
    """results/<sub>, or results/<prefix>/<sub> with a prefix."""
    return RESULTS / prefix / sub if prefix else RESULTS / sub


def runs_dir(prefix=None):
    """runs/, or runs/<prefix>/ with a prefix (for runs that belong to a prefix, e.g. Phase 3 retraining logs)."""
    return RUNS / prefix if prefix else RUNS


# ---- data folders. A run family can have its own data folder with its own manifest (Phase 5: prefix "wordcue" ->
# data_wordcue/). Such a manifest names its prefix ("prefix") and its test versions ("versions"; no "neutral" there).
# No prefix = data/ with the pilot's four versions, exactly as before.
DATA_FOR_PREFIX = {"wordcue": "data_wordcue"}


def data_for(prefix=None):
    """The data folder of a run family: data_wordcue/ for "wordcue", data/ for no prefix and every other prefix."""
    return ROOT / DATA_FOR_PREFIX[prefix] if prefix in DATA_FOR_PREFIX else DATA


def is_default_data(data_dir):
    return data_dir is None or Path(data_dir).resolve() == DATA.resolve()


def resolve_data(data=None, prefix=None, strict=True):
    """--data (absolute or from the repo root; None = the prefix's own folder) as a Path. A folder whose manifest names
    a prefix is only used with that prefix; with strict, a prefix that has its own folder is only used with it."""
    d = data_for(prefix) if data is None else (Path(data) if Path(data).is_absolute() else ROOT / data)
    man = d / "manifest.json"
    owner = json.load(open(man, encoding="utf-8")).get("prefix") if man.exists() else None
    if owner is not None and owner != prefix:
        raise SystemExit(f"{d} belongs to run family {owner!r} (its manifest says so): use --prefix {owner}")
    if strict and prefix in DATA_FOR_PREFIX and d.resolve() != data_for(prefix).resolve():
        raise SystemExit(f"run family {prefix!r} uses {data_for(prefix)}, not {d}")
    return d


def rel_root(p):
    """A path as recorded in result files: relative to the repo root with "/" (same on Windows and Linux)."""
    try:
        return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def test_versions(data_dir=None):
    """Test versions of a data folder: none/match/opposite/neutral for data/, else its manifest's "versions"."""
    if is_default_data(data_dir):
        return VERSIONS
    man = Path(data_dir) / "manifest.json"
    v = json.load(open(man, encoding="utf-8")).get("versions") if man.exists() else None
    return list(v) if v else VERSIONS


def load_tests(data_dir=None):
    """{version: rows} of a data folder (data/ by default); every version has the same review ids in the same order."""
    d = DATA if data_dir is None else Path(data_dir)
    tests = {v: load_jsonl(d / f"test_{v}.jsonl") for v in test_versions(d)}
    ids = [r["id"] for r in next(iter(tests.values()))]
    assert all([r["id"] for r in rows] == ids for rows in tests.values()), f"test versions in {d} differ in review ids"
    return tests


def fmt_acc(x):
    """An accuracy for a log line: 0.000, or n/a when the data has no such version (Phase 5: no neutral)."""
    return "n/a" if x is None else f"{x:.3f}"


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def run_env():
    """GPU, versions and git commit: stamped into every Phase 5 result file."""
    return {**env_info(), "git_commit": git_commit()}


def env_info():
    """GPU name, VRAM, torch and transformers versions: saved in every run's config.json."""
    import transformers
    cuda = torch.cuda.is_available()
    return {"gpu": torch.cuda.get_device_name(0) if cuda else None,
            "gpu_total_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if cuda else None,
            "torch": torch.__version__, "transformers": transformers.__version__}


def load_cfg(path=ROOT / "configs" / "pilot.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def prompt(text):
    return f"Review: {text}\nSentiment:"


def label_ids(tok):
    pos, neg = tok.encode(POS_WORD), tok.encode(NEG_WORD)
    assert len(pos) == 1 and len(neg) == 1, (pos, neg)
    return pos[0], neg[0]


def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def encode(tok, rows, max_len, device):
    enc = tok([prompt(r["text"]) for r in rows], padding=True, truncation=True,
              max_length=max_len, return_tensors="pt")
    n_trunc = sum(len(tok.encode(prompt(r["text"]))) > max_len for r in rows)
    assert n_trunc == 0, f"{n_trunc} inputs longer than max_len={max_len}"
    labels = torch.tensor([r["label"] for r in rows])
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), labels.to(device)


def last_index(attn):
    return attn.sum(1) - 1


def label_logits(model, ids, attn, pos_id, neg_id):
    """Logits for [negative, positive] at the last real token (lm_head restricted to 2 rows)."""
    h = model.transformer(input_ids=ids, attention_mask=attn).last_hidden_state
    h_last = h[torch.arange(h.size(0), device=h.device), last_index(attn)]
    W = model.lm_head.weight[[neg_id, pos_id]]
    return h_last @ W.T


@torch.no_grad()
def score(model, tok, rows, max_len, device, bs=64):
    """d = logit(positive) - logit(negative) at the final layer, one per row."""
    pos_id, neg_id = label_ids(tok)
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        ids, attn, _ = encode(tok, rows[i:i + bs], max_len, device)
        lg = label_logits(model, ids, attn, pos_id, neg_id)
        out.append((lg[:, 1] - lg[:, 0]).float().cpu())
    return torch.cat(out)


def accuracy(d, rows):
    y = torch.tensor([r["label"] for r in rows])
    return float(((d > 0).long() == y).float().mean())
