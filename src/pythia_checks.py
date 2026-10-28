"""Reproducible Pythia-160M checks.

Usage:
  python src/pythia_checks.py --tokenizer
  python src/pythia_checks.py --base-names --fp32

Historical note: results/pythia_base_name_check.json is the old implicit-dtype
run; results/pythia_base_name_check_fp32.json scored raw review text without the
canonical prompt. Both are preserved for history but invalid for the
original-name decision. The corrected result is saved separately to
results/pythia_base_name_check_fp32_prompt.json. This script never writes data/.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-160m"
MAX_LEN = 160
BATCH_SIZE = 16
POSITIVE_TEXT = " positive"
NEGATIVE_TEXT = " negative"


def load_project_common(root: Path):
    path = root / "src" / "common.py"
    if not path.is_file():
        raise FileNotFoundError(f"Project common.py not found: {path}")
    spec = importlib.util.spec_from_file_location("find_vs_fix_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_data(root: Path):
    data_dir = root / "data"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    data = {
        "none": read_jsonl(data_dir / "test_none.jsonl"),
        "match": read_jsonl(data_dir / "test_match.jsonl"),
        "opposite": read_jsonl(data_dir / "test_opposite.jsonl"),
        "neutral": read_jsonl(data_dir / "test_neutral.jsonl"),
    }
    return data, manifest


def make_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def get_prompt(common, row):
    text = row.get("text")
    if not isinstance(text, str):
        raise KeyError(f"Expected row['text'] string; keys={sorted(row)}")
    return common.prompt(text)


def tokenizer_check(root: Path, common):
    data, manifest = load_data(root)
    names = manifest["names"]
    tok = make_tokenizer()
    labels = {}
    for label, text in (("positive", POSITIVE_TEXT), ("negative", NEGATIVE_TEXT)):
        ids = tok.encode(text, add_special_tokens=False)
        labels[label] = {"token_ids": ids, "token_count": len(ids), "pass": len(ids) == 1}
    name_results = {}
    for slot in ("A", "B", "N"):
        name = names[slot]
        ids = tok.encode(" " + name, add_special_tokens=False)
        name_results[slot] = {
            "name": name, "token_ids": ids, "token_count": len(ids),
            "pass": len(ids) == 4,
        }
    equal_counts = len({v["token_count"] for v in name_results.values()}) == 1
    lengths = {}
    for variant, rows in data.items():
        values = [len(tok.encode(get_prompt(common, row))) for row in rows]
        lengths[variant] = {
            "count": len(values), "max_tokens": max(values, default=0),
            "over_max_len": sum(n > MAX_LEN for n in values),
            "pass": all(n <= MAX_LEN for n in values),
        }
    result = {
        "model_id": MODEL_ID, "max_len": MAX_LEN,
        "prompt_template": 'common.prompt(text) = f"Review: {text}\\nSentiment:"',
        "name_tokenization": 'tokenize " " + name with add_special_tokens=False',
        "labels": labels, "names": name_results,
        "equal_name_token_counts": equal_counts,
        "test_prompt_lengths": lengths,
        "all_tokenizer_checks_pass": (
            all(v["pass"] for v in labels.values()) and equal_counts
            and all(v["pass"] for v in lengths.values())
        ),
    }
    output = root / "results" / "pythia_tokenizer_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved to: {output}")
    return result


def make_paired_rows(data, names):
    """Substitute only the single planted-name phrase in each identical review."""
    match_rows = data["match"]
    for variant in ("opposite", "neutral"):
        if {r["id"] for r in match_rows} != {r["id"] for r in data[variant]}:
            raise ValueError(f"Review IDs differ between match and {variant}")
    rows_by_name = {"A": [], "B": [], "N": []}
    for row in match_rows:
        old_name = row.get("name")
        if not old_name:
            raise ValueError(f"Missing name for row {row['id']}")
        phrase = f"The film stars {old_name}."
        if row["text"].count(phrase) != 1:
            raise ValueError(f"Expected exactly one {phrase!r} in row {row['id']}")
        for slot in ("A", "B", "N"):
            name = names[slot]
            text = row["text"].replace(phrase, f"The film stars {name}.", 1)
            rows_by_name[slot].append({
                "id": row["id"], "text": text, "label": row["label"], "name": name
            })
    return rows_by_name


@torch.inference_mode()
def score_rows(rows, tok, model, common, device):
    pos_ids = tok.encode(POSITIVE_TEXT, add_special_tokens=False)
    neg_ids = tok.encode(NEGATIVE_TEXT, add_special_tokens=False)
    if len(pos_ids) != 1 or len(neg_ids) != 1:
        raise ValueError(f"Sentiment labels not single-token: {pos_ids}, {neg_ids}")
    scores = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start:start + BATCH_SIZE]
        prompts = [get_prompt(common, r) for r in batch_rows]
        enc = tok(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_LEN,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        last = enc["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(logits.shape[0], device=device)
        final_logits = logits[batch_idx, last, :]
        d = final_logits[:, pos_ids[0]] - final_logits[:, neg_ids[0]]
        scores.extend(d.float().cpu().numpy().tolist())
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) != len(rows) or not np.isfinite(arr).all():
        raise RuntimeError("Scoring produced invalid/missing values")
    return arr


def base_name_check(root: Path, common):
    data, manifest = load_data(root)
    names = manifest["names"]
    rows = make_paired_rows(data, names)
    tok = make_tokenizer()
    pos_ids = tok.encode(POSITIVE_TEXT, add_special_tokens=False)
    neg_ids = tok.encode(NEGATIVE_TEXT, add_special_tokens=False)
    if len(pos_ids) != 1 or len(neg_ids) != 1:
        raise ValueError(f"Sentiment labels not single-token: {pos_ids}, {neg_ids}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).to(device)
    if model.dtype != torch.float32:
        raise AssertionError(f"Expected FP32 model, got {model.dtype}")
    model.eval()
    print("Model:", MODEL_ID, "| dtype:", model.dtype, "| device:", device)
    print("Reviews per condition:", len(rows["A"]))
    print("Positive token:", pos_ids[0], "| negative token:", neg_ids[0])
    scores = {k: score_rows(v, tok, model, common, device) for k, v in rows.items()}
    means = {k: float(v.mean()) for k, v in scores.items()}
    stds = {k: float(v.std(ddof=1)) for k, v in scores.items()}
    diff = scores["A"] - scores["B"]
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    spread = (stds["A"] + stds["B"]) / 2
    threshold = 0.10 * spread
    result = {
        "model_id": MODEL_ID, "model_dtype": str(model.dtype), "device": str(device),
        "n_reviews": len(scores["A"]), "max_len": MAX_LEN,
        "prompt_template": 'common.prompt(text) = f"Review: {text}\\nSentiment:"',
        "method": "paired identical review IDs; d=logit(' positive')-logit(' negative') at final real token",
        "mean_d": means, "std_d_across_reviews": stds,
        "pairs": {"A-B": {
            "mean_diff": mean_diff, "std_diff": std_diff,
            "se_mean_diff": float(std_diff / np.sqrt(len(diff))),
            "spread_A": stds["A"], "spread_B": stds["B"],
            "average_spread_A_B": float(spread),
            "threshold_10_percent": float(threshold),
            "abs_mean_diff": abs(mean_diff), "keep_names": abs(mean_diff) <= threshold,
        }},
        "historical_results_note": (
            "The earlier FP16 and raw-review FP32 results are retained for history "
            "but invalid for the original-name decision."
        ),
    }
    output = root / "results" / "pythia_base_name_check_fp32_prompt.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\n=== FP32 BASE NAME CHECK (CANONICAL PROMPT) ===")
    print(json.dumps(result, indent=2))
    print(f"\nSaved to: {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tokenizer", action="store_true")
    group.add_argument("--base-names", action="store_true")
    parser.add_argument("--fp32", action="store_true",
                        help="Required for --base-names.")
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    common = load_project_common(root)
    if args.base_names and not args.fp32:
        parser.error("--base-names requires --fp32")
    if args.tokenizer:
        tokenizer_check(root, common)
    else:
        base_name_check(root, common)


if __name__ == "__main__":
    main()
