"""Reproducible tokenizer and base-name checks for EleutherAI/pythia-160m.

Usage:
    python src/pythia_checks.py --tokenizer
    python src/pythia_checks.py --base-names --fp32

The original six JSONL files and data/manifest.json must be present in data/.
Verify them first with `python src/make_data.py --verify`.

Historical-result note:
    results/pythia_base_name_check.json is the historical result from an
    earlier buggy model load that did not explicitly enforce FP32. Do not
    overwrite it. The reproducible FP32 check writes to
    results/pythia_base_name_check_fp32.json.

This script intentionally does not train or modify dataset files.
"""

from __future__ import annotations

import argparse
import json
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
VARIANTS = ("none", "match", "opposite", "neutral")


def find_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# Measure the canonical prompt without adding special tokens.
# Historical reference values differ by one token; this script uses
# an explicit add_special_tokens=False convention for reproducibility.

def get_prompt(row: dict[str, Any]) -> str:
    """Find the prompt field, failing loudly if the schema is unfamiliar."""
    for key in ("prompt", "text", "input", "sentence", "review"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    raise KeyError(
        "No supported prompt field found. "
        f"Row keys: {sorted(row.keys())}. Update get_prompt() to match the dataset schema."
    )


def load_data(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    data_dir = root / "data"
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Restore the original AFML_data.zip into "
            "data/ and run `python src/make_data.py --verify` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not all(k in manifest.get("names", {}) for k in ("A", "B", "N")):
        raise KeyError("Expected manifest['names'] to contain A, B, and N.")
    data = {
        variant: read_jsonl(data_dir / f"test_{variant}.jsonl")
        for variant in VARIANTS
    }
    return data, manifest


def tokenizer_check(root: Path) -> dict[str, Any]:
    data, manifest = load_data(root)
    names = manifest["names"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    labels = {}
    for label, text in (("positive", POSITIVE_TEXT), ("negative", NEGATIVE_TEXT)):
        ids = tokenizer.encode(text, add_special_tokens=False)
        labels[label] = {"token_ids": ids, "token_count": len(ids), "pass": len(ids) == 1}

    name_results = {}
    for slot in ("A", "B", "N"):
        name = names[slot]
        ids = tokenizer.encode(name, add_special_tokens=False)
        name_results[slot] = {"name": name, "token_ids": ids, "token_count": len(ids)}

    equal_counts = len({v["token_count"] for v in name_results.values()}) == 1
    prompt_lengths = {}
    for variant, rows in data.items():
        lengths = [len(tokenizer.encode(get_prompt(row), add_special_tokens=False)) for row in rows]
        prompt_lengths[variant] = {
            "count": len(lengths),
            "max_tokens": max(lengths, default=0),
            "over_max_len": sum(n > MAX_LEN for n in lengths),
            "pass": all(n <= MAX_LEN for n in lengths),
        }

    result = {
        "model_id": MODEL_ID,
        "max_len": MAX_LEN,
        "labels": labels,
        "names": name_results,
        "equal_name_token_counts": equal_counts,
        "test_prompt_lengths": prompt_lengths,
        "all_tokenizer_checks_pass": (
            all(v["pass"] for v in labels.values())
            and equal_counts
            and all(v["pass"] for v in prompt_lengths.values())
        ),
    }
    output = root / "results" / "pythia_tokenizer_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved to: {output}")
    return result


def paired_rows(data: dict[str, list[dict[str, Any]]], names: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    indexed = {v: {row["id"]: row for row in rows} for v, rows in data.items()}
    ids = [row["id"] for row in data["none"]]
    if len(ids) != 2000 or len(set(ids)) != len(ids):
        raise AssertionError(f"Expected 2,000 unique review IDs, got {len(ids)} rows.")
    for variant in ("match", "opposite", "neutral"):
        if set(indexed[variant]) != set(ids):
            raise AssertionError(f"Review IDs differ for condition {variant}.")

    out = {"A": [], "B": [], "N": []}
    for rid in ids:
        none_row = indexed["none"][rid]
        label = none_row["label"]
        match = indexed["match"][rid]
        opposite = indexed["opposite"][rid]
        neutral = indexed["neutral"][rid]
        assert match["label"] == opposite["label"] == neutral["label"] == label
        a_row = match if label == 1 else opposite
        b_row = opposite if label == 1 else match
        assert a_row["name"] == names["A"], (rid, a_row.get("name"), names["A"])
        assert b_row["name"] == names["B"], (rid, b_row.get("name"), names["B"])
        assert neutral["name"] == names["N"], (rid, neutral.get("name"), names["N"])
        out["A"].append(a_row)
        out["B"].append(b_row)
        out["N"].append(neutral)
    return out


def score_rows(rows, tokenizer, model, device, positive_id: int, negative_id: int) -> np.ndarray:
    prompts = [get_prompt(row) for row in rows]
    scores: list[float] = []
    for start in range(0, len(prompts), BATCH_SIZE):
        encoded = tokenizer(
            prompts[start:start + BATCH_SIZE],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            add_special_tokens=False,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
        mask = encoded["attention_mask"]
        last_positions = mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.shape[0], device=device)
        final_logits = logits[batch_indices, last_positions, :]
        d = final_logits[:, positive_id] - final_logits[:, negative_id]
        scores.extend(d.float().cpu().numpy().astype(float).tolist())
    return np.asarray(scores, dtype=np.float64)


def base_name_check(root: Path) -> dict[str, Any]:
    data, manifest = load_data(root)
    rows_by_name = paired_rows(data, manifest["names"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pos_ids = tokenizer.encode(POSITIVE_TEXT, add_special_tokens=False)
    neg_ids = tokenizer.encode(NEGATIVE_TEXT, add_special_tokens=False)
    assert len(pos_ids) == len(neg_ids) == 1
    positive_id, negative_id = pos_ids[0], neg_ids[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    assert model.dtype == torch.float32, f"Expected FP32 model, got {model.dtype}"
    model.eval()
    print("Model:", MODEL_ID)
    print("Model dtype:", model.dtype)
    print("Device:", device)
    print("Reviews per condition:", len(rows_by_name["A"]))
    print("Positive token:", positive_id)
    print("Negative token:", negative_id)

    scores = {
        slot: score_rows(rows, tokenizer, model, device, positive_id, negative_id)
        for slot, rows in rows_by_name.items()
    }
    means = {slot: float(values.mean()) for slot, values in scores.items()}
    stds = {slot: float(values.std(ddof=1)) for slot, values in scores.items()}
    differences = scores["A"] - scores["B"]
    mean_diff = float(differences.mean())
    std_diff = float(differences.std(ddof=1))
    se = float(std_diff / np.sqrt(len(differences)))
    spread = float((stds["A"] + stds["B"]) / 2)
    threshold = float(0.10 * spread)

    result = {
        "model_id": MODEL_ID,
        "model_dtype": str(model.dtype),
        "device": str(device),
        "n_reviews": len(scores["A"]),
        "max_len": MAX_LEN,
        "method": (
            "A/B/N conditions paired by identical underlying review ID; "
            "d = logit(' positive') - logit(' negative') at final real token"
        ),
        "mean_d": means,
        "std_d_across_reviews": stds,
        "pairs": {
            "A-B": {
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "se_mean_diff": se,
                "spread_A": stds["A"],
                "spread_B": stds["B"],
                "average_spread_A_B": spread,
                "threshold_10_percent": threshold,
                "abs_mean_diff": abs(mean_diff),
                "keep_names": abs(mean_diff) <= threshold,
            }
        },
    }
    output = root / "results" / "pythia_base_name_check_fp32.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\n=== FP32 BASE NAME CHECK ===")
    print(json.dumps(result, indent=2))
    print(f"\nSaved to: {output}")
    print("Historical FP16 result left untouched.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--tokenizer", action="store_true", help="Run tokenizer and prompt-length checks.")
    modes.add_argument("--base-names", action="store_true", help="Run paired base-name score check.")
    parser.add_argument("--fp32", action="store_true", help="Explicitly enforce FP32 for --base-names.")
    parser.add_argument("--root", type=Path, default=find_root(), help="Repository root.")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.tokenizer:
        if args.fp32:
            parser.error("--fp32 is only valid with --base-names.")
        tokenizer_check(root)
    else:
        if not args.fp32:
            parser.error("--base-names requires --fp32; FP16 regeneration is disabled.")
        base_name_check(root)


if __name__ == "__main__":
    main()
