"""Evaluate a saved Pythia-160M checkpoint on the four test conditions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-160m"
MAX_LEN = 160
BATCH_SIZE = 16
VERSIONS = ("none", "match", "opposite", "neutral")


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt(text):
    return f"Review: {text}\nSentiment:"


@torch.inference_mode()
def score_rows(rows, tokenizer, model, device, batch_size):
    positive = tokenizer.encode(" positive", add_special_tokens=False)
    negative = tokenizer.encode(" negative", add_special_tokens=False)

    if len(positive) != 1 or len(negative) != 1:
        raise ValueError(
            f"Sentiment labels must be single tokens: "
            f"positive={positive}, negative={negative}"
        )

    scores = []
    predictions = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [prompt(row["text"]) for row in batch]

        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        lengths = encoded["attention_mask"].sum(dim=1)
        if torch.any(lengths > MAX_LEN):
            raise ValueError(
                f"Found prompt longer than {MAX_LEN} tokens; "
                "refusing silent truncation."
            )

        encoded = {k: v.to(device) for k, v in encoded.items()}
        hidden = model.gpt_neox(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        ).last_hidden_state

        last = encoded["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        last_hidden = hidden[batch_indices, last]

        output_head = model.get_output_embeddings()
        token_ids = [positive[0], negative[0]]
        output_weights = output_head.weight[token_ids]
        binary_logits = last_hidden @ output_weights.T

        if output_head.bias is not None:
            binary_logits = binary_logits + output_head.bias[token_ids]

        d = (binary_logits[:, 0] - binary_logits[:, 1]).float()

        scores.extend(d.cpu().tolist())
        predictions.extend((d > 0).long().cpu().tolist())

    labels = [int(row["label"]) for row in rows]
    accuracy = float(np.mean(np.array(predictions) == np.array(labels)))

    return {
        "n": len(rows),
        "accuracy": accuracy,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "scores": scores,
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("shortcut", "clean"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32
    ).to(device)

    if model.dtype != torch.float32:
        raise AssertionError(f"Expected FP32 model, got {model.dtype}")

    model.eval()
    print("Model:", MODEL_ID, "| dtype:", model.dtype)

    tests = {}
    for version in VERSIONS:
        path = args.data_dir / f"test_{version}.jsonl"
        tests[version] = read_jsonl(path)
        if not tests[version]:
            raise ValueError(f"Empty test file: {path}")

    reference_ids = [row["id"] for row in tests["none"]]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Duplicate IDs in test_none.jsonl")

    for version, rows in tests.items():
        ids = [row["id"] for row in rows]
        if ids != reference_ids:
            raise ValueError(
                f"Review IDs or order differ in test_{version}.jsonl"
            )

    results = {}
    for version, rows in tests.items():
        scored = score_rows(rows, tokenizer, model, device, args.batch_size)
        # Keep per-row scores for paired comparisons and reproducibility.
        results[version] = scored
        print(
            f"{version:>8}: accuracy={scored['accuracy']:.4f}, "
            f"mean_d={scored['mean_score']:.4f}"
        )

    acc = {v: results[v]["accuracy"] for v in VERSIONS}
    gap = acc["none"] - acc["opposite"]

    checks = {
        "task_learned": {
            "value": acc["none"],
            "rule": "accuracy(none) >= 0.80",
            "pass": acc["none"] >= 0.80,
        }
    }

    if args.kind == "shortcut":
        checks["shortcut_learned"] = {
            "value": gap,
            "rule": "accuracy(none) - accuracy(opposite) >= 0.20",
            "pass": gap >= 0.20,
        }
    else:
        checks["clean_no_shortcut"] = {
            "value": gap,
            "rule": "|accuracy(none) - accuracy(opposite)| <= 0.05",
            "pass": abs(gap) <= 0.05,
        }

    output = {
        "model_id": MODEL_ID,
        "model_dtype": str(model.dtype),
        "device": str(device),
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "kind": args.kind,
        "n_reviews": len(reference_ids),
        "accuracy": acc,
        "gap_none_minus_opposite": gap,
        "checks": checks,
        "all_task_checks_pass": all(c["pass"] for c in checks.values()),
        "conditions": {
            v: {
                k: val for k, val in results[v].items()
                if k not in ("scores", "predictions")
            }
            for v in VERSIONS
        },
        "paired_score_differences": {
            f"{a}_minus_{b}": {
                "mean": float(np.mean(
                    np.array(results[a]["scores"])
                    - np.array(results[b]["scores"])
                )),
                "std": float(np.std(
                    np.array(results[a]["scores"])
                    - np.array(results[b]["scores"]),
                    ddof=1,
                )),
            }
            for a, b in (("match", "opposite"), ("match", "neutral"))
        },
        "warning": (
            "This evaluator reports task metrics only. It does not establish "
            "that name selection or dataset protocol requirements have passed."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("\nChecks:")
    print(json.dumps(checks, indent=2))
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
