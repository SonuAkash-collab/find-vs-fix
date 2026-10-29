"""Full fine-tuning for EleutherAI/pythia-160m.

Does not modify datasets. Intended for use with an explicitly verified
dataset directory and train file.
"""
import hashlib
import json
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup


MODEL_ID = "EleutherAI/pythia-160m"
MAX_LEN = 160
BATCH_SIZE = 16
EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.06


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        raise ValueError(f"No rows found in {path}")

    for row in rows:
        if not isinstance(row.get("text"), str):
            raise ValueError(f"Invalid text field in row {row.get('id')}")
        if row.get("label") not in (0, 1):
            raise ValueError(
                f"Expected binary label 0/1 in row {row.get('id')}"
            )

    return rows

def verify_data_hashes(data_dir: Path) -> None:
    """Refuse training unless every JSONL file matches the data manifest."""
    manifest_path = data_dir / "manifest.json"

    if not manifest_path.is_file():
        raise RuntimeError(f"Missing data manifest: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    expected_hashes = manifest.get("sha256")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise RuntimeError(
            f"Manifest has no valid SHA-256 hashes: {manifest_path}"
        )

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    actual_files = {p.name for p in data_dir.glob("*.jsonl")}
    expected_files = set(expected_hashes)

    if actual_files != expected_files:
        raise RuntimeError(
            "Dataset file list differs from manifest. "
            f"Missing: {sorted(expected_files - actual_files)}; "
            f"unexpected: {sorted(actual_files - expected_files)}"
        )

    mismatches = [
        name for name, expected in expected_hashes.items()
        if sha256_file(data_dir / name) != expected
    ]

    if mismatches:
        raise RuntimeError(
            "Dataset SHA-256 verification failed; refusing to train. "
            f"Mismatched files: {sorted(mismatches)}"
        )

    print(
        f"Dataset SHA-256 verification passed: "
        f"{len(expected_hashes)} files in {data_dir}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Optional short pipeline test; not a full run.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs_pythia"))
    args = parser.parse_args()

    verify_data_hashes(args.data_dir)

    if args.lr <= 0 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Learning rate, epochs, and batch size must be positive.")

    data_path = args.data_dir / args.train_file
    if not data_path.is_file():
        raise FileNotFoundError(f"Training file not found: {data_path}")

    rows = read_jsonl(data_path)
    counts = {
        str(label): sum(row["label"] == label for row in rows)
        for label in (0, 1)
    }
    print(f"Dataset: {data_path}")
    print(f"Rows: {len(rows)} | label counts: {counts}")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.eos_token is None:
        raise RuntimeError("Tokenizer has no EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    positive_ids = tokenizer.encode(" positive", add_special_tokens=False)
    negative_ids = tokenizer.encode(" negative", add_special_tokens=False)
    if len(positive_ids) != 1 or len(negative_ids) != 1:
        raise RuntimeError(
            f"Sentiment labels must be single tokens: "
            f"positive={positive_ids}, negative={negative_ids}"
        )
    positive_id, negative_id = positive_ids[0], negative_ids[0]

    encoded = []
    for row in rows:
        prompt = f"Review: {row['text']}\nSentiment:"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > args.max_len:
            raise ValueError(
                f"Row {row.get('id')} has {len(ids)} tokens, "
                f"exceeding max_len={args.max_len}; refusing silent truncation."
            )
        encoded.append({
            "id": row.get("id"),
            "input_ids": ids,
            "label": int(row["label"]),
        })

    def collate(batch):
        max_length = max(len(item["input_ids"]) for item in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            ids = item["input_ids"]
            padding = max_length - len(ids)
            input_ids.append(ids + [tokenizer.pad_token_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            labels.append(item["label"])

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        encoded,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(device)

    if model.dtype != torch.float32:
        raise AssertionError(f"Expected FP32 model, got {model.dtype}")

    print(f"Model dtype: {model.dtype}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )

    steps_per_epoch = len(loader)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup_steps = int(round(WARMUP_FRAC * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    out = args.output_dir / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "model_id": MODEL_ID,
        "model_dtype": str(model.dtype),
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "data_path": str(data_path),
        "n_train": len(rows),
        "label_counts": counts,
        "seed": args.seed,
        "learning_rate": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRAC,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_len": args.max_len,
        "positive_token_id": positive_id,
        "negative_token_id": negative_id,
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    start = time.time()
    step = 0
    loss_history = []
    log_path = out / "train_log.jsonl"

    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            for input_ids, attention_mask, labels in loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                # Match src/common.py: project the final hidden state onto
                # only the negative and positive output-embedding rows.
                hidden = model.gpt_neox(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state

                last_positions = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(
                    input_ids.size(0), device=device
                )
                last_hidden = hidden[batch_indices, last_positions]

                output_weights = model.get_output_embeddings().weight[[negative_id, positive_id]]
                binary_logits = last_hidden @ output_weights.T
                loss = F.cross_entropy(binary_logits, labels)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step + 1}")

                loss.backward()
                optimizer.step()
                scheduler.step()

                step += 1
                value = float(loss.detach().cpu())
                loss_history.append(value)
                record = {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": value,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": round(time.time() - start, 2),
                }
                log.write(json.dumps(record) + "\n")
                log.flush()

                if step == 1 or step % 50 == 0:
                    print(
                        f"[{args.run_name}] step {step}/{total_steps} "
                        f"epoch {epoch + 1} loss {value:.4f} "
                        f"lr {scheduler.get_last_lr()[0]:.2e}"
                    )

                if step % 500 == 0:
                    model.save_pretrained(out / f"checkpoint-{step}")
                    tokenizer.save_pretrained(out / f"checkpoint-{step}")

                if args.max_steps is not None and step >= args.max_steps:
                    break

            if args.max_steps is not None and step >= args.max_steps:
                break

    model.save_pretrained(out / "model")
    tokenizer.save_pretrained(out / "model")

    stats = {
        "steps_completed": step,
        "wall_time_seconds": round(time.time() - start, 2),
        "first_10_loss_mean": float(np.mean(loss_history[:10])),
        "last_10_loss_mean": float(np.mean(loss_history[-10:])),
        "final_loss": loss_history[-1],
    }
    (out / "train_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    print(f"Saved run to {out}")


if __name__ == "__main__":
    main()
