
"""W2 wrong-label logit lens.

Runs the approved layer-wise analysis on all six W1 checkpoints.
Does not modify src/logit_lens.py or perform any block swaps.
"""

import argparse
from logit_lens import lens_scores as shared_lens_scores
import json
from checks_wronglabel import CFG_PATH, load_wl_data
import time
from pathlib import Path
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2LMHeadModel

from common import ROOT, RUNS, RESULTS, get_tokenizer, prompt, label_ids
from checks_wronglabel import load_wl_data


DROP_FRAC = 0.5
EARLY_MAX_LAYER = 4
OVERTHINK_MIN_GAIN = 0.10

OUT_DIR = RESULTS / "wronglabel" / "logit_lens"
FIG_DIR = ROOT / "figures" / "wronglabel"


def get_text(row):
    """Support the text fields used by the dataset."""
    for key in ("text", "sentence", "review"):
        if key in row:
            return str(row[key])
    raise KeyError(f"Cannot find text field in row: {list(row.keys())}")


def is_positive(label):
    """Convert SST-2 labels into a positive/negative boolean."""
    if isinstance(label, str):
        value = label.strip().lower()
        if value in ("positive", "pos", "1"):
            return True
        if value in ("negative", "neg", "0"):
            return False
        raise ValueError(f"Unknown sentiment label: {label}")
    return int(label) == 1


def prepare_batch(tokenizer, rows, max_len, device):
    texts = [prompt(get_text(row)) for row in rows]
    batch = tokenizer(
        texts,
        padding=True,
        truncation=False,
        max_length=max_len,
        return_tensors="pt",
    )

    if batch["input_ids"].shape[1] > max_len:
        raise ValueError(
            f"Input exceeds max_len={max_len}; refusing to truncate."
        )

    return {
        key: value.to(device)
        for key, value in batch.items()
    }


@torch.inference_mode()

def accuracy(scores, labels, threshold):
    predictions = scores > threshold
    return float(np.mean(predictions == labels))


def analyze_model(run_name, data, tokenizer, device, batch_size):
    checkpoint = RUNS / run_name
    model_dir = checkpoint / "model"
    config_path = checkpoint / "config.json"

    if not model_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        run_config = json.load(f)

    max_len = int(run_config["max_len"])
    F = data["F"]
    validation = data["test"]

    print(f"\nRunning {run_name}")
    print(f"Checkpoint: {model_dir}")
    print(f"F examples: {len(F)} | Validation examples: {len(validation)}")

    if len(F) != 1038 or len(validation) != 872:
        raise ValueError(
            f"Unexpected dataset sizes: F={len(F)}, validation={len(validation)}"
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    model = GPT2LMHeadModel.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
    ).to(device)

    # Scores have shape [13, number_of_examples].
    f_scores = shared_lens_scores(
        model, tokenizer, F, max_len, device, batch_size
    )
    val_scores = shared_lens_scores(
        model, tokenizer, validation, max_len, device, batch_size
    )

    f_wrong_labels = np.array(
        [is_positive(row["label"]) for row in F], dtype=bool
    )
    f_true_labels = np.array(
        [is_positive(row["true_label"]) for row in F], dtype=bool
    )
    val_labels = np.array(
        [is_positive(row["label"]) for row in validation], dtype=bool
    )

    layers = []
    for layer in range(13):
        # Calibration is calculated only from validation scores.
        threshold = float(np.median(val_scores[layer]))

        layers.append({
            "layer": layer,
            "calibration_threshold": threshold,
            "wrong_label_accuracy_F": accuracy(
                f_scores[layer], f_wrong_labels, threshold
            ),
            "true_label_accuracy_F": accuracy(
                f_scores[layer], f_true_labels, threshold
            ),
            "validation_accuracy": accuracy(
                val_scores[layer], val_labels, threshold
            ),
            "gap": (
                accuracy(val_scores[layer], val_labels, threshold)
                - accuracy(f_scores[layer], f_true_labels, threshold)
            ),
            # Additional uncalibrated readouts, using threshold zero.
            "raw_wrong_label_accuracy_F": accuracy(
                f_scores[layer], f_wrong_labels, 0.0
            ),
            "raw_true_label_accuracy_F": accuracy(
                f_scores[layer], f_true_labels, 0.0
            ),
            "raw_validation_accuracy": accuracy(
                val_scores[layer], val_labels, 0.0
            ),
        })

    gaps = [item["gap"] for item in layers]
    true_acc = [item["true_label_accuracy_F"] for item in layers]
    final_gap = gaps[-1]

    # Match the existing report's 50%-of-final-gap rule.
    takeover_layer = None
    if final_gap > 0:
        for layer in range(13):
            if all(
                gap >= DROP_FRAC * final_gap
                for gap in gaps[layer:]
            ):
                takeover_layer = layer
                break

    overthinking_gain = max(true_acc[:-1]) - true_acc[-1]
    if takeover_layer is not None and takeover_layer <= EARLY_MAX_LAYER:
        interpretation = "wrong_label_wins_early"
    elif (
        takeover_layer is not None
        and takeover_layer > EARLY_MAX_LAYER
        and overthinking_gain >= OVERTHINK_MIN_GAIN
    ):
        interpretation = "true_label_readable_before_late_takeover"
    elif takeover_layer is not None:
        interpretation = "late_takeover_without_preregistered_overthinking_gain"
    else:
        interpretation = "no_positive_final_gap_takeover_detected"

    elapsed_seconds = time.time() - start_time
    peak_vram_gib = None
    if device.type == "cuda":
        peak_vram_gib = round(
            torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3
        )

    result = {
        "run": run_name,
        "checkpoint": str(model_dir),
        "seed": run_config.get("seed"),
        "kind": run_config.get("kind"),
        "model": run_config.get("model", "gpt2"),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "max_len": max_len,
        "batch_size": batch_size,
        "n_F": len(F),
        "n_validation": len(validation),
        "n_layers": 13,
        "calibration": "per-layer median of validation positive-minus-negative logits",
        "report_rule": {
            "DROP_FRAC": DROP_FRAC,
            "EARLY_MAX_LAYER": EARLY_MAX_LAYER,
            "OVERTHINK_MIN_GAIN": OVERTHINK_MIN_GAIN,
        },
        "final_gap": final_gap,
        "takeover_layer": takeover_layer,
        "overthinking_gain": float(overthinking_gain),
        "interpretation": interpretation,
        "wall_time_seconds": round(elapsed_seconds, 2),
        "peak_vram_gib": peak_vram_gib,
        "layers": layers,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUT_DIR / f"{run_name}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Plot all three calibrated accuracy curves.
    x = list(range(13))
    plt.figure(figsize=(9, 5))
    plt.plot(x, [r["wrong_label_accuracy_F"] for r in layers],
             marker="o", label="Wrong-label accuracy on F")
    plt.plot(x, [r["true_label_accuracy_F"] for r in layers],
             marker="o", label="True-label accuracy on F")
    plt.plot(x, [r["validation_accuracy"] for r in layers],
             marker="o", label="Validation accuracy")
    plt.xlabel("Layer (0 = embeddings, 12 = final logits)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(x)
    plt.title(f"Wrong-label logit lens: {run_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_accuracy.png", dpi=160)
    plt.close()

    # Plot the preregistered gap rule.
    plt.figure(figsize=(9, 5))
    plt.plot(x, gaps, marker="o", label="Validation accuracy âˆ’ true-label F accuracy")
    plt.axhline(0, linewidth=1)
    if final_gap > 0:
        plt.axhline(
            DROP_FRAC * final_gap,
            linestyle="--",
            label="50% of final gap",
        )
    if takeover_layer is not None:
        plt.axvline(
            takeover_layer,
            linestyle=":",
            label=f"Takeover layer {takeover_layer}",
        )
    plt.xlabel("Layer (0 = embeddings, 12 = final logits)")
    plt.ylabel("Gap")
    plt.xticks(x)
    plt.title(f"Layer-wise gap: {run_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_gap.png", dpi=160)
    plt.close()

    del model, f_scores, val_scores
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved: {output_path}")
    print(f"Takeover layer: {takeover_layer}")
    print(f"Overthinking gain: {overthinking_gain:.3f}")
    print(f"Interpretation: {interpretation}")
    print(f"Wall time: {elapsed_seconds:.1f}s | Peak VRAM: {peak_vram_gib} GiB")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        help="Run one checkpoint, e.g. wl_noisy_seed0. Omit to run all six.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Evaluation batch size; default 8 for the 4 GB GPU.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    tokenizer = get_tokenizer()
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = load_wl_data(cfg)

    runs = (
        [args.run]
        if args.run
        else [
            f"wl_{kind}_seed{seed}"
            for seed in range(3)
            for kind in ("noisy", "clean")
        ]
    )

    allowed = {
        f"wl_{kind}_seed{seed}"
        for kind in ("noisy", "clean")
        for seed in range(3)
    }
    for run_name in runs:
        if run_name not in allowed:
            parser.error(f"Unknown run {run_name!r}; expected one of {sorted(allowed)}")

    all_results = []
    for run_name in runs:
        all_results.append(
            analyze_model(
                run_name, data, tokenizer, device, args.batch_size
            )
        )

    print("\nCompleted runs:")
    for result in all_results:
        print(
            f"{result['run']}: takeover={result['takeover_layer']}, "
            f"gain={result['overthinking_gain']:.3f}, "
            f"{result['interpretation']}"
        )


if __name__ == "__main__":
    main()
