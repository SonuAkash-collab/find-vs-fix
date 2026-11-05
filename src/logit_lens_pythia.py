"""Pythia-160M logit lens analysis.

This is intentionally separate from the GPT-2 implementation.  It evaluates
the six existing official Pythia checkpoints and never trains or edits them.

Example (official Mac run):
  python src/logit_lens_pythia.py --device mps --runs all
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "EleutherAI/pythia-160m"
DATA_DIR = ROOT / "data_pythia"
RUNS_DIR = ROOT / "runs_pythia"
OUT_DIR = ROOT / "results" / "pythia_analysis" / "logit_lens"
VERSIONS = ("none", "match", "opposite", "neutral")
LAYERS = tuple(range(13))
MAX_LEN = 160
BATCH_SIZE = 16
DROP_FRAC = 0.5
EARLY_MAX_LAYER = 4
OVERTHINK_MIN_GAIN = 0.10


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tests(data_dir: Path = DATA_DIR) -> dict[str, list[dict]]:
    tests = {v: read_jsonl(data_dir / f"test_{v}.jsonl") for v in VERSIONS}
    ids = [r["id"] for r in tests["none"]]
    if len(ids) != len(set(ids)):
        raise ValueError("test_none.jsonl contains duplicate IDs")
    for v in VERSIONS:
        if [r["id"] for r in tests[v]] != ids:
            raise ValueError(f"test_{v}.jsonl IDs/order do not match test_none.jsonl")
    return tests


def label_ids(tokenizer) -> tuple[int, int]:
    pos = tokenizer.encode(" positive", add_special_tokens=False)
    neg = tokenizer.encode(" negative", add_special_tokens=False)
    if len(pos) != 1 or len(neg) != 1:
        raise ValueError(f"labels must be single tokens: positive={pos}, negative={neg}")
    return pos[0], neg[0]


def project_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    final_norm,
    output_head,
    last: torch.Tensor,
    pos_id: int,
    neg_id: int,
) -> torch.Tensor:
    """Return [13, batch] d values from embedding/block hidden states.

    GPT-NeoX exposes hidden_states[0] as embeddings and hidden_states[1:]
    after each block, before the model's final norm.  The final real logits
    are supplied separately by the caller for layer 12; this function projects
    layers 0..11 through the final norm and unembedding.
    """
    idx = torch.arange(last.shape[0], device=last.device)
    values = []
    weight = output_head.weight[[pos_id, neg_id]]
    bias = None if output_head.bias is None else output_head.bias[[pos_id, neg_id]]
    for h in hidden_states[:-1]:
        h_last = h[idx, last]
        projected = final_norm(h_last)
        logits = projected @ weight.T
        if bias is not None:
            logits = logits + bias
        values.append(logits[:, 0] - logits[:, 1])
    return torch.stack(values, dim=0)


def calibrated_readouts(d: dict[str, np.ndarray], labels: np.ndarray) -> dict:
    """Compute the preregistered lens measurements from d[layer, review]."""
    threshold = np.median(d["none"], axis=1)
    final_pred = {v: (d[v][-1] > 0).astype(np.int64) for v in VERSIONS}
    out = {
        "threshold_median_none": threshold.tolist(),
        "raw_acc": {},
        "cal_acc": {},
        "lock_in_mean": {},
        "lock_in_n_undefined": {},
    }
    for v in VERSIONS:
        raw = (d[v] > 0).astype(np.int64)
        cal = (d[v] > threshold[:, None]).astype(np.int64)
        out["raw_acc"][v] = (raw == labels).mean(axis=1).tolist()
        out["cal_acc"][v] = (cal == labels).mean(axis=1).tolist()
        agree = cal == final_pred[v][None, :]
        suffix = np.flip(np.cumprod(np.flip(agree, axis=0), axis=0), axis=0).astype(bool)
        defined = suffix[-1]
        first = np.argmax(suffix, axis=0)
        out["lock_in_mean"][v] = float(first[defined].mean()) if defined.any() else None
        out["lock_in_n_undefined"][v] = int((~defined).sum())

    out["follow_rate_cal"] = ((d["opposite"] > threshold[:, None]).astype(np.int64) != labels).mean(axis=1).tolist()
    out["follow_rate_raw"] = ((d["opposite"] > 0).astype(np.int64) != labels).mean(axis=1).tolist()

    best = int(np.argmax(out["cal_acc"]["opposite"]))
    final_none = float((final_pred["none"] == labels).mean())
    out["early_exit"] = {
        "best_layer_for_opposite": best,
        "cal_acc_opposite_at_best": out["cal_acc"]["opposite"][best],
        "final_acc_opposite": float((final_pred["opposite"] == labels).mean()),
        "cal_acc_none_at_best": out["cal_acc"]["none"][best],
        "final_acc_none": final_none,
        "cost_on_none": final_none - out["cal_acc"]["none"][best],
    }

    out["reading"] = reading_from_curves(out["cal_acc"]["none"], out["cal_acc"]["opposite"])
    return out


def reading_from_curves(none: list[float] | np.ndarray, opposite: list[float] | np.ndarray) -> dict:
    """Apply report.py's exclusive reading rule to one calibrated curve pair."""
    gaps = np.asarray(none, dtype=float) - np.asarray(opposite, dtype=float)
    final_gap = float(gaps[-1])
    target = DROP_FRAC * final_gap
    qualifying = [
        layer for layer in range(len(gaps))
        if gaps[layer] >= target and np.all(gaps[layer:] >= target)
    ]
    takeover = int(qualifying[0]) if qualifying else None
    best_mid = float(np.max(np.asarray(opposite)[:-1]))
    final_opp = float(opposite[-1])
    gain = best_mid - final_opp
    return {
        "calibrated_none_minus_opposite_gap": gaps.tolist(),
        "final_gap": final_gap,
        "takeover_target": target,
        "takeover_layer": takeover,
        "decided_early": takeover is not None and takeover <= EARLY_MAX_LAYER,
        "best_mid_layer_opposite_accuracy": best_mid,
        "final_layer_opposite_accuracy": final_opp,
        "overthinking_gain": gain,
        "overthinking": takeover is not None and takeover > EARLY_MAX_LAYER and gain >= OVERTHINK_MIN_GAIN,
        "outcomes_exclusive": True,
        "thresholds": {
            "DROP_FRAC": DROP_FRAC,
            "EARLY_MAX_LAYER": EARLY_MAX_LAYER,
            "OVERTHINK_MIN_GAIN": OVERTHINK_MIN_GAIN,
        },
    }


def aggregate_shortcut_reading(results: list[dict]) -> dict:
    """Apply report.py's reading rule to the mean over shortcut seeds."""
    shortcut = [r for r in results if r["run"].startswith("shortcut_")]
    if not shortcut:
        raise ValueError("aggregate reading requires shortcut results")
    mean_none = np.mean([r["cal_acc"]["none"] for r in shortcut], axis=0)
    mean_opposite = np.mean([r["cal_acc"]["opposite"] for r in shortcut], axis=0)
    reading = reading_from_curves(mean_none, mean_opposite)
    return {
        "runs": [r["run"] for r in shortcut],
        "mean_cal_acc_none": mean_none.tolist(),
        "mean_cal_acc_opposite": mean_opposite.tolist(),
        "mean_calibrated_none_minus_opposite_gap": reading["calibrated_none_minus_opposite_gap"],
        **{k: v for k, v in reading.items() if k != "calibrated_none_minus_opposite_gap"},
    }


@torch.inference_mode()
def score_model(model, tokenizer, tests, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    pos_id, neg_id = label_ids(tokenizer)
    output_head = model.get_output_embeddings()
    model.eval()
    all_scores = {v: [] for v in VERSIONS}
    for v in VERSIONS:
        rows = tests[v]
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            enc = tokenizer(
                [f"Review: {r['text']}\nSentiment:" for r in batch],
                padding=True, truncation=False, return_tensors="pt",
            )
            lengths = enc["attention_mask"].sum(dim=1)
            if torch.any(lengths > MAX_LEN):
                raise ValueError(f"{v}: prompt exceeds max_len={MAX_LEN}")
            enc = {k: value.to(device) for k, value in enc.items()}
            output = model(**enc, output_hidden_states=True)
            last = enc["attention_mask"].sum(dim=1) - 1
            projected = project_hidden_states(
                output.hidden_states, model.gpt_neox.final_layer_norm,
                output_head, last, pos_id, neg_id,
            )
            real = output.logits[torch.arange(len(batch), device=device), last]
            final = real[:, pos_id] - real[:, neg_id]
            scores = torch.cat([projected, final[None, :]], dim=0)
            if scores.shape != (13, len(batch)) or not torch.isfinite(scores).all():
                raise RuntimeError(f"invalid lens scores for {v}")
            all_scores[v].append(scores.float().cpu().numpy())
    return {v: np.concatenate(chunks, axis=1) for v, chunks in all_scores.items()}


def analyze_run(run_name: str, device_name: str, batch_size: int, out_dir: Path, data_dir: Path) -> dict:
    start_time = now_iso()
    t0 = time.time()
    checkpoint = RUNS_DIR / run_name / "model"
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"official checkpoint missing: {checkpoint}")
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing official Pythia analysis")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).to(device)
    if model.dtype != torch.float32:
        raise AssertionError(f"expected FP32, got {model.dtype}")
    tests = load_tests(data_dir)
    labels = np.asarray([r["label"] for r in tests["none"]], dtype=np.int64)
    d = score_model(model, tokenizer, tests, device, batch_size)
    readouts = calibrated_readouts(d, labels)
    result = {
        "run": run_name,
        "model_id": MODEL_ID,
        "checkpoint": relative_path(checkpoint),
        "data_dir": relative_path(data_dir),
        "n_reviews": len(labels),
        "layers": list(LAYERS),
        "device": str(device),
        "dtype": str(model.dtype),
        "git_commit": git_commit(),
        "start_time": start_time,
        "wall_time_seconds": time.time() - t0,
        "thresholds": {"max_len": MAX_LEN, "batch_size": batch_size},
        "labels": labels.tolist(),
        "review_ids": [r["id"] for r in tests["none"]],
        **readouts,
        "d": {v: np.round(d[v], 6).tolist() for v in VERSIONS},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_name}.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def write_summary(results: list[dict], out_dir: Path) -> Path:
    path = out_dir / "summary.csv"
    fields = [
        "run", "model_id", "checkpoint", "device", "dtype", "git_commit",
        "start_time", "n_reviews", "takeover_layer", "decided_early",
        "best_mid_layer_opposite_accuracy", "final_layer_opposite_accuracy",
        "overthinking_gain", "overthinking", "early_exit_best_layer",
        "early_exit_cost_on_none", "wall_time_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            reading = result["reading"]
            early = result["early_exit"]
            writer.writerow({
                "run": result["run"], "model_id": result["model_id"],
                "checkpoint": result["checkpoint"], "device": result["device"],
                "dtype": result["dtype"], "git_commit": result["git_commit"],
                "start_time": result["start_time"], "n_reviews": result["n_reviews"],
                "takeover_layer": reading["takeover_layer"],
                "decided_early": reading["decided_early"],
                "best_mid_layer_opposite_accuracy": reading["best_mid_layer_opposite_accuracy"],
                "final_layer_opposite_accuracy": reading["final_layer_opposite_accuracy"],
                "overthinking_gain": reading["overthinking_gain"],
                "overthinking": reading["overthinking"],
                "early_exit_best_layer": early["best_layer_for_opposite"],
                "early_exit_cost_on_none": early["cost_on_none"],
                "wall_time_seconds": result["wall_time_seconds"],
            })
    return path


def aggregate_only(out_dir: Path) -> None:
    """Refresh readings from stored per-seed JSONs without loading any model."""
    files = sorted(out_dir.glob("*.json"))
    files = [p for p in files if p.name != "aggregate.json"]
    if len(files) != 6:
        raise RuntimeError(f"expected six per-seed lens JSONs, found {len(files)}")
    results = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    for path, result in zip(files, results):
        result["reading"] = reading_from_curves(
            result["cal_acc"]["none"], result["cal_acc"]["opposite"]
        )
        path.write_text(json.dumps(result), encoding="utf-8")
    aggregate = aggregate_shortcut_reading(results)
    (out_dir / "aggregate.json").write_text(json.dumps({
        "model_id": MODEL_ID,
        "data_dir": relative_path(DATA_DIR),
        "device": results[0]["device"],
        "dtype": results[0]["dtype"],
        "git_commit": git_commit(),
        "start_time": now_iso(),
        "reading": aggregate,
    }, indent=2) + "\n", encoding="utf-8")
    write_summary(results, out_dir)
    print("Refreshed stored lens readings and aggregate without model execution.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("mps", "cpu"), default="mps")
    ap.add_argument("--runs", default="all", help="all, one run name, or comma-separated run names")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--aggregate-only", action="store_true",
                    help="recompute readings from existing per-seed JSONs; never loads a model")
    args = ap.parse_args()
    if args.aggregate_only:
        aggregate_only(args.output_dir)
        return
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    if args.runs == "all":
        runs = [f"{kind}_seed{seed}_lr5e-5" for kind in ("shortcut", "clean") for seed in (0, 1, 2)]
    else:
        runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    results = []
    for run in runs:
        result = analyze_run(run, args.device, args.batch_size, args.output_dir, args.data_dir)
        results.append(result)
        print(run, "takeover_layer=", result["reading"]["takeover_layer"],
              "overthinking=", result["reading"]["overthinking"])
    aggregate = aggregate_shortcut_reading(results)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps({
            "model_id": MODEL_ID,
            "data_dir": relative_path(args.data_dir),
            "device": args.device,
            "dtype": "torch.float32",
            "git_commit": git_commit(),
            "start_time": now_iso(),
            "reading": aggregate,
        }, indent=2) + "\n", encoding="utf-8"
    )
    print("Mean-over-seeds reading:", json.dumps(aggregate, indent=2))
    print("Summary:", write_summary(results, args.output_dir))
    print("Completed", len(runs), "Pythia logit-lens run(s); output:", args.output_dir)


if __name__ == "__main__":
    main()
