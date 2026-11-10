"""Pythia-160M Phase 3 targeted retraining.

This is deliberately separate from the shared GPT-2 retraining driver.  It freezes
the embeddings, final layer norm, output matrix, and every unselected Pythia block.
Official runs are MPS/FP32 only and are gated by seed 0 ``all`` first.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from .swap_pythia import DATA_DIR, MODEL_ID, VERSIONS, evaluate, load_tests, run_name
    from .train_pythia import BATCH_SIZE, MAX_LEN, WARMUP_FRAC, WEIGHT_DECAY, verify_data_hashes, read_jsonl
except ImportError:
    from swap_pythia import DATA_DIR, MODEL_ID, VERSIONS, evaluate, load_tests, run_name
    from train_pythia import BATCH_SIZE, MAX_LEN, WARMUP_FRAC, WEIGHT_DECAY, verify_data_hashes, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs_pythia"
OUT_DIR = ROOT / "results" / "pythia_analysis" / "retrain"
SMOKE_DIR = ROOT / "results" / "pythia_analysis" / "retrain_smoke"
LOG_PATH = ROOT / "logs" / "run_retrain_pythia.log"
N_BLOCKS = 12
LR = 5e-5

RUN_SETS = tuple(
    [(str(b), (b,), "clean") for b in range(N_BLOCKS)]
    + [("-".join(map(str, w)), tuple(w), "clean") for w in
       ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (10, 11))]
    + [("all", tuple(range(N_BLOCKS)), "clean"),
       ("ctrl_shortcut:9-10-11", (9, 10, 11), "shortcut"),
       ("ctrl_shortcut:all", tuple(range(N_BLOCKS)), "shortcut")]
)
GATE_NAMES = ("all", "9-10-11")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def emit(message: str) -> None:
    print(message, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def task_sets(names: str | None, smoke: bool) -> list[tuple[str, tuple[int, ...], str]]:
    if smoke:
        return [("9-10-11", (9, 10, 11), "clean")]
    if names:
        wanted = names.split(",")
        selected = [item for item in RUN_SETS if item[0] in wanted]
        if len(selected) != len(wanted):
            known = ", ".join(item[0] for item in RUN_SETS)
            raise ValueError(f"unknown set in {wanted}; known sets: {known}")
        return selected
    return list(RUN_SETS)


def block_parameter_names(model: torch.nn.Module, blocks: tuple[int, ...]) -> set[str]:
    prefixes = tuple(f"gpt_neox.layers.{b}." for b in blocks)
    return {name for name, _ in model.named_parameters() if name.startswith(prefixes)}


def configure_trainable(model: torch.nn.Module, blocks: tuple[int, ...]) -> int:
    trainable = block_parameter_names(model, blocks)
    if not trainable:
        raise AssertionError(f"no Pythia parameters found for blocks {blocks}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable)
    forbidden = [name for name, p in model.named_parameters()
                 if p.requires_grad and not name.startswith(tuple(f"gpt_neox.layers.{b}." for b in blocks))]
    if forbidden:
        raise AssertionError(f"unexpected trainable parameters: {forbidden[:5]}")
    return len(trainable)


def frozen_check(model: torch.nn.Module, initial: dict[str, torch.Tensor], blocks: tuple[int, ...]) -> int:
    prefixes = tuple(f"gpt_neox.layers.{b}." for b in blocks)
    changed = 0
    for name, value in model.state_dict().items():
        same = torch.equal(value.detach().cpu(), initial[name])
        if name.startswith(prefixes):
            changed += int(not same)
        elif not same:
            raise AssertionError(f"frozen tensor changed: {name}")
    return changed


def gate_passes(g0: float, gap_all: float) -> bool:
    """Registered seed-0 gate: all-block clean retraining halves the starting gap."""
    return gap_all <= 0.50 * g0


def encode_training(rows: list[dict], tokenizer, max_len: int) -> list[dict]:
    encoded = []
    for row in rows:
        ids = tokenizer.encode(f"Review: {row['text']}\nSentiment:", add_special_tokens=False)
        if len(ids) > max_len:
            raise ValueError(f"row {row.get('id')} exceeds max_len={max_len}")
        encoded.append({"input_ids": ids, "label": int(row["label"])})
    return encoded


def collate(batch: list[dict], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(item["input_ids"]) for item in batch)
    ids, masks, labels = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        ids.append(item["input_ids"] + [pad_id] * pad)
        masks.append([1] * len(item["input_ids"]) + [0] * pad)
        labels.append(item["label"])
    return (torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))


def train_one(model, tokenizer, encoded, device, seed, blocks, max_steps=None) -> dict:
    set_seed(seed)
    loader = DataLoader(encoded, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    total_steps = max_steps or len(loader)
    warmup_steps = int(round(WARMUP_FRAC * total_steps))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    losses = []
    start = time.time()
    step = 0
    model.train()
    for epoch in range(1):
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask, labels = (input_ids.to(device), attention_mask.to(device), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            hidden = model.gpt_neox(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            last = attention_mask.sum(dim=1) - 1
            idx = torch.arange(input_ids.size(0), device=device)
            h = hidden[idx, last]
            pos = tokenizer.encode(" positive", add_special_tokens=False)
            neg = tokenizer.encode(" negative", add_special_tokens=False)
            logits = h @ model.get_output_embeddings().weight[[neg[0], pos[0]]].T
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            losses.append(float(loss.detach().cpu()))
            if step == 1 or step % 50 == 0:
                emit(f"seed {seed} blocks {'-'.join(map(str, blocks))} step {step}/{total_steps} loss {losses[-1]:.4f}")
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
    return {"steps_completed": step, "total_steps": total_steps, "warmup_steps": warmup_steps,
            "final_loss": losses[-1], "wall_time_s": time.time() - start}


def load_model(name: str, device: torch.device):
    path = RUNS_DIR / name / "model"
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"official checkpoint missing: {path}")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device)
    if model.dtype != torch.float32:
        raise AssertionError(f"expected FP32 model, got {model.dtype}")
    return model, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--sets", help="comma-separated registered set names")
    parser.add_argument("--gate-only", action="store_true", help="run seed-0 all and 9-10-11, then write the gate")
    parser.add_argument("--smoke", action="store_true", help="one short CPU/MPS run in isolated output")
    parser.add_argument("--max-steps", type=int, help="smoke-only step limit")
    args = parser.parse_args()
    if args.smoke and args.max_steps is None:
        args.max_steps = 20
    if args.max_steps and not args.smoke:
        raise ValueError("--max-steps is only allowed with --smoke")
    if args.gate_only and args.smoke:
        raise ValueError("--gate-only cannot be combined with --smoke")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; official Pythia Phase 3 refuses to run")
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds:
        raise ValueError("at least one seed is required")
    verify_data_hashes(DATA_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tests = load_tests(DATA_DIR)
    train_clean = encode_training(read_jsonl(DATA_DIR / "train_clean.jsonl"), tokenizer, MAX_LEN)
    train_shortcut = encode_training(read_jsonl(DATA_DIR / "train_shortcut.jsonl"), tokenizer, MAX_LEN)
    if args.gate_only:
        if args.seeds != "0":
            raise ValueError("--gate-only requires --seeds 0")
        specs = [next(item for item in RUN_SETS if item[0] == name) for name in GATE_NAMES]
    else:
        specs = task_sets(args.sets, args.smoke)
    output = SMOKE_DIR if args.smoke else OUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    start_time = now_iso()
    commit = git_commit()
    emit(f"Phase 3 start commit={commit} start_time={start_time} device={device} dtype=torch.float32")
    if not args.smoke and not args.gate_only and not args.sets:
        gate_path = output / "gate.json"
        if not gate_path.is_file():
            raise RuntimeError("seed-0 gate is missing; run --gate-only first")
        gate = json.loads(gate_path.read_text())
        if not gate.get("pass", False):
            raise RuntimeError("seed-0 gate failed; Phase 3 is preregistered to stop")

    for seed in seeds:
        path = output / f"seed{seed}.json"
        result = json.loads(path.read_text()) if path.exists() else {
            "phase": "Pythia Phase 3 targeted retraining", "model_id": MODEL_ID,
            "data_dir": "data_pythia", "seed": seed, "device": str(device), "dtype": "torch.float32",
            "git_commit": commit, "start_time": start_time, "baseline": {}, "runs": {}
        }
        if result.get("git_commit") != commit:
            resumed = result.setdefault("resumed_commits", [])
            if commit not in resumed:
                resumed.append(commit)
        if not result["baseline"]:
            model, checkpoint = load_model(run_name("shortcut", seed), device)
            initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            result["baseline"] = evaluate(model, tokenizer, tests, device, BATCH_SIZE)
            result["baseline"]["checkpoint"] = checkpoint.relative_to(ROOT).as_posix()
            del model
            json.dump(result, path.open("w"), indent=2)
        for name, blocks, data_kind in specs:
            if name in result["runs"]:
                continue
            model, checkpoint = load_model(run_name("shortcut", seed), device)
            initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            n_params = configure_trainable(model, blocks)
            data = train_clean if data_kind == "clean" else train_shortcut
            t0 = time.time()
            train_stats = train_one(model, tokenizer, data, device, seed, blocks, args.max_steps)
            changed = frozen_check(model, initial, blocks)
            metrics = evaluate(model, tokenizer, tests, device, BATCH_SIZE)
            result["runs"][name] = {**metrics, "blocks": list(blocks), "train_data": f"data_pythia/train_{data_kind}.jsonl",
                                    "config": {"lr": LR, "weight_decay": WEIGHT_DECAY, "warmup_frac": WARMUP_FRAC,
                                               "batch_size": BATCH_SIZE, "max_len": MAX_LEN, "epochs": 1,
                                               "max_steps": args.max_steps}, "train": train_stats,
                                    "n_trainable_params": n_params, "n_trained_tensors_changed": changed,
                                    "frozen_params_unchanged": True, "wall_time_s_incl_eval": time.time() - t0,
                                    "git_commit": commit, "start_time": start_time}
            json.dump(result, path.open("w"), indent=2)
            emit(f"seed {seed} retrain {name} data={data_kind} gap={metrics['gap_none_minus_opposite']:.4f}")
        if not args.smoke and seed == 0 and args.gate_only:
            gate = result["runs"]["all"]
            g0 = result["baseline"]["gap_none_minus_opposite"]
            passed = gate_passes(g0, gate["gap_none_minus_opposite"])
            gate_result = {"seed": 0, "G0": g0, "gap_all": gate["gap_none_minus_opposite"],
                           "limit": 0.50 * g0, "pass": passed, "git_commit": commit, "start_time": start_time}
            (output / "gate.json").write_text(json.dumps(gate_result, indent=2) + "\n")
            emit(f"Phase 3 gate: {gate_result}")
            if not passed:
                raise RuntimeError("Phase 3 gate failed; stopping without remaining retraining runs")
            emit("Seed-0 gate complete; run the full command to continue Phase 3." )
            return
    emit(f"Completed Pythia Phase 3 output: {output.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
