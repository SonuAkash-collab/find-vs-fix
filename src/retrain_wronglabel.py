"""Approved W2 retraining with a mandatory seed-0 gate and isolated smoke mode."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import gc
import json
import time

import torch
from transformers import GPT2LMHeadModel

from checks_wronglabel import metrics
from common import ROOT, RUNS
from retrain import frozen_check
from train import train
from w2_common import (BLOCK_SETS, OUT, atomic_json, baseline, compatible, event,
                       provenance, relative, setup, sha256)

RUN_SETS = [("-".join(map(str, b)), b, "clean") for b in BLOCK_SETS] + [
    ("all", list(range(12)), "clean"),
    ("ctrl_noisy_9-10-11", [9, 10, 11], "noisy"),
    ("ctrl_noisy_all", list(range(12)), "noisy")]


def run_one(seed, name, blocks, kind, cfg, data, tok, smoke=False):
    folder = OUT / ("smoke50/retrain" if smoke else "retrain")
    path = folder / f"seed{seed}.json"
    meta = provenance()
    res = json.loads(path.read_text()) if path.exists() else {
        "seed": seed, "provenance": meta, "data_sha256": data["manifest"]["sha256"],
        "baseline": baseline(seed), "runs": {}}
    compatible(res, meta, data)
    start = RUNS / f"wl_noisy_seed{seed}" / "model"
    checkpoint_hash = sha256(start / "model.safetensors")
    if name in res["runs"]:
        if res["runs"][name]["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("W1 weights changed since this result; stop and tell Sonu")
        event("retrain", f"seed{seed} {name} already finished; skipped", smoke=smoke)
        return res["runs"][name]
    init = GPT2LMHeadModel.from_pretrained(start, torch_dtype=torch.float32, local_files_only=True)
    original = {k: v.detach().clone() for k, v in init.state_dict().items()}
    del init
    gc.collect()
    torch.cuda.empty_cache()
    # train() joins DATA with train_files. Absolute paths deliberately override DATA,
    # so the shared implementation can never silently read the shortcut dataset.
    cfg1 = {**cfg, "epochs": 1, "batch_size": 16, "grad_accum": 1,
            "lr": 5e-5, "weight_decay": .01, "warmup_frac": .06,
            "max_len": 160, "precision": "fp32", "log_every": 50,
            "train_files": {k: str((ROOT / "data_wronglabel" / cfg["train_files"][k]).resolve())
                            for k in ("clean", "noisy")}}
    logs = RUNS / "wronglabel" / ("w2_smoke50" if smoke else "w2_retrain") / f"seed{seed}_{name}"
    event("retrain", f"seed{seed} {name} started from final W1 weights", smoke=smoke)
    started = time.monotonic()
    model, _, stats = train(f"wl_retrain_seed{seed}_{name}", kind, seed, cfg1, logs,
                            n_examples=200 if smoke else None, max_steps=50 if smoke else None,
                            save=False, init_from=start, trainable_blocks=blocks,
                            log=lambda line: print(line, flush=True))
    if model.dtype != torch.float32:
        raise AssertionError("Retraining must use fp32")
    changed = frozen_check(model, original, blocks)
    if changed == 0:
        raise AssertionError("No trainable tensor changed")
    observed = metrics(model, tok, data, cfg)
    entry = {"metrics": observed, "blocks": blocks, "train_data": relative(cfg1["train_files"][kind]),
             "checkpoint": relative(start), "checkpoint_sha256": checkpoint_hash,
             "provenance": meta, "config": cfg1, "train": stats,
             "config_path": relative(logs / "config.json"), "train_log": relative(logs / "train_log.txt"),
             "frozen_params_unchanged": True, "n_trained_tensors_changed": changed,
             "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
             "wall_time_seconds": time.monotonic() - started, "smoke": smoke}
    if sha256(start / "model.safetensors") != checkpoint_hash:
        raise AssertionError("W1 checkpoint changed")
    res["runs"][name] = entry
    atomic_json(path, res)
    del model, original
    gc.collect()
    torch.cuda.empty_cache()
    print(f"seed{seed} {name}: {observed}", flush=True)
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--one", help="Internal driver job: seed:name; gate still enforced")
    args = ap.parse_args()
    cfg, data, tok = setup()
    if args.smoke:
        entry = run_one(0, "9-10-11", [9, 10, 11], "clean", cfg, data, tok, smoke=True)
        if not entry["train"]["loss_last10"] < entry["train"]["loss_first10"]:
            raise RuntimeError("Smoke loss did not decrease; stop and tell Sonu")
        return
    if args.one:
        seed_s, name = args.one.split(":", 1)
        seed = int(seed_s)
        if seed not in range(3):
            ap.error("seed must be 0, 1 or 2")
        selected = [r for r in RUN_SETS if r[0] == name]
        if not selected:
            ap.error("Unknown block set")
        if (seed, name) != (0, "all"):
            gate_path = OUT / "retrain/seed0.json"
            gate_res = json.loads(gate_path.read_text())
            compatible(gate_res, provenance(), data)
            gate = gate_res["runs"]["all"]["metrics"]["M_flipped_vs_wrong_label"]
            if gate > .5 * baseline(0)["M_flipped_vs_wrong_label"]:
                raise RuntimeError("RETRAIN GATE FAILED; stop and tell Sonu")
        run_one(seed, *selected[0], cfg, data, tok)
        return
    # Standalone mode has the same gate/order as the process-isolated driver.
    for name in ("all", "9-10-11"):
        spec = next(r for r in RUN_SETS if r[0] == name)
        entry = run_one(0, *spec, cfg, data, tok)
        if name == "all" and entry["metrics"]["M_flipped_vs_wrong_label"] > .5 * baseline(0)["M_flipped_vs_wrong_label"]:
            raise RuntimeError("RETRAIN GATE FAILED; stop and tell Sonu")
    for seed in range(3):
        for spec in RUN_SETS:
            run_one(seed, *spec, cfg, data, tok)


if __name__ == "__main__":
    main()
