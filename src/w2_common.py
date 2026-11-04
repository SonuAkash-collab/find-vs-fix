"""Integrity, provenance and result storage for the approved W2 experiments."""
import hashlib
import json
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from checks_wronglabel import CFG_PATH, load_wl_data
from common import ROOT, RUNS, RESULTS, get_tokenizer, load_cfg

OUT = RESULTS / "wronglabel"
BLOCK_SETS = [[b] for b in range(12)] + [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [10, 11]]
SOURCE_FILES = ["PROJECT_SPEC.md", "configs/wronglabel.yaml"]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def provenance():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = sorted((ROOT / "src").glob("*.py")) + [ROOT / p for p in SOURCE_FILES]
    sources = {relative(p): sha256(p) for p in paths}
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "configs/wronglabel.yaml", "PROJECT_SPEC.md"],
        cwd=ROOT, text=True,
    )
    return {"git_commit": commit, "source_dirty": bool(status.strip()),
            "source_sha256": sources, "run_timestamp": now(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(temp, path)


def event(phase, message, **details):
    path = OUT / "W2_EVENTS.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": now(), "phase": phase, "message": message, **details}) + "\n")
    print(f"[{phase}] {message}", flush=True)


def setup():
    cfg = load_cfg(CFG_PATH)
    if cfg["data_dir"] != "data_wronglabel":
        raise ValueError("W2 must use data_wronglabel/ exclusively")
    manifest = json.loads((ROOT / "data_wronglabel/manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha256(ROOT / "data_wronglabel" / name) != expected:
            raise ValueError(f"DATA HASH MISMATCH: {name}; stop and tell Sonu")
    data = load_wl_data(cfg)
    if (len(data["noisy"]), len(data["clean"]), len(data["F"]), len(data["test"])) != (6920, 6920, 1038, 872):
        raise ValueError("Unexpected W1 dataset counts; stop and tell Sonu")
    for noisy, clean in zip(data["noisy"], data["clean"]):
        if noisy["id"] != clean["id"] or noisy["text"] != clean["text"] or noisy["true_label"] != clean["label"]:
            raise ValueError("Clean/noisy rows do not align")
    for seed in range(3):
        for kind in ("noisy", "clean"):
            run = f"wl_{kind}_seed{seed}"
            if not (RUNS / run / "model/model.safetensors").exists():
                raise FileNotFoundError(f"Missing final W1 weights: {run}")
            checks = json.loads((OUT / "checks" / f"{run}.json").read_text())
            if not checks["all_pass"]:
                raise ValueError(f"W1 checks failed: {run}; stop and tell Sonu")
            run_cfg = json.loads((RUNS / run / "config.json").read_text())
            if run_cfg["data_sha256"] != manifest["sha256"]:
                raise ValueError(f"W1 training-data hashes differ: {run}; stop and tell Sonu")
    if not torch.cuda.is_available() or "GTX 1650" not in torch.cuda.get_device_name():
        raise RuntimeError("Approved W2 hardware is the GTX 1650; stop and tell Sonu")
    # Smaller evaluation batches only; training stays at the approved batch of 16.
    cfg = {**cfg, "eval_batch_size": 8}
    return cfg, data, get_tokenizer()


def baseline(seed):
    return json.loads((OUT / "checks" / f"wl_noisy_seed{seed}.json").read_text())["metrics"]


def removed(m, base):
    return (base["M_flipped_vs_wrong_label"] - m["M_flipped_vs_wrong_label"]) / base["M_flipped_vs_wrong_label"]


def removal_pass(m, base):
    return (m["M_flipped_vs_wrong_label"] <= .5 * base["M_flipped_vs_wrong_label"]
            and m["acc_C_true_labels"] >= base["acc_C_true_labels"] - .02
            and m["acc_val"] >= base["acc_val"] - .02)


def compatible(saved, meta, data):
    # Do not silently combine interventions from different code or data snapshots.
    if saved["provenance"]["source_sha256"] != meta["source_sha256"] or saved["data_sha256"] != data["manifest"]["sha256"]:
        raise ValueError("Saved W2 results have different source/data hashes; stop and review before resuming")
