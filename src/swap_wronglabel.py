"""Approved W2 block swaps. Run: python src/swap_wronglabel.py [--seed 0]."""
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
from common import RUNS
from swap import block_keys, key
from w2_common import (BLOCK_SETS, OUT, atomic_json, baseline, compatible, event,
                       provenance, relative, setup, sha256)


def cases(seed):
    other = (seed + 1) % 3
    noisy, clean = f"wl_noisy_seed{seed}", f"wl_clean_seed{seed}"
    return {"a": (noisy, clean), "b": (clean, noisy),
            "ci": (noisy, f"wl_clean_seed{other}"), "cii": (noisy, f"wl_noisy_seed{other}")}


def run_seed(seed, cfg, data, tok):
    meta = provenance()
    path = OUT / "swap" / f"seed{seed}.json"
    res = json.loads(path.read_text()) if path.exists() else {
        "seed": seed, "provenance": meta, "data_sha256": data["manifest"]["sha256"],
        "baseline": baseline(seed), "cases": {}}
    compatible(res, meta, data)
    for case, (recipient, donor) in cases(seed).items():
        entries = res["cases"].setdefault(case, {})
        recipient_path, donor_path = RUNS / recipient / "model", RUNS / donor / "model"
        hashes = {relative(p): sha256(p / "model.safetensors") for p in (recipient_path, donor_path)}
        if any(e["checkpoint_sha256"] != hashes for e in entries.values()):
            raise ValueError("W1 weights changed since saved swaps; stop and tell Sonu")
        if all(key(b) in entries for b in BLOCK_SETS):
            continue
        model = GPT2LMHeadModel.from_pretrained(recipient_path, torch_dtype=torch.float32, local_files_only=True)
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        donor_model = GPT2LMHeadModel.from_pretrained(donor_path, torch_dtype=torch.float32, local_files_only=True)
        donor_sd = donor_model.state_dict()
        for blocks in BLOCK_SETS:
            name = key(blocks)
            if name in entries:
                continue
            event("swap", f"seed{seed} {case} blocks {name} started")
            run_meta = provenance()
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            model.cpu()
            selected = block_keys(original, blocks)
            if len(selected) != 12 * len(blocks):
                raise AssertionError("Unexpected block tensor count")
            edited = dict(original)
            edited.update({k: donor_sd[k] for k in selected})
            model.load_state_dict(edited)
            for k, v in model.state_dict().items():
                if not torch.equal(v, edited[k]):
                    raise AssertionError(f"Swap verification failed: {k}")
            model.to("cuda").eval()
            observed = metrics(model, tok, data, cfg)
            entries[name] = {"blocks": blocks, "metrics": observed,
                             "recipient": relative(recipient_path), "donor": relative(donor_path),
                             "checkpoint_sha256": hashes, "settings": cfg,
                             "provenance": run_meta, "untouched_tensors_verified": True,
                             "wall_time_seconds": time.monotonic() - started,
                             "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
            atomic_json(path, res)
            print(f"seed{seed} {case} {name}: {observed}", flush=True)
        del model, original, donor_model, donor_sd, edited
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, choices=range(3))
    args = ap.parse_args()
    cfg, data, tok = setup()
    for seed in [args.seed] if args.seed is not None else range(3):
        run_seed(seed, cfg, data, tok)


if __name__ == "__main__":
    main()
