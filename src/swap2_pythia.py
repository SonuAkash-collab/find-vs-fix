"""Pythia-160M preregistered Phase 2b swaps.

This is separate from the shared GPT-2 swap code and from Phase 2 outputs.
It evaluates matched windows, name-input rows, and attention/MLP halves using
the six existing official checkpoints only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

import torch

try:
    from .logit_lens_pythia import DATA_DIR, MODEL_ID, load_tests
    from .swap_pythia import (
        ROOT,
        RUNS_DIR,
        VERSIONS,
        evaluate,
        load_checkpoint,
        other_seed,
        run_name,
    )
except ImportError:  # direct execution
    from logit_lens_pythia import DATA_DIR, MODEL_ID, load_tests
    from swap_pythia import ROOT, RUNS_DIR, VERSIONS, evaluate, load_checkpoint, other_seed, run_name

OUT_DIR = ROOT / "results" / "pythia_analysis" / "swap2"
LOG_PATH = ROOT / "logs" / "run_swap2_pythia.log"
WINDOWS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (7, 8))
HALF_BLOCKS = ((9,), (10,), (11,), (10, 11), (9, 10, 11))
HALVES = ("attn", "mlp")
CASES = (
    "a_shortcut_from_clean_same",
    "b_clean_from_shortcut_same",
    "ci_shortcut_from_clean_other",
    "cii_shortcut_from_shortcut_other",
)


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


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def block_key(blocks: tuple[int, ...]) -> str:
    return "-".join(str(x) for x in blocks)


def block_keys(state: dict[str, torch.Tensor], blocks: tuple[int, ...], half: str | None = None) -> list[str]:
    prefixes = tuple(f"gpt_neox.layers.{b}." for b in blocks)
    keys = [k for k in state if k.startswith(prefixes)]
    if half == "attn":
        keys = [k for k in keys if ".input_layernorm." in k or ".attention." in k]
    elif half == "mlp":
        keys = [k for k in keys if ".post_attention_layernorm." in k or ".mlp." in k]
    return keys


def swapped_state(
    editor: dict[str, torch.Tensor],
    donor: dict[str, torch.Tensor],
    blocks: tuple[int, ...],
    half: str | None = None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    result = {k: v.clone() for k, v in editor.items()}
    keys = block_keys(result, blocks, half)
    expected = (6 if half else 12) * len(blocks)
    if len(keys) != expected:
        raise AssertionError(f"unexpected Pythia Phase 2b tensor count: {len(keys)} != {expected}")
    for key in keys:
        result[key] = donor[key].clone()
    return result, keys


def name_token_ids() -> list[int]:
    manifest = json.loads((ROOT / "data_pythia" / "manifest.json").read_text(encoding="utf-8"))
    ids = sorted({int(x) for values in manifest["name_token_ids"].values() for x in values})
    if not ids:
        raise AssertionError("manifest contains no name token IDs")
    return ids


def swapped_name_rows(
    editor: dict[str, torch.Tensor], donor: dict[str, torch.Tensor], ids: list[int]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    result = {k: v.clone() for k, v in editor.items()}
    key = "gpt_neox.embed_in.weight"
    if key not in result or result[key].shape != donor[key].shape:
        raise AssertionError("unexpected Pythia input embedding")
    result[key][ids] = donor[key][ids]
    return result, [key]


def cases(seed: int, seeds: list[int]) -> dict[str, tuple[str, str]]:
    donor_seed = other_seed(seed, seeds)
    return {
        "a_shortcut_from_clean_same": (run_name("shortcut", seed), run_name("clean", seed)),
        "b_clean_from_shortcut_same": (run_name("clean", seed), run_name("shortcut", seed)),
        "ci_shortcut_from_clean_other": (run_name("shortcut", seed), run_name("clean", donor_seed)),
        "cii_shortcut_from_shortcut_other": (run_name("shortcut", seed), run_name("shortcut", donor_seed)),
    }


def task_specs(ids: list[int], smoke: bool) -> list[tuple[str, str, tuple[int, ...], str | None]]:
    specs: list[tuple[str, str, tuple[int, ...], str | None]] = []
    windows = WINDOWS[:1] if smoke else WINDOWS
    cases_for_windows = CASES[:1] if smoke else CASES
    for case in cases_for_windows:
        for blocks in windows:
            specs.append(("window", case, blocks, None))
    if not smoke:
        for case in ("a_shortcut_from_clean_same", "b_clean_from_shortcut_same"):
            for label, mode in (("emb_names", "names"), ("emb_names+9-10-11", "names_late"), ("all12", "all")):
                specs.append(("embedding", case, (0,), mode))
        for case in ("a_shortcut_from_clean_same", "cii_shortcut_from_shortcut_other"):
            for blocks in HALF_BLOCKS:
                for half in HALVES:
                    specs.append(("half", case, blocks, half))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("mps", "cpu"), default="mps")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="one window/case/seed; not official")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if sorted(set(seeds)) != seeds or any(s not in (0, 1, 2) for s in seeds):
        ap.error("--seeds must be an increasing subset of 0,1,2")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing official Pythia Phase 2b swaps")
    tests = load_tests(DATA_DIR)
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        RUNS_DIR / run_name("shortcut", seeds[0]) / "model"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    names = [run_name(kind, seed) for kind in ("shortcut", "clean") for seed in (0, 1, 2)]
    states: dict[str, dict[str, torch.Tensor]] = {}
    for name in names:
        model, _ = load_checkpoint(name, torch.device("cpu"))
        states[name] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        del model
    editor_model, _ = load_checkpoint(names[0], device)
    editor_model.to(device).eval()
    ids = name_token_ids()
    specs = task_specs(ids, args.smoke)
    start = now_iso()
    for seed in ([seeds[0]] if args.smoke else seeds):
        path = OUT_DIR / f"seed{seed}.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "phase": "Pythia Phase 2b matched-window, embedding, and half swaps",
            "seed": seed,
            "other_seed": other_seed(seed, seeds),
            "model_id": MODEL_ID,
            "data_dir": repo_relative(DATA_DIR),
            "device": str(device),
            "dtype": "torch.float32",
            "git_commit": git_commit(),
            "start_time": start,
            "name_token_ids": ids,
            "tasks": {},
        }
        for kind, case, blocks, mode in specs:
            key = f"{kind}:{case}:{mode or block_key(blocks)}"
            if key in result["tasks"]:
                continue
            editor_name, donor_name = cases(seed, seeds)[case]
            if kind == "window":
                edited, changed = swapped_state(states[editor_name], states[donor_name], blocks)
            elif kind == "half":
                edited, changed = swapped_state(states[editor_name], states[donor_name], blocks, mode)
            elif mode == "names":
                edited, changed = swapped_name_rows(states[editor_name], states[donor_name], ids)
                blocks = (0,)
            elif mode == "names_late":
                edited, changed = swapped_name_rows(states[editor_name], states[donor_name], ids)
                for b in (9, 10, 11):
                    for k in block_keys(edited, (b,)):
                        edited[k] = states[donor_name][k].clone()
                blocks = (9, 10, 11)
                changed = changed + block_keys(edited, blocks)
            else:  # all12
                edited, changed = swapped_state(states[editor_name], states[donor_name], tuple(range(12)))
                blocks = tuple(range(12))
            editor_model.load_state_dict(edited)
            entry_start = time.perf_counter()
            measured = evaluate(editor_model, tokenizer, tests, device, args.batch_size)
            measured["wall_time_s"] = time.perf_counter() - entry_start
            result["tasks"][key] = {
                "kind": kind,
                "case": case,
                "mode": mode,
                "blocks": list(blocks),
                "n_tensors_swapped": len(changed),
                "metrics": measured,
            }
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            emit(f"{seed} {key} {measured['gap_none_minus_opposite']}")
    emit(f"Completed Pythia Phase 2b output: {OUT_DIR}")


if __name__ == "__main__":
    main()
