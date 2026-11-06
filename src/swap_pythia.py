"""Pythia-160M Phase 2 activation swaps.

Separate from the GPT-2 swap implementation.  This script only performs
forward-pass swaps on the six existing official Pythia checkpoints.

Official Mac command:
  python src/swap_pythia.py --device mps --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .logit_lens_pythia import DATA_DIR, MODEL_ID, VERSIONS, load_tests
except ImportError:  # direct execution: python src/swap_pythia.py
    from logit_lens_pythia import DATA_DIR, MODEL_ID, VERSIONS, load_tests

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs_pythia"
OUT_DIR = ROOT / "results" / "pythia_analysis" / "swap"
BLOCKS = tuple(range(12))
BLOCK_SETS = tuple((b,) for b in BLOCKS) + ((10, 11), (9, 10, 11))
OFFICIAL_RUN_WALL_TIME_NOTE = (
    "6h 31m 39s total across the interrupted and resumed official Phase 2 invocations; "
    "from logs/run_swap.log"
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def run_name(kind: str, seed: int) -> str:
    return f"{kind}_seed{seed}_lr5e-5"


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def block_key(blocks: tuple[int, ...]) -> str:
    return "-".join(str(b) for b in blocks)


def block_keys(state_dict: dict[str, torch.Tensor], blocks: tuple[int, ...]) -> list[str]:
    prefixes = tuple(f"gpt_neox.layers.{b}." for b in blocks)
    return [k for k in state_dict if k.startswith(prefixes)]


def swapped_state(editor: dict[str, torch.Tensor], donor: dict[str, torch.Tensor], blocks: tuple[int, ...]):
    result = {k: v.clone() for k, v in editor.items()}
    keys = block_keys(result, blocks)
    expected = 12 * len(blocks)
    if len(keys) != expected:
        raise AssertionError(f"unexpected Pythia block tensor count: got {len(keys)}, expected {expected}")
    for key in keys:
        result[key] = donor[key].clone()
    return result, keys


def other_seed(seed: int, seeds: list[int]) -> int:
    return seeds[(seeds.index(seed) + 1) % len(seeds)]


def cases(seed: int, seeds: list[int]):
    donor_seed = other_seed(seed, seeds)
    return {
        "a_shortcut_from_clean_same": (run_name("shortcut", seed), run_name("clean", seed)),
        "b_clean_from_shortcut_same": (run_name("clean", seed), run_name("shortcut", seed)),
        "ci_shortcut_from_clean_other": (run_name("shortcut", seed), run_name("clean", donor_seed)),
        "cii_shortcut_from_shortcut_other": (run_name("shortcut", seed), run_name("shortcut", donor_seed)),
    }


@torch.inference_mode()
def evaluate(model, tokenizer, tests, device: torch.device, batch_size: int) -> dict:
    pos = tokenizer.encode(" positive", add_special_tokens=False)
    neg = tokenizer.encode(" negative", add_special_tokens=False)
    if len(pos) != 1 or len(neg) != 1:
        raise ValueError(f"labels must be single tokens: {pos}, {neg}")
    scores = {}
    for version in VERSIONS:
        values = []
        rows = tests[version]
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            enc = tokenizer([f"Review: {r['text']}\nSentiment:" for r in batch],
                            padding=True, truncation=False, return_tensors="pt")
            lengths = enc["attention_mask"].sum(dim=1)
            if torch.any(lengths > 160):
                raise ValueError(f"{version}: prompt exceeds max_len=160")
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model.gpt_neox(**enc).last_hidden_state
            last = enc["attention_mask"].sum(dim=1) - 1
            idx = torch.arange(len(batch), device=device)
            h = hidden[idx, last]
            logits = model.get_output_embeddings()(h)
            values.extend((logits[:, pos[0]] - logits[:, neg[0]]).float().cpu().tolist())
        d = np.asarray(values, dtype=np.float64)
        labels = np.asarray([r["label"] for r in rows])
        scores[version] = {
            "n": len(rows),
            "accuracy": float(np.mean((d > 0).astype(int) == labels)),
            "mean_d": float(d.mean()),
        }
    acc = {v: scores[v]["accuracy"] for v in VERSIONS}
    scores["gap_none_minus_opposite"] = acc["none"] - acc["opposite"]
    scores["accuracy"] = acc
    return scores


def load_checkpoint(name: str, device: torch.device):
    path = RUNS_DIR / name / "model"
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"official checkpoint missing: {path}")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    if model.dtype != torch.float32:
        raise AssertionError(f"expected FP32 checkpoint, got {model.dtype}")
    return model, path


def pilot_baseline(name: str) -> dict:
    path = ROOT / "results" / "pythia_lr_pilot" / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"pilot baseline missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_matches_pilot(name: str, measured: dict) -> None:
    expected = pilot_baseline(name)
    for version in VERSIONS:
        actual = measured["accuracy"][version]
        reference = expected["accuracy"][version]
        if actual != reference:
            raise AssertionError(f"{name} {version}: {actual} != pilot {reference}")
    if measured["gap_none_minus_opposite"] != expected["gap_none_minus_opposite"]:
        raise AssertionError(
            f"{name} gap {measured['gap_none_minus_opposite']} != "
            f"pilot {expected['gap_none_minus_opposite']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("mps", "cpu"), default="mps")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="one seed, one block, case (a); not official")
    ap.add_argument("--baseline-only", action="store_true", help="evaluate six unswapped models against pilot checks")
    ap.add_argument("--repro", action="store_true", help="re-evaluate seed 0 case (a) blocks 0 and 9-10-11")
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    if sum((args.smoke, args.baseline_only, args.repro)) > 1:
        ap.error("--smoke, --baseline-only, and --repro are mutually exclusive")
    seeds = [int(x) for x in args.seeds.split(",")]
    if sorted(set(seeds)) != seeds or any(s not in (0, 1, 2) for s in seeds):
        ap.error("--seeds must be an increasing subset of 0,1,2")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing official Pythia swaps")
    tests = load_tests(DATA_DIR)
    tokenizer = AutoTokenizer.from_pretrained(RUNS_DIR / run_name("shortcut", seeds[0]) / "model")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    names = [run_name(kind, seed) for kind in ("shortcut", "clean") for seed in (0, 1, 2)]
    states = {}
    for name in names:
        model, _ = load_checkpoint(name, torch.device("cpu"))
        states[name] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        del model
    editor_model, _ = load_checkpoint(names[0], device)
    editor_model.to(device).eval()
    if args.baseline_only:
        started = time.perf_counter()
        baseline = {
            "phase": "Pythia Phase 2 unswapped baseline",
            "model_id": MODEL_ID,
            "data_dir": repo_relative(DATA_DIR),
            "device": str(device),
            "dtype": "torch.float32",
            "git_commit": git_commit(),
            "start_time": now_iso(),
            "runs": {},
        }
        for name in names:
            model, _ = load_checkpoint(name, device)
            model.to(device).eval()
            entry_start = time.perf_counter()
            measured = evaluate(model, tokenizer, tests, device, args.batch_size)
            measured["wall_time_s"] = time.perf_counter() - entry_start
            assert_matches_pilot(name, measured)
            baseline["runs"][name] = measured
            del model
        baseline["total_wall_time_s"] = time.perf_counter() - started
        output = args.output_dir.parent / "baseline.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print("Baseline matches all six pilot files:", output)
        return
    run_seeds = [0] if (args.smoke or args.repro) else seeds
    run_sets = ((0,),) if args.smoke else ((0,), (9, 10, 11)) if args.repro else BLOCK_SETS
    run_cases = ("a_shortcut_from_clean_same",)
    repro_stored = {}
    if args.repro:
        existing = args.output_dir / "seed0.json"
        if not existing.is_file():
            raise FileNotFoundError(f"existing Phase 2 result missing: {existing}")
        old = json.loads(existing.read_text(encoding="utf-8"))
        old_blocks = old["cases"]["a_shortcut_from_clean_same"]["by_blocks"]
        repro_stored = {key: old_blocks[key]["gap_none_minus_opposite"] for key in ("0", "9-10-11")}
    repro_rerun = {}
    start_time = now_iso()
    for seed in run_seeds:
        path = args.output_dir / f"seed{seed}.json"
        result = json.loads(path.read_text()) if path.exists() else {
            "phase": "Pythia Phase 2 activation swaps",
            "seed": seed,
            "other_seed": other_seed(seed, seeds),
            "model_id": MODEL_ID,
            "data_dir": repo_relative(DATA_DIR),
            "device": str(device),
            "dtype": "torch.float32",
            "git_commit": git_commit(),
            "start_time": start_time,
            "cases": {},
            "official_run_wall_time_note": OFFICIAL_RUN_WALL_TIME_NOTE,
        }
        result["data_dir"] = repo_relative(DATA_DIR)
        result["official_run_wall_time_note"] = OFFICIAL_RUN_WALL_TIME_NOTE
        for case_name in run_cases:
            editor_name, donor_name = cases(seed, seeds)[case_name]
            result["cases"].setdefault(case_name, {"editor": editor_name, "donor": donor_name, "by_blocks": {}})
            for blocks in run_sets:
                key = block_key(blocks)
                if key in result["cases"][case_name]["by_blocks"] and not args.repro:
                    continue
                edited, keys = swapped_state(states[editor_name], states[donor_name], blocks)
                editor_model.load_state_dict(edited)
                entry_start = time.perf_counter()
                measured = evaluate(editor_model, tokenizer, tests, device, args.batch_size)
                measured["wall_time_s"] = time.perf_counter() - entry_start
                if args.repro:
                    repro_rerun[key] = measured["gap_none_minus_opposite"]
                measured.update({"blocks": list(blocks), "n_tensors_swapped": len(keys)})
                result["cases"][case_name]["by_blocks"][key] = measured
                args.output_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(seed, case_name, key, measured["gap_none_minus_opposite"], flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.repro:
        checks = {}
        for key in ("0", "9-10-11"):
            checks[key] = {
                "stored_gap": repro_stored[key],
                "rerun_gap": repro_rerun[key],
                "identical": repro_stored[key] == repro_rerun[key],
            }
        if not all(item["identical"] for item in checks.values()):
            raise AssertionError(f"reproducibility mismatch: {checks}")
        repro = {
            "phase": "Pythia Phase 2 reproducibility check",
            "git_commit": git_commit(),
            "device": str(device),
            "dtype": "torch.float32",
            "data_dir": repo_relative(DATA_DIR),
            "start_time": now_iso(),
            "checks": checks,
            "note": "Gaps independently recomputed after the required post-code-commit rerun.",
        }
        (args.output_dir / "repro.json").write_text(json.dumps(repro, indent=2) + "\n", encoding="utf-8")
    print("Completed Pythia Phase 2 swap output:", args.output_dir)


if __name__ == "__main__":
    main()
