"""LoRA driver (PROJECT_SPEC.md "Phase L1"). Offline; resumable; a crashed step is restarted once, a second crash = STOP.
Log: logs/run_lora.log

Usage:
  python src/run_lora.py pilot    # 12 seed-0 runs (lr x rank x {shortcut, clean}): train_lora.py, then checks.py;
                                  # one file per run in results/lora_pilot/<run>.json; then `choose`
  python src/run_lora.py choose   # apply the pre-registered rule -> results/lora_pilot/choice.json, and if an lr passes,
                                  # write it into configs/lora.yaml (never typed by hand)
Pilot run names: lora_r<r>_lr<lr>_<kind>_seed0, e.g. lora_r4_lr3e-4_shortcut_seed0 (only used with checks.py --run).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import RESULTS, RUNS  # noqa: E402
from train_lora import load_lora_cfg  # noqa: E402

LOG = ROOT / "logs" / "run_lora.log"
LORA_YAML = ROOT / "configs" / "lora.yaml"
PILOT_DIR = RESULTS / "lora_pilot"
KINDS = ("shortcut", "clean")
FROZEN_CODE = ("src/train.py", "src/train_lora.py", "src/common.py")  # must not change between pilot and main runs
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
RULE = ("An lr passes if, for BOTH rank 4 and rank 64, the shortcut run AND its clean twin pass all their checks. "
        "Pick the smallest lr that passes; the same lr for both ranks. If none passes: STOP and tell Sonu.")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def lr_tag(lr):
    """1e-4 -> '1e-4', 3e-4 -> '3e-4' (as in the run names)."""
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def pilot_runs(lcfg):
    """(name, lr, rank, kind) for the 12 pilot runs, in training order: lr ascending, then rank, then kind."""
    return [(f"lora_r{r}_lr{lr_tag(lr)}_{k}_seed0", lr, r, k)
            for lr in sorted(lcfg["lr_grid"]) for r in lcfg["ranks"] for k in KINDS]


def code_fingerprint():
    out = {f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in FROZEN_CODE}
    try:
        out["git_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                           text=True).stdout.strip() or None
    except OSError:
        out["git_commit"] = None
    return out


def run(args, capture=False):
    """Run src/<script> with args. Streams output into the log (or captures it). Returns (exit code, output, error)."""
    p = subprocess.Popen([sys.executable, str(ROOT / "src" / args[0]), *args[1:]], cwd=ROOT, env=ENV,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT if not capture else subprocess.PIPE,
                         text=True, encoding="utf-8", errors="replace")
    if capture:
        out, err_text = p.communicate()
        lines = err_text.splitlines()
    else:
        out, lines = "", []
        for line in p.stdout:
            line = line.rstrip()
            if "Loading weights" in line or not line:
                continue
            log(line)
            lines = (lines + [line])[-30:]
        p.wait()
    err = next((l for l in reversed(lines) if "Error" in l or "Exception" in l), lines[-1] if lines else "")
    return p.returncode, out, err


def step(label, args, capture=False):
    """Run one step; restart once on crash; STOP the driver on a second crash. Returns captured stdout."""
    log(f"--- {label}: {' '.join(args)}")
    for attempt in (1, 2):
        code, out, err = run(args, capture)
        if code == 0:
            return out
        log(f"CRASH {label} exit {code} (attempt {attempt}): {err}")
        if attempt == 1:
            log(f"restarting {label} once")
    log(f"STOP: {label} crashed twice")
    sys.exit(1)


def trained(name):
    d = RUNS / name
    return (d / "lora_check.json").exists() and (d / "model" / "config.json").exists()


def pilot():
    lcfg = load_lora_cfg()
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    runs = pilot_runs(lcfg)
    log(f"=== run_lora pilot start (offline): {len(runs)} runs, python {sys.executable}")
    for i, (name, lr, rank, kind) in enumerate(runs, 1):
        out = PILOT_DIR / f"{name}.json"
        if out.exists():
            log(f"[{i}/{len(runs)}] {name}: already done, skipped")
            continue
        t0 = time.time()
        if trained(name):
            log(f"[{i}/{len(runs)}] {name}: model already trained, running checks only")
        else:
            step(f"[{i}/{len(runs)}] train {name}", ["train_lora.py", "--run", name, "--lr", repr(lr)])
        s = step(f"[{i}/{len(runs)}] checks {name}", ["checks.py", "--run", name], capture=True)
        checks = json.loads(s[s.index("{"):])
        d = RUNS / name
        rec = {"run": name, "lr": lr, "rank": rank, "kind": kind, "seed": 0, **checks,
               "train": json.load(open(d / "train_stats.json")), "merge_check": json.load(open(d / "lora_check.json")),
               "config": json.load(open(d / "config.json")), "code": code_fingerprint(),
               "driver_wall_time_s": time.time() - t0}
        json.dump(rec, open(out, "w"), indent=2)
        each = ", ".join(f"{k} {'pass' if v['pass'] else 'FAIL'}" for k, v in checks["checks"].items())
        acc = checks["accuracy"]
        log(f"[{i}/{len(runs)}] {name}: acc none {acc['none']:.3f} opp {acc['opposite']:.3f} "
            f"gap {checks['gap_none_minus_opposite']:.3f} -> {'PASS' if checks['all_pass'] else 'FAIL'} ({each}); "
            f"train {rec['train']['wall_time_s'] / 60:.1f} min, peak {rec['train']['peak_vram_allocated_gib']:.2f} GiB")
    log("=== pilot: all runs done")
    return choose()


def choose():
    """Apply the Phase L1 rule to results/lora_pilot/*.json. Returns the chosen lr or None."""
    lcfg = load_lora_cfg()
    runs = pilot_runs(lcfg)
    missing = [n for n, *_ in runs if not (PILOT_DIR / f"{n}.json").exists()]
    if missing:
        log(f"choose: pilot incomplete, missing {missing}")
        sys.exit(1)
    recs = {n: json.load(open(PILOT_DIR / f"{n}.json")) for n, *_ in runs}
    table = {}
    for lr in sorted(lcfg["lr_grid"]):
        by_rank = {}
        for r in lcfg["ranks"]:
            s, c = recs[f"lora_r{r}_lr{lr_tag(lr)}_shortcut_seed0"], recs[f"lora_r{r}_lr{lr_tag(lr)}_clean_seed0"]
            by_rank[f"r{r}"] = {"shortcut_run": s["run"], "shortcut_checks": {k: v["pass"] for k, v in s["checks"].items()},
                                "clean_run": c["run"], "clean_checks": {k: v["pass"] for k, v in c["checks"].items()},
                                "pass": bool(s["all_pass"] and c["all_pass"])}
        table[lr_tag(lr)] = {"lr": lr, "by_rank": by_rank, "pass": all(v["pass"] for v in by_rank.values())}
    passing = [v["lr"] for v in table.values() if v["pass"]]
    chosen = min(passing) if passing else None
    gpus = sorted({r["config"]["gpu"] for r in recs.values()})
    code = sorted({json.dumps({k: v for k, v in r["code"].items() if k != "git_commit"}) for r in recs.values()})
    choice = {"rule": RULE, "lr_grid": sorted(lcfg["lr_grid"]), "table": table, "passing_lrs": passing,
              "chosen_lr": chosen, "gpus": gpus, "all_runs_same_gpu": len(gpus) == 1,
              "all_runs_same_code": len(code) == 1,
              "reason": (f"smallest lr in the grid at which both ranks' shortcut runs and clean twins pass all checks "
                         f"(passing: {[lr_tag(x) for x in passing]})") if chosen is not None else
                        "no lr in the grid passes for both ranks: STOP and tell Sonu (Phase L1 rule)",
              "seed0_main_runs": ({f"lora_r{r}_{k}_seed0": f"lora_r{r}_lr{lr_tag(chosen)}_{k}_seed0"
                                   for r in lcfg["ranks"] for k in KINDS} if chosen is not None else None)}
    json.dump(choice, open(PILOT_DIR / "choice.json", "w"), indent=2)
    for tag, v in table.items():
        log(f"choose: lr {tag}: " + ", ".join(f"{k} {'pass' if b['pass'] else 'FAIL'}" for k, b in v["by_rank"].items())
            + f" -> {'PASS' if v['pass'] else 'fail'}")
    if not choice["all_runs_same_gpu"] or not choice["all_runs_same_code"]:
        log(f"WARNING: pilot runs differ in GPU {gpus} or code: report this to Sonu")
    if chosen is None:
        log("choose: NO lr passes for both ranks. STOP and tell Sonu (pre-registered rule). configs/lora.yaml unchanged.")
        return None
    text = LORA_YAML.read_text(encoding="utf-8")
    new, n = re.subn(r"(?m)^lr: .*$", f"lr: {chosen:.1e}".replace("e-0", "e-") +
                     "                  # chosen by the Phase L1 pilot: see results/lora_pilot/choice.json", text)
    assert n == 1, "configs/lora.yaml: expected exactly one 'lr:' line"
    LORA_YAML.write_text(new, encoding="utf-8")
    log(f"choose: chosen lr {lr_tag(chosen)} (written to configs/lora.yaml); seed-0 main runs = the pilot runs at this lr")
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "choose"])
    args = ap.parse_args()
    chosen = pilot() if args.mode == "pilot" else choose()
    if chosen is None:
        sys.exit(2)  # no lr passes: the pre-registered rule says stop


if __name__ == "__main__":
    main()
