"""Phase 4 driver: activation patching on all 6 full fine-tuning models (shortcut + clean, seeds 0-2). Offline.

Usage: python src/run_patching.py          Log: logs/patching.log
       python src/run_patching.py --prefix lora_r4 [--adapters DIR]   Log: logs/patching_lora_r4.log
--prefix / --adapters are passed through to src/patching.py (results in results/<prefix>/patching/).
One step per seed (shortcut, then clean). Finished (seed, model) entries are skipped, so a re-run resumes.
A crashed step is restarted once; a second crash = STOP. Then the Phase 4 verdict (all 3 seeds) is logged.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "patching.log"
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def run(args):
    """Run src/patching.py with args, stream output into the log; return (exit code, last error line)."""
    p = subprocess.Popen([sys.executable, str(ROOT / "src" / "patching.py"), *args], cwd=ROOT, env=ENV,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if "Loading weights" in line or not line:
            continue
        log(line)
        tail = (tail + [line])[-30:]
    p.wait()
    err = next((l for l in reversed(tail) if "Error" in l or "STOP" in l), tail[-1] if tail else "")
    return p.returncode, err


def step(label, args):
    """Run one step; restart once on crash; stop the driver on a second crash or on a STOP."""
    log(f"--- {label}: patching.py {' '.join(args)}")
    t0 = time.time()
    for attempt in (1, 2):
        code, err = run(args)
        if code == 0:
            log(f"--- {label} finished ({(time.time() - t0) / 60:.1f} min)")
            return
        log(f"CRASH {label} exit {code} (attempt {attempt}): {err}")
        if "STOP" in err:
            break  # a failed pre-registered check is not a crash: never retried
        if attempt == 1:
            log(f"restarting {label} once (it resumes from saved results)")
    log(f"STOP: {label} failed")
    sys.exit(1)


def main():
    global LOG
    sys.path.insert(0, str(ROOT / "src"))
    from common import prefix_arg, results_dir
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=prefix_arg, help="run family, e.g. lora_r4 (passed to patching.py)")
    ap.add_argument("--adapters", type=Path, help="with --prefix: folder of unmerged LoRA adapters (passed through)")
    a = ap.parse_args()
    if a.adapters and not a.prefix:
        ap.error("--adapters needs --prefix")
    extra = (["--prefix", a.prefix] if a.prefix else []) + (["--adapters", str(a.adapters.resolve())] if a.adapters else [])
    name = f"patching_{a.prefix}" if a.prefix else "patching"
    LOG = ROOT / "logs" / f"{name}.log"
    # keep Windows from idle-sleeping while this process runs (does not change any setting; ends with the process)
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as e:  # noqa: BLE001
        log(f"note: could not request keep-awake: {e}")
    LOG.parent.mkdir(exist_ok=True)
    (ROOT / "logs" / f"{name}.pid").write_text(str(os.getpid()))
    t_start = time.time()
    log(f"=== run_patching start (offline), pid {os.getpid()}, python {sys.executable}" + (f", {' '.join(extra)}" if extra else ""))
    for s in (0, 1, 2):
        step(f"seed {s}", ["--seeds", str(s), "--models", "shortcut", "clean", *extra])

    from patching import verdict
    v = verdict(results_dir("patching", a.prefix))
    log(f"Phase 4 verdict (all 3 seeds needed): {v['verdict']}; arrived at (seeds 0/1/2) {v['arrived_at']}; "
        f"F(notlast) L0-4 per seed {json.dumps([[None if x is None else round(x, 3) for x in s] for s in v['F_notlast_L0_to_4']])}")
    log(f"=== run_patching done: total {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
