"""Night 2 driver: Phase 2b swaps, Phase 3 gate, rest of Phase 3, then report.py. Offline.

Usage: python src/run_night2.py          Log: logs/run_night2.log
Each step resumes from saved results. A crashed step is restarted once; a second crash = STOP.
GATE (PROJECT_SPEC.md Phase 3): if retraining "all" blocks of shortcut_seed0 does not bring the gap to
<= 50% of the unswapped gap, STOP Phase 3 (no fixes), write the report, exit.
"""
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Windows: stdout/stderr redirected to a file use the locale codepage and crash on non-ASCII; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "run_night2.log"
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
       "PYTHONUTF8": "1"}
GATE_MAX_FRAC = 0.50


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def run(args):
    """Run src/<script> with args, stream output into the log; return (exit code, last error line)."""
    p = subprocess.Popen([sys.executable, str(ROOT / "src" / args[0]), *args[1:]], cwd=ROOT, env=ENV,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if "Loading weights" in line or not line:
            continue
        log(line)
        tail = (tail + [line])[-30:]
    p.wait()
    err = next((l for l in reversed(tail) if "Error" in l or "Exception" in l), tail[-1] if tail else "")
    return p.returncode, err


def step(label, args):
    """Run one step; restart once on crash; stop the driver on a second crash."""
    log(f"--- {label}: {' '.join(args)}")
    t0 = time.time()
    for attempt in (1, 2):
        code, err = run(args)
        if code == 0:
            log(f"--- {label} finished ({(time.time() - t0) / 60:.1f} min)")
            return
        log(f"CRASH {label} exit {code} (attempt {attempt}): {err}")
        if attempt == 1:
            log(f"restarting {label} once (it resumes from saved results)")
    log(f"STOP: {label} crashed twice")
    sys.exit(1)


def main():
    # keep Windows from idle-sleeping while this process runs (does not change any setting; ends with the process)
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as e:  # noqa: BLE001
        log(f"note: could not request keep-awake: {e}")
    (ROOT / "logs" / "run_night2.pid").write_text(str(os.getpid()))
    t_start = time.time()
    log(f"=== run_night2 start (offline), pid {os.getpid()}, python {sys.executable}")

    step("Phase 2b swaps", ["swap.py", "--phase2b"])
    step("Phase 3 gate runs (seed 0: all, 9-10-11)", ["retrain.py", "--seeds", "0", "--sets", "all,9-10-11"])

    r = json.load(open(ROOT / "results" / "retrain" / "seed0.json"))
    G = r["baseline"]["gap_none_minus_opposite"]
    g_all = r["runs"]["all"]["gap_none_minus_opposite"]
    ok = g_all <= GATE_MAX_FRAC * G
    log(f"GATE: seed 0 retrain 'all' gap {g_all:.3f} vs unswapped gap {G:.3f} "
        f"(limit {GATE_MAX_FRAC:.0%} = {GATE_MAX_FRAC * G:.3f}) -> {'PASS' if ok else 'FAIL'}")
    if ok:
        step("Phase 3 all runs", ["retrain.py"])
    else:
        log("STOP Phase 3: gate failed. No fixes tried (no extra epochs, no resets). Writing report.")
    step("report", ["report.py"])
    log(f"=== run_night2 done{'' if ok else ' (Phase 3 stopped at the gate)'}: "
        f"total {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
