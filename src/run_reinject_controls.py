"""Phase 6b driver (PROJECT_SPEC.md "Phase 6b: controls for Phase 6"). Offline, inference only.

Usage: python src/run_reinject_controls.py      Log: logs/reinject_controls.log (pid in logs/reinject_controls.pid)
Steps (each resumes: a seed whose result file exists is skipped):
  1. full FT: src/reinject_controls.py --seeds s for s = 0, 1, 2 -> results/reinject_controls/seed<s>.json
  2. RESULTS_REINJECT_CONTROLS.md (src/report_reinject_controls.py), from the result files only.
A crashed step is restarted once; a second crash = STOP. A failed pre-registered check ("STOP" in the output) is never
retried. Exit codes: 0 done, 1 STOP / crash.
"""
import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

# Windows: stdout/stderr redirected to a file use the locale codepage and crash on non-ASCII; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a TextIOWrapper (e.g. redirected by a test harness)
        pass

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "reinject_controls.log"
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
       "PYTHONUTF8": "1"}
SEEDS = (0, 1, 2)


def rel_paths(msg):
    """Paths inside the repo as relative to the repo root (.\\src\\x.py), so no user's home folder ends up in the log."""
    for root in dict.fromkeys((str(ROOT), ROOT.as_posix())):
        msg = msg.replace(root, ".")
    return msg


def log(msg):
    msg = rel_paths(str(msg))
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def run(args):
    """Run src/<script> with args, stream its output into the log; return (exit code, last error line)."""
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
    err = next((l for l in reversed(tail) if "STOP" in l or "Error" in l or "Exception" in l), tail[-1] if tail else "")
    return p.returncode, err


def step(label, args):
    """One step; restarted once on a crash; a STOP (failed check) is not retried. Exits the driver on failure."""
    log(f"--- {label}: {' '.join(args)}")
    t0 = time.time()
    for attempt in (1, 2):
        code, err = run(args)
        if code == 0:
            log(f"--- {label} finished ({(time.time() - t0) / 60:.1f} min)")
            return
        log(f"CRASH {label} exit {code} (attempt {attempt}): {err}")
        if "STOP" in err:
            break
        if attempt == 1:
            log(f"restarting {label} once (finished seeds are skipped)")
    log(f"STOP: {label} failed")
    sys.exit(1)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:  # keep Windows from idle-sleeping while this process runs (ends with the process)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as e:  # noqa: BLE001
        log(f"note: could not request keep-awake: {e}")
    LOG.parent.mkdir(exist_ok=True)
    (ROOT / "logs" / "reinject_controls.pid").write_text(str(os.getpid()))
    t0 = time.time()
    log(f"=== run_reinject_controls start (offline), pid {os.getpid()}, python {sys.executable}")
    for s in SEEDS:
        step(f"full FT seed {s}", ["reinject_controls.py", "--seeds", str(s)])
    step("report", ["report_reinject_controls.py"])
    log(f"=== run_reinject_controls done: total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
