"""Phase 2 driver: run src/swap.py (resumable) then src/report.py, offline. Log: logs/run_swap.log

Usage: python src/run_swap.py
If swap.py crashes it is restarted once (it resumes); if the same error happens again, stop.
"""
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
LOG = ROOT / "logs" / "run_swap.log"
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
       "PYTHONUTF8": "1"}


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def run(script):
    """Run a script, stream its output into the log; return (exit code, last error line)."""
    p = subprocess.Popen([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, env=ENV,
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


def main():
    log("=== run_swap start (offline)")
    last_err = None
    while True:
        code, err = run("swap.py")
        if code == 0:
            break
        log(f"CRASH swap.py exit {code}: {err}")
        if err == last_err:
            log("STOP: same error twice in a row")
            sys.exit(1)
        last_err = err
        log("restarting swap.py (it resumes from saved cases)")
    code, err = run("report.py")
    if code != 0:
        log(f"STOP: report.py failed exit {code}: {err}")
        sys.exit(1)
    log("=== run_swap done: swap + report finished")


if __name__ == "__main__":
    main()
