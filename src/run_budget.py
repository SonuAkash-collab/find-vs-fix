"""Phase 3b driver: retrain budget runs, then report.py. Offline.

Usage: python src/run_budget.py          Log: logs/run_budget.log
Each step resumes from saved results. A crashed step is restarted once; a second crash = STOP.
"""
import os
import sys
import time

import run_night2
from run_night2 import ROOT, log, step

# Windows: stdout/stderr redirected to a file use the locale codepage and crash on non-ASCII; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

run_night2.LOG = ROOT / "logs" / "run_budget.log"


def main():
    (ROOT / "logs" / "run_budget.pid").write_text(str(os.getpid()))
    t0 = time.time()
    log(f"=== run_budget start (offline), pid {os.getpid()}, python {sys.executable}")
    step("Phase 3b budget runs", ["retrain.py", "--budget"])
    step("report", ["report.py"])
    log(f"=== run_budget done: total {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
