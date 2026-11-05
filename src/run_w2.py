"""Run approved W2 in order, retaining all retry events and enforcing the gate.

Usage: python src/run_w2.py [--lens-only]
Each job uses a fresh process; a crashed job restarts once from step 0.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

import argparse
import json
import subprocess
import sys

from common import ROOT
from w2_common import OUT, atomic_json, baseline, event, setup

# Windows: stdout/stderr redirected to a file use the locale codepage and crash on non-ASCII; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def execute(job, script, *args):
    state_path = OUT / "W2_DRIVER.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"failures": {}}
    failures = state["failures"].get(job, 0)
    if failures >= 2:
        raise RuntimeError(f"{job} already crashed twice; stop and tell Sonu")
    logdir = ROOT / "logs/wronglabel_w2"
    logdir.mkdir(parents=True, exist_ok=True)
    while failures < 2:
        event("driver", "job started", job=job, attempt=failures + 1)
        with (logdir / f"{job}_attempt{failures + 1}.log").open("a", encoding="utf-8") as f:
            result = subprocess.run([sys.executable, "-u", str(ROOT / "src" / script), *args],
                                    cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            event("driver", "job finished", job=job)
            return
        failures += 1
        state["failures"][job] = failures
        atomic_json(state_path, state)
        event("driver", "job failed", job=job, exit_code=result.returncode, attempt=failures)
        if failures < 2:
            event("driver", "retrying job from step 0", job=job)
    raise RuntimeError(f"{job} crashed twice; stop and tell Sonu. See logs/wronglabel_w2/")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens-only", action="store_true")
    args = ap.parse_args()
    lock = ROOT / "logs/wronglabel_w2.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents two drivers from using the GPU or result files together.
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    try:
        setup()
        event("driver", "W1 hashes, models, checks and approved GPU verified")
        execute("lens", "logit_lens_wronglabel.py")
        if args.lens_only:
            execute("report", "report_wronglabel.py")
            return
        execute("smoke", "retrain_wronglabel.py", "--smoke")
        for seed in range(3):
            execute(f"swap_seed{seed}", "swap_wronglabel.py", "--seed", str(seed))
        execute("retrain_seed0_all", "retrain_wronglabel.py", "--one", "0:all")
        gate = json.loads((OUT / "retrain/seed0.json").read_text())["runs"]["all"]["metrics"]["M_flipped_vs_wrong_label"]
        if gate > .5 * baseline(0)["M_flipped_vs_wrong_label"]:
            raise RuntimeError("Seed-0 all-block retraining gate FAILED; remaining retraining stopped. Tell Sonu.")
        execute("retrain_seed0_9-10-11", "retrain_wronglabel.py", "--one", "0:9-10-11")
        from retrain_wronglabel import RUN_SETS
        for seed in range(3):
            for name, _, _ in RUN_SETS:
                if seed == 0 and name in ("all", "9-10-11"):
                    continue
                execute(f"retrain_seed{seed}_{name}", "retrain_wronglabel.py", "--one", f"{seed}:{name}")
        execute("report", "report_wronglabel.py")
        event("driver", "W2 complete; results ready for Sonu review")
    except Exception as exc:
        event("driver", "stopped", reason=str(exc))
        subprocess.run([sys.executable, str(ROOT / "src/report_wronglabel.py")], cwd=ROOT, check=False)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
