"""Driver for the whole pilot, start to finish, offline:
6 trainings (+ checks) -> logit lens -> figures + RESULTS.md -> MORNING_REPORT.md.

Usage: python src/run_pilot.py
Resumable: skips runs whose metrics.json exists and lens files that exist.
Overnight rules (PROJECT_SPEC.md): shortcut_seed0 and clean_seed0 run first. If ONLY "shortcut_learned" fails there,
the one allowed fallback (10% name prevalence) is applied automatically, once. Any other failure stops the driver.
Exit codes: 0 done; 2 checks failed; 1 crash.
Log: logs/run_pilot.log
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONUTF8"] = "1"

import json
import logging
import shutil
import subprocess
import sys
import time
import traceback

import torch

from checks import base_check, run_checks
from common import DATA, LOGS, RESULTS, ROOT, RUNS, load_cfg
from train import train

# Windows: stdout/stderr redirected to a file use the locale codepage and crash on non-ASCII; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

FALLBACK_PREVALENCE = 0.10


def setup_log():
    LOGS.mkdir(exist_ok=True)
    log = logging.getLogger("pilot")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(LOGS / "run_pilot.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def one_run(name, cfg, log):
    kind, seed = name.split("_seed")
    out = RUNS / name
    run_cfg = dict(cfg)
    while True:
        try:
            model, tok, stats = train(name, kind, int(seed), run_cfg, out, log=log.info)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if run_cfg["batch_size"] == 1:
                raise
            run_cfg = {**run_cfg, "batch_size": run_cfg["batch_size"] // 2, "grad_accum": run_cfg["grad_accum"] * 2}
            log.info(f"[{name}] OOM -> batch_size {run_cfg['batch_size']}, grad_accum {run_cfg['grad_accum']} "
                     f"(effective batch stays {run_cfg['batch_size'] * run_cfg['grad_accum']})")
    res = run_checks(model, tok, kind, run_cfg)
    metrics = {"run": name, **res, "train": stats,
               "batch_size": run_cfg["batch_size"], "grad_accum": run_cfg["grad_accum"]}
    (RESULTS / "checks").mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(RESULTS / "checks" / f"{name}.json", "w"), indent=2)
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)  # written last = run complete
    log.info(f"[{name}] acc {json.dumps({k: round(v, 4) for k, v in res['accuracy'].items()})} "
             f"gap {res['gap_none_minus_opposite']:.4f} wall {stats['wall_time_s']:.0f}s "
             f"peakVRAM {stats['peak_vram_allocated_gib']:.2f}GiB")
    for k, c in res["checks"].items():
        log.info(f"[{name}] CHECK {k}: {'PASS' if c['pass'] else 'FAIL'} value={c['value']:.4f} rule: {c['rule']}")
    del model
    torch.cuda.empty_cache()
    return metrics


def train_all(cfg, log):
    """Returns (ok, failed_metrics). Gate = first two runs, both run before deciding."""
    order = cfg["run_order"]
    gate = order[:2]
    failed = []
    for name in order:
        mpath = RUNS / name / "metrics.json"
        if mpath.exists():
            m = json.load(open(mpath))
            log.info(f"[{name}] skip (metrics.json exists, all_pass={m['all_pass']})")
        else:
            log.info(f"[{name}] start")
            m = one_run(name, cfg, log)
        if not m["all_pass"]:
            failed.append(m)
        if failed and (name not in gate or name == gate[-1]):
            return False, failed
    return True, []


def fallback_allowed(failed, cfg):
    man = json.load(open(DATA / "manifest.json"))
    if man["name_prevalence"] != 0.05 or len(failed) != 1:
        return False
    m = failed[0]
    bad = [k for k, c in m["checks"].items() if not c["pass"]]
    return m["run"] == cfg["run_order"][0] and bad == ["shortcut_learned"]


def apply_fallback(cfg, failed, log):
    m = failed[0]
    log.info(f"FALLBACK: only shortcut_learned failed on {m['run']} "
             f"(gap {m['gap_none_minus_opposite']:.4f}). Regenerating data at prevalence {FALLBACK_PREVALENCE}.")
    # keep the failed 5% evidence before deleting the runs
    keep = RESULTS / "fallback_5pct"
    keep.mkdir(parents=True, exist_ok=True)
    for name in cfg["run_order"][:2]:
        src = RESULTS / "checks" / f"{name}.json"
        if src.exists():
            shutil.move(str(src), keep / f"{name}.json")
    shutil.copy(DATA / "manifest.json", keep / "manifest_5pct.json")
    subprocess.run([sys.executable, str(ROOT / "src" / "make_data.py"), "--prevalence", str(FALLBACK_PREVALENCE)],
                   check=True, stdout=open(LOGS / "make_data_fallback.log", "w"), stderr=subprocess.STDOUT)
    man = json.load(open(DATA / "manifest.json"))
    man["fallback"] = (f"prevalence raised 0.05 -> {FALLBACK_PREVALENCE} because shortcut_learned failed on "
                       f"{m['run']} (gap {m['gap_none_minus_opposite']:.4f} < 0.20); 5% results kept in "
                       f"results/fallback_5pct/")
    json.dump(man, open(DATA / "manifest.json", "w"), indent=2)
    for name in cfg["run_order"][:2]:
        if (RUNS / name).exists():
            shutil.rmtree(RUNS / name)
    log.info("FALLBACK: data regenerated, manifest updated, shortcut_seed0/clean_seed0 deleted; restarting training")


def write_morning_report(status, detail, cfg):
    """Automatic report from files only (a fuller report may replace it later)."""
    o = ["# MORNING REPORT (written automatically by src/run_pilot.py)", "",
         f"Status: **{status}**", "", detail, ""]
    man = json.load(open(DATA / "manifest.json"))
    if man.get("fallback"):
        o += [f"Fallback applied: {man['fallback']}", ""]
    o += ["## Runs", "", "| run | acc none | acc opposite | gap | checks | wall time (s) | peak VRAM (GiB) |",
          "|---|---|---|---|---|---|---|"]
    for name in cfg["run_order"]:
        p = RESULTS / "checks" / f"{name}.json"
        if not p.exists():
            o.append(f"| {name} | not run | | | | | |")
            continue
        m = json.load(open(p))
        cs = "; ".join(f"{k} {'PASS' if c['pass'] else 'FAIL'}" for k, c in m["checks"].items())
        o.append(f"| {name} | {m['accuracy']['none']:.3f} | {m['accuracy']['opposite']:.3f} | "
                 f"{m['gap_none_minus_opposite']:.3f} | {cs} | {m['train']['wall_time_s']:.0f} | "
                 f"{m['train']['peak_vram_allocated_gib']:.2f} |")
    o.append("")
    if (ROOT / "RESULTS.md").exists():
        o += ["Full results: [RESULTS.md](RESULTS.md). Figures: [lens_accuracy.png](figures/lens_accuracy.png), "
              "[follow_rate.png](figures/follow_rate.png).", ""]
    tail = (LOGS / "run_pilot.log").read_text(encoding="utf-8").splitlines()[-40:]
    o += ["## Last 40 lines of logs/run_pilot.log", "", "```", *tail, "```", ""]
    (ROOT / "MORNING_REPORT.md").write_text("\n".join(o), encoding="utf-8")


def main():
    cfg = load_cfg()
    log = setup_log()
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    log.info(f"=== driver start (offline); order {cfg['run_order']}")
    try:
        if not (RESULTS / "base_gpt2_name_check.json").exists():
            b = base_check(cfg)
            json.dump(b, open(RESULTS / "base_gpt2_name_check.json", "w"), indent=2)
            log.info(f"[base] mean d {json.dumps({k: round(v, 4) for k, v in b['mean_d'].items()})} "
                     f"pairs {json.dumps({k: round(v['mean_diff'], 4) for k, v in b['pairs'].items()})} "
                     f"std across reviews {json.dumps({k: round(v, 4) for k, v in b['std_d_across_reviews'].items()})}")

        ok, failed = train_all(cfg, log)
        if not ok and fallback_allowed(failed, cfg):
            apply_fallback(cfg, failed, log)
            ok, failed = train_all(cfg, log)
        if not ok:
            msg = "STOP: failed checks: " + "; ".join(
                f"{m['run']}: " + ", ".join(f"{k}={c['value']:.4f} ({c['rule']})" for k, c in m["checks"].items()
                                           if not c["pass"]) for m in failed)
            log.info(msg)
            write_morning_report("STOPPED (checks failed)", msg, cfg)
            sys.exit(2)
        log.info("all 6 runs passed their checks")

        from logit_lens import lens_run
        for name in cfg["run_order"]:
            if (RESULTS / "logit_lens" / f"{name}.json").exists():
                log.info(f"[lens] {name} skip (exists)")
                continue
            out = lens_run(name, cfg)
            log.info(f"[lens] {name} cal_acc opposite {[round(x, 3) for x in out['cal_acc']['opposite']]}")

        subprocess.run([sys.executable, str(ROOT / "src" / "report.py")], check=True)
        log.info(f"=== driver done in {time.time() - t0:.0f}s: RESULTS.md and figures written")
        write_morning_report("FINISHED", "All 6 runs passed their checks; logit lens, figures and RESULTS.md done.", cfg)
    except SystemExit:
        raise
    except BaseException:
        tb = traceback.format_exc()
        log.info("CRASH:\n" + tb)
        write_morning_report("CRASHED", "Exact error:\n\n```\n" + tb + "```", cfg)
        sys.exit(1)


if __name__ == "__main__":
    main()
