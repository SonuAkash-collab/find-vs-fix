"""Phase 5 driver (PROJECT_SPEC.md "Phase 5: a less obvious shortcut: word choice"), start to finish, offline.

Usage: python src/run_wordcue.py            Log: logs/wordcue.log (pid in logs/wordcue.pid)
Steps (each resumes from saved files; a crashed step is restarted once, a second crash = STOP; a failed
pre-registered check ("STOP" in the step's output) is never retried):
  1. data: src/make_wordcue.py (rebuild must match data_wordcue/manifest.json if it holds hashes), then --verify
  2. base GPT-2 on film vs movie forms (reported, no rule): checks.py --prefix wordcue --base
  3. wordcue_shortcut_seed0 + wordcue_clean_seed0 (full FT, configs/pilot.yaml, data_wordcue/), checks.
     GATE: if ONLY "shortcut learned" fails (gap < 0.20) on shortcut_seed0: the one fallback, once (10% prevalence:
     data rebuilt, the 5% checks kept in results/wordcue/fallback_5pct/, both seed-0 runs redone). Still failing, or
     anything else failing: STOP (exit 2), report written.
  4. seeds 1, 2 (+ checks; a failure = STOP, exit 2)
  5. logit lens (6 runs)   6. Phase 2 swaps (a, b, c-i, c-ii; singles 0-11, 10-11, 9-10-11)
  7. Phase 3 retraining on data_wordcue/train_clean.jsonl: seed 0 "all" + "9-10-11" first; GATE: "all" must bring the
     gap to <= 50% of G_0, else Phase 3 stops there (no fixes) and the driver goes on
  8. Phase 4 patching in cue mode (shortcut + clean, per seed)   9. RESULTS_WORDCUE.md (src/report_wordcue.py)
Results: runs/wordcue_*, results/wordcue/. Exit codes: 0 done (also when the Phase 3 gate failed), 2 checks failed
(STOP), 1 crash.
Internal: python src/run_wordcue.py train <kind>_seed<k>   (one training run + checks; used by step 3/4)
"""
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
PREFIX = "wordcue"
DATA_W = ROOT / "data_wordcue"
RES = ROOT / "results" / PREFIX
RUNS = ROOT / "runs"
LOG = ROOT / "logs" / "wordcue.log"
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
PREVALENCE, FALLBACK_PREVALENCE = 0.05, 0.10
GATE_MAX_FRAC = 0.50
SEEDS = (0, 1, 2)
KINDS = ("shortcut", "clean")
VRAM_LIMIT_GIB = 4.5


def log(msg):
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
    err = next((l for l in reversed(tail) if "STOP" in l or "Error" in l or "Exception" in l or "FAILED" in l),
               tail[-1] if tail else "")
    return p.returncode, err


class Stop(Exception):
    """A pre-registered STOP (failed check) or a second crash."""

    def __init__(self, msg, code):
        super().__init__(msg)
        self.code = code


def step(label, args):
    """One step; restarted once on a crash; a failed check (STOP / FAILED / MISMATCH in its output) is not retried."""
    log(f"--- {label}: {' '.join(args)}")
    t0 = time.time()
    for attempt in (1, 2):
        code, err = run(args)
        if code == 0:
            log(f"--- {label} finished ({(time.time() - t0) / 60:.1f} min)")
            return
        log(f"CRASH {label} exit {code} (attempt {attempt}): {err}")
        if any(k in err for k in ("STOP", "FAILED", "MISMATCH")):
            raise Stop(f"{label}: {err}", 2)  # a pre-registered check failed: never retried
        if attempt == 1:
            log(f"restarting {label} once (it resumes from saved results)")
    raise Stop(f"{label} crashed twice: {err}", 1)


def manifest():
    return json.load(open(DATA_W / "manifest.json", encoding="utf-8"))


# ---- one training run + checks (subprocess: python src/run_wordcue.py train shortcut_seed0)
def train_cmd(short):
    import torch
    from checks import run_checks
    from common import load_cfg, rel_root, resolve_data, run_env, run_kind
    from train import train
    cfg = load_cfg()
    data_dir = resolve_data(None, PREFIX)
    name = f"{PREFIX}_{short}"
    kind, seed = run_kind(name), int(name.rsplit("_seed", 1)[1])
    out = RUNS / name
    if (out / "metrics.json").exists():
        print(f"[{name}] skip (metrics.json exists)")
        return
    man = manifest()
    t0 = time.time()
    run_cfg = dict(cfg)
    while True:  # overnight rule 5: on OOM halve the batch and double accumulation (effective batch stays 16)
        try:
            model, tok, stats = train(name, kind, seed, run_cfg, out, log=lambda m: print(m, flush=True),
                                      data_dir=data_dir)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if run_cfg["batch_size"] == 1:
                raise
            run_cfg = {**run_cfg, "batch_size": run_cfg["batch_size"] // 2, "grad_accum": run_cfg["grad_accum"] * 2}
            print(f"[{name}] OOM -> batch_size {run_cfg['batch_size']}, grad_accum {run_cfg['grad_accum']}", flush=True)
    res = run_checks(model, tok, kind, run_cfg, data_dir=data_dir)
    metrics = {"run": name, **res, "train": stats, "batch_size": run_cfg["batch_size"],
               "grad_accum": run_cfg["grad_accum"], "data": rel_root(data_dir), "prevalence": man["prevalence"],
               "train_file_sha256": man["sha256"][cfg["train_files"][kind]],
               "env": {**run_env(), "wall_time_s_incl_checks": round(time.time() - t0, 1)}}
    (RES / "checks").mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(RES / "checks" / f"{name}.json", "w"), indent=2)
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)  # written last = run complete
    print(f"[{name}] acc {json.dumps({k: round(v, 4) for k, v in res['accuracy'].items()})} gap "
          f"{res['gap_none_minus_opposite']:.4f} wall {stats['wall_time_s']:.0f}s peakVRAM "
          f"{stats['peak_vram_allocated_gib']:.2f}GiB (reserved {stats['peak_vram_reserved_gib']:.2f})", flush=True)
    if stats["peak_vram_reserved_gib"] > VRAM_LIMIT_GIB:
        print(f"[{name}] NOTE: peak reserved VRAM above {VRAM_LIMIT_GIB} GiB", flush=True)
    for k, c in res["checks"].items():
        print(f"[{name}] CHECK {k}: {'PASS' if c['pass'] else 'FAIL'} value={c['value']:.4f} rule: {c['rule']}",
              flush=True)


def metrics_of(short):
    """Saved metrics of a run, checked against the current data (a run trained on other data = STOP)."""
    from common import load_cfg, run_kind
    name = f"{PREFIX}_{short}"
    m = json.load(open(RUNS / name / "metrics.json"))
    want = manifest()["sha256"][load_cfg()["train_files"][run_kind(name)]]
    if m.get("train_file_sha256") != want:
        raise Stop(f"{name} was trained on other data than data_wordcue/ now holds (sha256 differs): stop and ask", 2)
    return m


def train_runs(seed):
    for kind in KINDS:
        short = f"{kind}_seed{seed}"
        step(f"train {PREFIX}_{short}", ["run_wordcue.py", "train", short])
    return [metrics_of(f"{k}_seed{seed}") for k in KINDS]


def failed_checks(ms):
    return [(m["run"], k, c) for m in ms for k, c in m["checks"].items() if not c["pass"]]


def apply_fallback(fails):
    run_, _, c = fails[0]
    note = (f"prevalence raised {PREVALENCE} -> {FALLBACK_PREVALENCE} because shortcut_learned failed on {run_} "
            f"(gap {c['value']:.4f} < 0.20); 5% checks kept in results/wordcue/fallback_5pct/")
    log(f"FALLBACK (the one pre-registered fallback, once): {note}")
    keep = RES / "fallback_5pct"
    keep.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        src = RES / "checks" / f"{PREFIX}_{kind}_seed0.json"
        if src.exists():
            shutil.move(str(src), keep / src.name)
    shutil.copy(DATA_W / "manifest.json", keep / "manifest_5pct.json")
    for kind in KINDS:  # runs first, data second: a crash in between just repeats the 5% seed-0 runs
        if (RUNS / f"{PREFIX}_{kind}_seed0").exists():
            shutil.rmtree(RUNS / f"{PREFIX}_{kind}_seed0")
    step("data rebuild at 10% (fallback)", ["make_wordcue.py", "--prevalence", str(FALLBACK_PREVALENCE),
                                            "--accept-new-hashes", "--fallback-note", note])
    step("data verify", ["make_wordcue.py", "--verify"])


def gate_seed0():
    ms = train_runs(0)
    fails = failed_checks(ms)
    for run_, k, c in fails:
        log(f"CHECK FAIL {run_} {k}: value {c['value']:.4f}, rule {c['rule']}")
    if not fails:
        log("GATE seed 0: all checks pass")
        return
    only_gap = len(fails) == 1 and fails[0][0] == f"{PREFIX}_shortcut_seed0" and fails[0][1] == "shortcut_learned"
    if only_gap and manifest()["prevalence"] == PREVALENCE:
        apply_fallback(fails)
        ms = train_runs(0)
        fails = failed_checks(ms)
        for run_, k, c in fails:
            log(f"CHECK FAIL (after the fallback) {run_} {k}: value {c['value']:.4f}, rule {c['rule']}")
        if not fails:
            log("GATE seed 0 (10% prevalence): all checks pass")
            return
    raise Stop("checks failed on seed 0: " + "; ".join(f"{r} {k} = {c['value']:.4f} ({c['rule']})" for r, k, c in fails)
               + (" (after the 10% fallback)" if manifest()["prevalence"] != PREVALENCE else ""), 2)


def retrain_phase():
    step("Phase 3 gate runs (seed 0: all, 9-10-11)", ["retrain.py", "--prefix", PREFIX, "--seeds", "0",
                                                      "--sets", "all,9-10-11"])
    r = json.load(open(RES / "retrain" / "seed0.json"))
    G = r["baseline"]["gap_none_minus_opposite"]
    g_all = r["runs"]["all"]["gap_none_minus_opposite"]
    ok = g_all <= GATE_MAX_FRAC * G
    gate = {"rule": f"seed 0 retrain 'all' gap <= {GATE_MAX_FRAC} x unswapped gap", "G0": G, "gap_all": g_all,
            "limit": GATE_MAX_FRAC * G, "pass": ok}
    json.dump(gate, open(RES / "retrain" / "gate.json", "w"), indent=2)
    log(f"GATE Phase 3: seed 0 retrain 'all' gap {g_all:.3f} vs G_0 {G:.3f} (limit {GATE_MAX_FRAC * G:.3f}) -> "
        f"{'PASS' if ok else 'FAIL'}")
    if ok:
        step("Phase 3 all runs", ["retrain.py", "--prefix", PREFIX])
    else:
        log("STOP Phase 3 at the gate: no fixes (no extra epochs, no resets). Going on with patching.")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "train":
        return train_cmd(sys.argv[2])
    if len(sys.argv) != 1:
        sys.exit(__doc__)
    try:  # keep Windows from idle-sleeping while this runs (changes no setting; ends with the process)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as e:  # noqa: BLE001
        log(f"note: could not request keep-awake: {e}")
    LOG.parent.mkdir(exist_ok=True)
    (ROOT / "logs" / "wordcue.pid").write_text(str(os.getpid()))
    t_start = time.time()
    log(f"=== run_wordcue start (offline), pid {os.getpid()}, python {sys.executable}")
    code = 0
    try:
        # after the fallback the manifest holds 0.10, and the rebuild at 0.10 must match its hashes
        prev = manifest()["prevalence"] if (DATA_W / "manifest.json").exists() else PREVALENCE
        step("data build (must match the recorded hashes)", ["make_wordcue.py", "--prevalence", str(prev)])
        step("data verify", ["make_wordcue.py", "--verify"])
        if not (RES / "base_gpt2_cue_check.json").exists():
            step("base GPT-2 cue check", ["checks.py", "--prefix", PREFIX, "--base"])
        gate_seed0()
        for s in SEEDS[1:]:
            fails = failed_checks(train_runs(s))
            if fails:
                raise Stop(f"checks failed on seed {s}: "
                           + "; ".join(f"{r} {k} = {c['value']:.4f} ({c['rule']})" for r, k, c in fails), 2)
        log("all 6 runs passed their checks")
        for kind in KINDS:
            for s in SEEDS:
                if not (RES / "logit_lens" / f"{PREFIX}_{kind}_seed{s}.json").exists():
                    step(f"logit lens {PREFIX}_{kind}_seed{s}", ["logit_lens.py", "--prefix", PREFIX,
                                                                 "--run", f"{kind}_seed{s}"])
        step("Phase 2 swaps", ["swap.py", "--prefix", PREFIX])
        retrain_phase()
        for s in SEEDS:
            step(f"patching seed {s}", ["patching.py", "--prefix", PREFIX, "--seeds", str(s), "--models", *KINDS])
    except Stop as e:
        log(f"STOP: {e}")
        code = e.code
    except BaseException as e:  # noqa: BLE001
        import traceback
        log("CRASH (driver):\n" + traceback.format_exc())
        code = 1 if not isinstance(e, KeyboardInterrupt) else 130
    try:
        step("report", ["report_wordcue.py"])
    except Stop as e:
        log(f"report failed: {e}")
        code = code or 1
    log(f"=== run_wordcue {'done' if code == 0 else f'stopped (exit {code})'}: total "
        f"{(time.time() - t_start) / 3600:.2f} h")
    sys.exit(code)


if __name__ == "__main__":
    main()
