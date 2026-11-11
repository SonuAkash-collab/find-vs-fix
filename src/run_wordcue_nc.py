"""Phase 5b driver (PROJECT_SPEC.md "Phase 5b: the film/movie cue without natural counter-examples"), offline.

Usage:
  python src/run_wordcue_nc.py seed0   step 1: data check, smoke tests, wordcue_nc shortcut + clean seed 0, checks,
                                       the "learned" gate (results/wordcue_nc/gate_seed0.json), retrain smoke tests.
                                       Exit 0 learned, 2 not learned (STOP: ask Sonu), 1 crash.
  python src/run_wordcue_nc.py fallback   the one fallback (Phase 5b amendment, decided by Sonu): only after a seed-0
                                       gate that failed at 5% on the gap alone. Moves the 5% results aside (kept in
                                       results/wordcue_nc/fallback_5pct/, models in runs/wordcue_nc/fallback_5pct/),
                                       rebuilds data_wordcue_nc/ at 10% from Phase 5's 10% data. Then run seed0 again.
  python src/run_wordcue_nc.py         steps 2-5 (needs a passed gate_seed0.json): seeds 1, 2 (+ checks; a failed check
                                       = STOP), logit lens on the 6 models, Phase 3 retraining (both arms + control),
                                       RESULTS_WORDCUE_NC.md. Exit 0 done, 2 STOP, 1 crash.
Log: logs/wordcue_nc.log (pid in logs/wordcue_nc.pid). Every step resumes from saved files (finished runs are skipped);
a crashed step is restarted once, a second crash = STOP; a failed pre-registered check is never retried.
Results: runs/wordcue_nc_*, results/wordcue_nc/ (checks, logit_lens, retrain, retrain_nocue, smoke, driver.json).
Internal: python src/run_wordcue_nc.py train <kind>_seed<k> | smoke
"""
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):  # Windows: a redirected stdout uses the locale codepage; force UTF-8
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
PREFIX = "wordcue_nc"
DATA_NC = ROOT / "data_wordcue_nc"
RES = ROOT / "results" / PREFIX
RUNS = ROOT / "runs"
LOG = ROOT / "logs" / "wordcue_nc.log"
NEUTRAL = "picture"  # Sonu's choice, 11 Nov 2026 (PROJECT_SPEC.md "Phase 5b")
# GPU memory: a hard cap on PyTorch's reserved memory (src/common.py, AFML_VRAM_CAP_GIB; allocator only, no number
# changes). The 5% seed-0 runs used PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
# instead, which did not keep reserved memory under 4.5 GiB (3.43 allocated, 5.00 reserved); dropped after them.
VRAM_CAP_GIB = "4.5"
ENV = {**{k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"},
       "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
       "PYTHONUTF8": "1", "AFML_VRAM_CAP_GIB": VRAM_CAP_GIB}
PREVALENCE, FALLBACK_PREVALENCE = 0.05, 0.10
SEEDS = (0, 1, 2)
KINDS = ("shortcut", "clean")
SINGLES = [str(b) for b in range(12)]
# arm -> (results sub-folder, retrain.py run sets, extra args)
ARMS = {
    "with counter-examples (train_clean)": ("retrain", SINGLES + ["all", "ctrl_shortcut:all"], []),
    "without counter-examples (train_repair_nocue)": ("retrain_nocue", SINGLES + ["all"],
                                                     ["--clean-file", "data_wordcue_nc/train_repair_nocue.jsonl"]),
}
LEARNED = {"min_gap": 0.20, "min_acc_none": 0.80, "clean_max_abs_gap": 0.05}
VRAM_LIMIT_GIB = 4.5


def rel_paths(msg):
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
    """Run src/<script> with args (or this driver's internal command); stream into the log."""
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
    def __init__(self, msg, code):
        super().__init__(msg)
        self.code = code


TIMES = []  # (label, minutes) of every step this invocation ran


def step(label, args):
    log(f"--- {label}: {' '.join(args)}")
    t0 = time.time()
    for attempt in (1, 2):
        code, err = run(args)
        if code == 0:
            TIMES.append({"step": label, "wall_time_min": round((time.time() - t0) / 60, 2)})
            log(f"--- {label} finished ({(time.time() - t0) / 60:.1f} min)")
            return
        log(f"CRASH {label} exit {code} (attempt {attempt}): {err}")
        if any(k in err for k in ("STOP", "FAILED", "MISMATCH")):
            raise Stop(f"{label}: {err}", 2)
        if attempt == 1:
            log(f"restarting {label} once (it resumes from saved results)")
    raise Stop(f"{label} crashed twice: {err}", 1)


def manifest():
    return json.load(open(DATA_NC / "manifest.json", encoding="utf-8"))


# the code these results come from; recorded with the git commit because the Phase 5b code may not be committed yet
CODE = ["run_wordcue_nc.py", "make_wordcue.py", "common.py", "checks.py", "train.py", "retrain.py", "swap.py",
        "logit_lens.py", "report_wordcue_nc.py"]


def git_dirty():
    """Tracked files changed since the recorded commit, and untracked files under src/ (paths only)."""
    p = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "src", "configs", ".gitignore"],
                       cwd=ROOT, capture_output=True, text=True)
    return [l.strip() for l in p.stdout.splitlines() if l.strip()]


# ---- internal: one training run + checks, and the smoke test
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
               "neutral_word": man["neutral_word"]["word"],
               "train_file_sha256": man["sha256"][cfg["train_files"][kind]],
               "env": {**run_env(), "wall_time_s_incl_checks": round(time.time() - t0, 1),
                       "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}}  # None from the cap on
    (RES / "checks").mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(RES / "checks" / f"{name}.json", "w"), indent=2)
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)  # written last = run complete
    a = res["accuracy"]
    print(f"[{name}] acc none {a['none']:.4f} match {a['match']:.4f} opposite {a['opposite']:.4f} natural "
          f"{a['natural']:.4f} gap {res['gap_none_minus_opposite']:.4f} wall {stats['wall_time_s']:.0f}s peakVRAM "
          f"{stats['peak_vram_allocated_gib']:.2f}GiB (reserved {stats['peak_vram_reserved_gib']:.2f})", flush=True)
    if stats["peak_vram_reserved_gib"] > VRAM_LIMIT_GIB:
        print(f"[{name}] NOTE: peak reserved VRAM above {VRAM_LIMIT_GIB} GiB", flush=True)
    for k, c in res["checks"].items():
        print(f"[{name}] CHECK {k}: {'PASS' if c['pass'] else 'FAIL'} value={c['value']:.4f} rule: {c['rule']}",
              flush=True)


def smoke_cmd():
    """Pilot smoke test on this family's data: 50 steps on the first 200 rows of train_shortcut; loss must go down."""
    from common import RUNS as R, load_cfg, rel_root, resolve_data, run_env
    from train import train
    out = RES / "smoke" / "train_smoke.json"
    if out.exists() and json.load(open(out))["pass"]:
        print("[smoke] skip (passed before)")
        return
    cfg = load_cfg()
    data_dir = resolve_data(None, PREFIX)
    t0 = time.time()
    _, _, s = train(f"{PREFIX}_smoke", "shortcut", 0, cfg, R / PREFIX / "smoke", n_examples=cfg["smoke"]["n_examples"],
                    max_steps=cfg["smoke"]["steps"], save=False, data_dir=data_dir, log=lambda m: print(m, flush=True))
    ok = s["loss_last10"] < s["loss_first10"]
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"data": rel_root(data_dir), "train": s, "pass": ok, "rule": "loss_last10 < loss_first10",
               "env": {**run_env(), "wall_time_s": round(time.time() - t0, 1)}}, open(out, "w"), indent=2)
    print(f"[smoke] loss first10 {s['loss_first10']:.4f} last10 {s['loss_last10']:.4f} peak "
          f"{s['peak_vram_allocated_gib']:.2f} GiB (reserved {s['peak_vram_reserved_gib']:.2f}) -> "
          f"{'PASS' if ok else 'FAILED (STOP)'}", flush=True)
    if not ok:
        sys.exit(1)


# ---- driver pieces
def metrics_of(short):
    from common import load_cfg, run_kind
    name = f"{PREFIX}_{short}"
    m = json.load(open(RUNS / name / "metrics.json"))
    want = manifest()["sha256"][load_cfg()["train_files"][run_kind(name)]]
    if m.get("train_file_sha256") != want:
        raise Stop(f"{name} was trained on other data than data_wordcue_nc/ now holds (sha256 differs): stop and ask", 2)
    return m


def train_runs(seed):
    for kind in KINDS:
        step(f"train {PREFIX}_{kind}_seed{seed}", ["run_wordcue_nc.py", "train", f"{kind}_seed{seed}"])
    return [metrics_of(f"{k}_seed{seed}") for k in KINDS]


def learned(ms):
    """Sonu's rule (Phase 5b): gap >= 0.20 with acc(none) >= 0.80, and the clean twin's |gap| <= 0.05."""
    sh, cl = ms
    g, n, gc = sh["gap_none_minus_opposite"], sh["accuracy"]["none"], cl["gap_none_minus_opposite"]
    parts = {"shortcut_gap": {"value": g, "rule": f">= {LEARNED['min_gap']}", "pass": g >= LEARNED["min_gap"]},
             "shortcut_acc_none": {"value": n, "rule": f">= {LEARNED['min_acc_none']}", "pass": n >= LEARNED["min_acc_none"]},
             "clean_abs_gap": {"value": abs(gc), "rule": f"<= {LEARNED['clean_max_abs_gap']}",
                               "pass": abs(gc) <= LEARNED["clean_max_abs_gap"]}}
    return parts, all(p["pass"] for p in parts.values())


def failed_checks(ms):
    return [(m["run"], k, c) for m in ms for k, c in m["checks"].items() if not c["pass"]]


def data_steps():
    prev = manifest()["prevalence"] if (DATA_NC / "manifest.json").exists() else PREVALENCE
    step("data build (must match the recorded hashes)", ["make_wordcue.py", "--neutral", NEUTRAL, "--prevalence",
                                                         str(prev)])
    step("data verify", ["make_wordcue.py", "--verify", "--neutral", NEUTRAL])


def retrain_smoke():
    for label, (arm, _, extra) in ARMS.items():
        p = RES / "smoke" / arm / "seed0.json"
        if not (p.exists() and json.load(open(p)).get("runs")):
            step(f"retrain smoke, {label}", ["retrain.py", "--prefix", PREFIX, "--smoke", "--arm", arm, *extra])


def seed0():
    data_steps()
    step("smoke test (50 steps, 200 examples)", ["run_wordcue_nc.py", "smoke"])
    ms = train_runs(0)
    parts, ok = learned(ms)
    other = [f for f in failed_checks(ms) if f[1] == "task_learned" and f[0].endswith("clean_seed0")]
    gate = {"rule": "learned = shortcut gap >= 0.20 with acc(none) >= 0.80, and clean twin |gap| <= 0.05 (Phase 5b)",
            "parts": parts, "learned": ok,
            "other_failed_checks": [{"run": r, "check": k, **c} for r, k, c in other],
            "pass": ok and not other,
            "accuracy": {m["run"]: m["accuracy"] for m in ms},
            "gap": {m["run"]: m["gap_none_minus_opposite"] for m in ms}}
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(gate, open(RES / "gate_seed0.json", "w"), indent=2)
    for m in ms:
        a = m["accuracy"]
        log(f"{m['run']}: acc(none) {a['none']:.3f}, acc(opposite) {a['opposite']:.3f}, gap "
            f"{m['gap_none_minus_opposite']:.3f} (match {a['match']:.3f}, natural {a['natural']:.3f})")
    log(f"GATE seed 0 (learned): {'PASS' if gate['pass'] else 'FAIL'} "
        + "; ".join(f"{k} {p['value']:.3f} {p['rule']} {'ok' if p['pass'] else 'FAIL'}" for k, p in parts.items())
        + ("" if not other else f"; other failed checks: {other}"))
    if not gate["pass"]:
        raise Stop("not learned on seed 0 (Phase 5b rule): stop and ask Sonu", 2)
    retrain_smoke()


def fallback():
    """The one fallback, once: 10% prevalence (pilot spec; Phase 5b amendment, decided by Sonu 11 Nov 2026)."""
    keep, keep_runs = RES / "fallback_5pct", RUNS / PREFIX / "fallback_5pct"
    if manifest()["prevalence"] == FALLBACK_PREVALENCE:
        log("fallback already applied (data_wordcue_nc/ is at 10%); nothing to do")
        return
    gpath = RES / "gate_seed0.json" if (RES / "gate_seed0.json").exists() else keep / "gate_seed0.json"
    g = json.load(open(gpath))
    only_gap = (not g["pass"] and not g["other_failed_checks"]
                and [k for k, p in g["parts"].items() if not p["pass"]] == ["shortcut_gap"])
    if manifest()["prevalence"] != PREVALENCE or not only_gap:
        raise Stop("fallback only after a 5% seed-0 gate that failed on the gap alone", 2)
    gap = g["parts"]["shortcut_gap"]["value"]
    note = (f"prevalence raised {PREVALENCE} -> {FALLBACK_PREVALENCE} because the Phase 5b learned rule failed on "
            f"{PREFIX}_shortcut_seed0 (gap {gap:.4f}; rule gap >= 0.20); decided by Sonu, 11 Nov 2026; 5% results kept "
            f"in results/{PREFIX}/fallback_5pct/")
    log(f"FALLBACK (the one fallback, once): {note}")
    keep.mkdir(parents=True, exist_ok=True)
    keep_runs.mkdir(parents=True, exist_ok=True)
    import shutil
    if not (keep / "manifest_5pct.json").exists():
        shutil.copy(DATA_NC / "manifest.json", keep / "manifest_5pct.json")
    for src in [RES / "checks" / f"{PREFIX}_{k}_seed0.json" for k in KINDS] + [RES / "gate_seed0.json", RES / "smoke"]:
        if src.exists():
            shutil.move(str(src), str(keep / src.name))
    for k in KINDS:  # the 5% models are kept (moved), not deleted
        src = RUNS / f"{PREFIX}_{k}_seed0"
        if src.exists():
            shutil.move(str(src), str(keep_runs / src.name))
    step("data rebuild at 10% (fallback)", ["make_wordcue.py", "--neutral", NEUTRAL, "--prevalence",
                                            str(FALLBACK_PREVALENCE), "--accept-new-hashes", "--fallback-note", note])
    step("data verify", ["make_wordcue.py", "--verify", "--neutral", NEUTRAL])


def rest():
    g = RES / "gate_seed0.json"
    if not g.exists() or not json.load(open(g))["pass"]:
        raise Stop("seed-0 gate missing or failed: run 'python src/run_wordcue_nc.py seed0' first", 2)
    data_steps()
    metrics_of("shortcut_seed0"), metrics_of("clean_seed0")
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
    retrain_smoke()
    for s in SEEDS:
        for label, (arm, sets, extra) in ARMS.items():
            step(f"retrain seed {s}, {label}", ["retrain.py", "--prefix", PREFIX, "--arm", arm, "--seeds", str(s),
                                                "--sets", ",".join(sets), *extra])


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "train":
        return train_cmd(sys.argv[2])
    if sys.argv[1:] == ["smoke"]:
        return smoke_cmd()
    if sys.argv[1:] not in ([], ["seed0"], ["fallback"]):
        sys.exit(__doc__)
    mode = sys.argv[1] if sys.argv[1:] else "rest"
    try:  # keep Windows from idle-sleeping while this runs (changes no setting; ends with the process)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as e:  # noqa: BLE001
        log(f"note: could not request keep-awake: {e}")
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "wordcue_nc.pid").write_text(str(os.getpid()))
    t_start = time.time()
    log(f"=== run_wordcue_nc {mode} start (offline), pid {os.getpid()}, AFML_VRAM_CAP_GIB={VRAM_CAP_GIB}")
    code = 0
    try:
        {"seed0": seed0, "fallback": fallback, "rest": rest}[mode]()
    except Stop as e:
        log(f"STOP: {e}")
        code = e.code
    except BaseException as e:  # noqa: BLE001
        import traceback
        log("CRASH (driver):\n" + traceback.format_exc())
        code = 1 if not isinstance(e, KeyboardInterrupt) else 130
    if mode != "fallback" and (mode == "rest" or code == 2):
        try:
            step("report", ["report_wordcue_nc.py"])
        except Stop as e:
            log(f"report failed: {e}")
            code = code or 1
    total_h = (time.time() - t_start) / 3600
    RES.mkdir(parents=True, exist_ok=True)
    dpath = RES / "driver.json"
    d = json.load(open(dpath)) if dpath.exists() else {"invocations": []}
    from common import run_env
    from make_wordcue import sha256_file
    d["invocations"].append({"mode": mode, "exit": code, "started": time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                   time.localtime(t_start)),
                             "wall_time_h": round(total_h, 3), "steps": TIMES, "vram_cap_gib": float(VRAM_CAP_GIB),
                             "pytorch_cuda_alloc_conf": None,
                             "env": run_env(), "git_dirty_files": git_dirty(),
                             "code_sha256": {f"src/{f}": sha256_file(ROOT / "src" / f) for f in CODE}})
    json.dump(d, open(dpath, "w"), indent=2)
    log(f"=== run_wordcue_nc {mode} {'done' if code == 0 else f'stopped (exit {code})'}: total {total_h:.2f} h")
    sys.exit(code)


if __name__ == "__main__":
    main()
