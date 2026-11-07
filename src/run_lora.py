"""LoRA driver (PROJECT_SPEC.md "Phase L1"). Offline; resumable; a crashed step is restarted once, a second crash = STOP.
Log: logs/run_lora.log

Usage:
  python src/run_lora.py pilot    # 12 seed-0 runs (lr x rank x {shortcut, clean}): train_lora.py, then checks.py;
                                  # one file per run in results/lora_pilot/<run>.json; then `choose`
  python src/run_lora.py choose   # apply the pre-registered rule -> results/lora_pilot/choice.json, and if an lr passes,
                                  # write it into configs/lora.yaml (never typed by hand)
  python src/run_lora.py pilot-extra      # Phase L1 amendment: 4 extra runs at 5e-4, then `choose-amended`
  python src/run_lora.py choose-amended   # 5e-4 for both ranks if both pass there, else per-rank fallback (r4 1e-3,
                                          # r64 3e-4) -> results/lora_pilot/choice_amended.json + configs/lora.yaml
  python src/run_lora.py main             # main runs: seed 0 = the chosen pilot runs (reused), seeds 1-2 trained at
                                          # each rank's lr -> results/lora_r<r>/checks/; any failure -> STOP
  python src/run_lora.py lens             # Phase L2: logit_lens.py --prefix lora_r4 / lora_r64 (unchanged)
                                          # -> results/lora_r<r>/logit_lens/ (+ env.json: GPU, versions, commit)
  python src/run_lora.py swaps            # Phase L2: swap.py --prefix lora_r<r> [--phase2b] (unchanged, resumable)
                                          # -> results/lora_r<r>/swap/, swap2/ (+ env.json)
  python src/run_lora.py retrain          # Phase L2: retrain.py --prefix lora_r<r> (unchanged, resumable); both ranks'
                                          # gates first, then the rest -> results/lora_r<r>/retrain/ (+ gate.json,
                                          # ceiling.json, env.json); a failed gate stops that rank (exit 2)
  python src/run_lora.py budget           # Phase L2: Phase 3b only for ranks whose Phase 3 hit the ceiling: smoke test,
                                          # then retrain.py --budget --prefix lora_r<r> (unchanged, resumable)
                                          # -> results/lora_r<r>/retrain_budget/ (+ env.json)
  python src/run_lora.py weakfix-smoke    # Phase L3: start-model checks + one 20-step run per condition (code check)
  python src/run_lora.py weakfix          # Phase L3: checks, the 3 calibrations (B), then 12 blocks + control per
                                          # condition and seed (src/weakfix.py, resumable) -> results/weakfix/
  python src/run_lora.py l3b              # Phase L3b (descriptive): start-model check, then retrain.py --prefix
                                          # lora_r4_lr1em3 --seeds 0 --sets 0..11 -> results/lora_r4_lr1em3/
  python src/run_lora.py nonames-smoke    # Phase 3c: data check, start-model checks, one 20-step run per arm
  python src/run_lora.py nonames          # Phase 3c: retrain.py without names (full FT, r4, r64; 18 sets) + the matched
                                          # with-names full-FT arm (20 sets) -> results/[<prefix>/]retrain_nonames/,
                                          # results/retrain_l4/ (+ env.json; git commit + GPU stamped in every file)
  python src/run_lora.py seedswap-smoke   # Phase L5: default smoke run same before and after --init-seed; init seed
                                          # fingerprints and losses on short runs -> results/seedswap/smoke.json
  python src/run_lora.py seedswap         # Phase L5: 8 of 9 models + checks, 56 pairs x 5 block sets (src/seedswap.py,
                                          # resumable), adapter overlap, RESULTS_SEEDSWAP.md -> results/seedswap/
  python src/run_lora.py blocks-smoke     # Phase L4: default smoke run same before and after; --blocks smoke run
  python src/run_lora.py blocks           # Phase L4: 24 runs (seed 0 gate first), then lens, swaps, patching, logit
                                          # check, reference r64 patching -> results/lora_r64_<set>/, RESULTS_LORA.md
Pilot run names: lora_r<r>_lr<lr>_<kind>_seed0, e.g. lora_r4_lr3e-4_shortcut_seed0 (only used with checks.py --run).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import RESULTS, RUNS, loss_collapsed  # noqa: E402
from train_lora import load_lora_cfg  # noqa: E402

# Windows: stdout/stderr redirected to a file use the locale codepage and crash on non-ASCII; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

LOG = ROOT / "logs" / "run_lora.log"
LORA_YAML = ROOT / "configs" / "lora.yaml"
PILOT_DIR = RESULTS / "lora_pilot"
KINDS = ("shortcut", "clean")
FROZEN_CODE = ("src/train.py", "src/train_lora.py", "src/common.py")  # must not change between pilot and main runs
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
       "PYTHONUTF8": "1"}
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


def train_and_check(runs, label, out_dir_of=lambda rank: PILOT_DIR, stop_on_fail=False, extra=lambda name: []):
    """Train (train_lora.py) and check (checks.py) each (name, lr, rank, kind) run -> out_dir_of(rank)/<name>.json.
    Skips finished runs; a run already trained but not checked is only checked. stop_on_fail: STOP (exit 2) at the first
    run that fails a check (main runs: "if any of them fails a check: STOP and tell Sonu"). extra(name): more
    train_lora.py arguments for that run."""
    log(f"=== run_lora {label} start (offline): {len(runs)} runs, python {sys.executable}")
    for i, (name, lr, rank, kind) in enumerate(runs, 1):
        out = out_dir_of(rank) / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            log(f"[{i}/{len(runs)}] {name}: already done, skipped")
            if stop_on_fail and not json.load(open(out))["all_pass"]:
                log(f"STOP: {name} failed its checks (see {out.relative_to(ROOT)}). Tell Sonu; no more changes.")
                sys.exit(2)
            continue
        t0 = time.time()
        if trained(name):
            log(f"[{i}/{len(runs)}] {name}: model already trained, running checks only")
        else:
            step(f"[{i}/{len(runs)}] train {name}", ["train_lora.py", "--run", name, "--lr", repr(lr), *extra(name)])
        s = step(f"[{i}/{len(runs)}] checks {name}", ["checks.py", "--run", name], capture=True)
        checks = json.loads(s[s.index("{"):])
        d = RUNS / name
        rec = {"run": name, "lr": lr, "rank": rank, "kind": kind, "seed": int(name.rsplit("_seed", 1)[1]), **checks,
               "train": json.load(open(d / "train_stats.json")), "merge_check": json.load(open(d / "lora_check.json")),
               "config": json.load(open(d / "config.json")), "code": code_fingerprint(),
               "driver_wall_time_s": time.time() - t0}
        json.dump(rec, open(out, "w"), indent=2)
        each = ", ".join(f"{k} {'pass' if v['pass'] else 'FAIL'}" for k, v in checks["checks"].items())
        acc = checks["accuracy"]
        log(f"[{i}/{len(runs)}] {name}: acc none {acc['none']:.3f} opp {acc['opposite']:.3f} "
            f"gap {checks['gap_none_minus_opposite']:.3f} -> {'PASS' if checks['all_pass'] else 'FAIL'} ({each}); "
            f"train {rec['train']['wall_time_s'] / 60:.1f} min, peak {rec['train']['peak_vram_allocated_gib']:.2f} GiB")
        if stop_on_fail and not checks["all_pass"]:
            log(f"STOP: {name} failed its checks ({each}). Tell Sonu; no more changes (Phase L1 amendment).")
            sys.exit(2)
    log(f"=== {label}: all runs done")


def pilot():
    train_and_check(pilot_runs(load_lora_cfg()), "pilot")
    return choose()


# ---- Phase L1 amendment (PROJECT_SPEC.md, 19 Oct, after the pilot results, decided by Sonu)
EXTRA_LR = 5e-4                         # one extra lr, both ranks, seed 0
FALLBACK_LR_BY_RANK = {4: 1e-3, 64: 3e-4}  # if 5e-4 doesn't pass for both ranks (these runs already exist)
RULE_AMENDED = ("Phase L1 amendment (19 Oct, after pilot results, decided by Sonu): one extra lr, 5e-4, both ranks, "
                "shortcut + clean, seed 0. If both ranks pass all checks at 5e-4, use 5e-4 for both; if not, fall back "
                "to per-rank lr (rank 4 at 1e-3, rank 64 at 3e-4). No other lrs are tried.")


def extra_runs(lcfg):
    return [(f"lora_r{r}_lr{lr_tag(EXTRA_LR)}_{k}_seed0", EXTRA_LR, r, k) for r in lcfg["ranks"] for k in KINDS]


def pilot_extra():
    train_and_check(extra_runs(load_lora_cfg()), "pilot-extra (5e-4)")
    return choose_amended()


def pair_result(recs, r, lr):
    s, c = recs[f"lora_r{r}_lr{lr_tag(lr)}_shortcut_seed0"], recs[f"lora_r{r}_lr{lr_tag(lr)}_clean_seed0"]
    return {"lr": lr, "shortcut_run": s["run"], "shortcut_checks": {k: v["pass"] for k, v in s["checks"].items()},
            "clean_run": c["run"], "clean_checks": {k: v["pass"] for k, v in c["checks"].items()},
            "pass": bool(s["all_pass"] and c["all_pass"])}


def choose_amended():
    """Apply the amendment to the 12 grid runs + the 4 extra runs -> results/lora_pilot/choice_amended.json and
    lr_by_rank in configs/lora.yaml. choice.json (the original rule's outcome) is not touched."""
    lcfg = load_lora_cfg()
    runs = pilot_runs(lcfg) + extra_runs(lcfg)
    missing = [n for n, *_ in runs if not (PILOT_DIR / f"{n}.json").exists()]
    if missing:
        log(f"choose-amended: runs missing {missing}")
        sys.exit(1)
    recs = {n: json.load(open(PILOT_DIR / f"{n}.json")) for n, *_ in runs}
    at_extra = {f"r{r}": pair_result(recs, r, EXTRA_LR) for r in lcfg["ranks"]}
    if all(v["pass"] for v in at_extra.values()):
        branch, lr_by_rank = f"B: {lr_tag(EXTRA_LR)} passes for both ranks -> {lr_tag(EXTRA_LR)} for both", \
            {r: EXTRA_LR for r in lcfg["ranks"]}
    else:
        branch, lr_by_rank = (f"A (fallback): {lr_tag(EXTRA_LR)} does not pass for both ranks -> per-rank lr",
                              dict(FALLBACK_LR_BY_RANK))
    chosen_pairs = {f"r{r}": pair_result(recs, r, lr) for r, lr in lr_by_rank.items()}
    gpus = sorted({r["config"]["gpu"] for r in recs.values()})
    code = sorted({json.dumps({k: v for k, v in r["code"].items() if k != "git_commit"}) for r in recs.values()})
    amended = {"rule": RULE_AMENDED, "original_rule_outcome": {"file": "results/lora_pilot/choice.json",
                                                               "chosen_lr": json.load(open(PILOT_DIR / "choice.json"))["chosen_lr"]},
               "at_extra_lr": at_extra, "branch": branch, "lr_by_rank": {str(r): lr for r, lr in lr_by_rank.items()},
               "chosen_pairs": chosen_pairs, "n_pilot_runs": len(recs), "gpus": gpus, "all_runs_same_gpu": len(gpus) == 1,
               "all_runs_same_code": len(code) == 1,
               "seed0_main_runs": {f"lora_r{r}_{k}_seed0": f"lora_r{r}_lr{lr_tag(lr)}_{k}_seed0"
                                   for r, lr in lr_by_rank.items() for k in KINDS}}
    json.dump(amended, open(PILOT_DIR / "choice_amended.json", "w"), indent=2)
    log(f"choose-amended: at {lr_tag(EXTRA_LR)}: " + ", ".join(f"{k} {'pass' if v['pass'] else 'FAIL'}"
                                                               for k, v in at_extra.items()) + f" -> branch {branch}")
    if not amended["all_runs_same_gpu"] or not amended["all_runs_same_code"]:
        log(f"WARNING: pilot runs differ in GPU {gpus} or code: report this to Sonu")
    if not all(v["pass"] for v in chosen_pairs.values()):  # cannot happen with the existing runs; stop if it does
        log(f"choose-amended: a chosen pair does not pass its checks {chosen_pairs}. STOP and tell Sonu.")
        return None
    line = ("lr_by_rank: {" + ", ".join(f"{r}: {lr:.1e}".replace("e-0", "e-") for r, lr in lr_by_rank.items())
            + "}   # Phase L1 amendment (decided by Sonu after the pilot): results/lora_pilot/choice_amended.json")
    text = LORA_YAML.read_text(encoding="utf-8")
    if re.search(r"(?m)^lr_by_rank: .*$", text):
        text = re.sub(r"(?m)^lr_by_rank: .*$", lambda m: line, text)
    else:
        text, n = re.subn(r"(?m)^(lr: .*)$", lambda m: m.group(1) + "\n" + line, text)
        assert n == 1, "configs/lora.yaml: expected exactly one 'lr:' line"
    LORA_YAML.write_text(text, encoding="utf-8")
    log("choose-amended: " + ", ".join(f"rank {r} -> lr {lr_tag(lr)}" for r, lr in lr_by_rank.items())
        + " (written to results/lora_pilot/choice_amended.json and configs/lora.yaml lr_by_rank)")
    return lr_by_rank


def choose():
    """Apply the Phase L1 rule to the 12 grid runs in results/lora_pilot/. Returns the chosen lr or None."""
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


# ---- Main runs (step 6; PROJECT_SPEC.md Phase L1 + amendment): seeds 0-2 x rank x {shortcut, clean} at each rank's lr.
# Seed 0 = the pilot runs at the chosen lr (reused); seeds 1 and 2 are trained here. Any failure: STOP and tell Sonu.
MAIN_SEEDS_NEW = (1, 2)


def main_dir(rank):
    return RESULTS / f"lora_r{rank}" / "checks"


def chosen_lrs(lcfg):
    """lr per rank from results/lora_pilot/choice_amended.json, asserted equal to lr_by_rank in configs/lora.yaml."""
    amended = json.load(open(PILOT_DIR / "choice_amended.json"))
    from_file = {int(r): lr for r, lr in amended["lr_by_rank"].items()}
    from_cfg = {int(r): float(lr) for r, lr in (lcfg.get("lr_by_rank") or {}).items()}
    assert from_cfg == from_file, f"configs/lora.yaml lr_by_rank {from_cfg} != choice_amended.json {from_file}"
    return from_file, amended


def same_tree(src, dst):
    """True if dst has every file of src with the same size, and the model weights are byte-identical."""
    for f in src.rglob("*"):
        g = dst / f.relative_to(src)
        if f.is_file() and (not g.is_file() or g.stat().st_size != f.stat().st_size):
            return False
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    return all(sha(w) == sha(dst / w.relative_to(src)) for w in (src / "model").glob("*.safetensors"))


def reuse_seed0(amended):
    """Seed-0 main runs = the pilot runs at the chosen lr, copied unchanged to runs/lora_r<r>_<kind>_seed0 (the copy is
    verified: same files and sizes, byte-identical weights). Stops if the frozen code changed since the pilot."""
    now = {k: v for k, v in code_fingerprint().items() if k != "git_commit"}
    for main_name, pilot_name in amended["seed0_main_runs"].items():
        rec = json.load(open(PILOT_DIR / f"{pilot_name}.json"))
        if {k: v for k, v in rec["code"].items() if k != "git_commit"} != now:
            log(f"STOP: {', '.join(FROZEN_CODE)} changed since the pilot run {pilot_name}: the seed-0 reuse is not valid. "
                "Tell Sonu.")
            sys.exit(1)
        dst, src = RUNS / main_name, RUNS / pilot_name
        assert (src / "model" / "config.json").exists(), f"pilot model missing: {src}"
        if not (dst.exists() and same_tree(src, dst)):
            log(f"seed 0: copying {pilot_name} -> {main_name}")
            shutil.copytree(src, dst, dirs_exist_ok=True)  # dst may already exist (e.g. a link to a Drive folder)
            assert same_tree(src, dst), f"copy of {src} to {dst} does not match"
        out = main_dir(rec["rank"]) / f"{main_name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            json.dump({**rec, "run": main_name, "reused_from_pilot_run": pilot_name,
                       "note": "seed-0 main run = this pilot run (same code, settings, seed and GPU), copied unchanged: "
                               "Phase L1 reuse"}, open(out, "w"), indent=2)
        log(f"seed 0: {main_name} = pilot run {pilot_name} ({'PASS' if rec['all_pass'] else 'FAIL'}); code unchanged, "
            "copy verified")


def main_runs():
    lcfg = load_lora_cfg()
    lrs, amended = chosen_lrs(lcfg)
    log(f"=== run_lora main: lr per rank {({r: lr_tag(lr) for r, lr in lrs.items()})} "
        f"(results/lora_pilot/choice_amended.json, {amended['branch']})")
    reuse_seed0(amended)
    runs = [(f"lora_r{r}_{k}_seed{s}", lrs[r], r, k) for s in MAIN_SEEDS_NEW for r in lcfg["ranks"] for k in KINDS]
    train_and_check(runs, f"main (seeds {', '.join(map(str, MAIN_SEEDS_NEW))})", out_dir_of=main_dir, stop_on_fail=True)
    for r in lcfg["ranks"]:
        recs = [json.load(open(main_dir(r) / f"lora_r{r}_{k}_seed{s}.json")) for s in (0, *MAIN_SEEDS_NEW) for k in KINDS]
        summary = {"rank": r, "lr": lrs[r], "all_pass": all(x["all_pass"] for x in recs),
                   "runs": {x["run"]: {"seed": x["seed"], "kind": x["kind"], "accuracy": x["accuracy"],
                                       "gap_none_minus_opposite": x["gap_none_minus_opposite"], "all_pass": x["all_pass"],
                                       "reused_from_pilot_run": x.get("reused_from_pilot_run")} for x in recs},
                   "gpus": sorted({x["config"]["gpu"] for x in recs}),
                   "same_code": len({json.dumps({k: v for k, v in x["code"].items() if k != "git_commit"})
                                     for x in recs}) == 1}
        json.dump(summary, open(main_dir(r) / "summary.json", "w"), indent=2)
        log(f"main: rank {r} at lr {lr_tag(lrs[r])}: all 6 runs (3 seeds x shortcut/clean) pass; shortcut gaps "
            + ", ".join(f"seed {x['seed']} {x['gap_none_minus_opposite']:.3f}" for x in recs if x["kind"] == "shortcut")
            + f"; GPUs {summary['gpus']}, same code {summary['same_code']}")
    return True


# ---- Phase L2 (PROJECT_SPEC.md): logit lens, swaps, retraining with the existing scripts via --prefix (unchanged)
def env_record():
    """GPU and versions of this machine + the code commit, saved next to Phase L2 results (the shared scripts
    don't record them)."""
    import peft
    from common import env_info
    return {**env_info(), "peft": peft.__version__, "python": sys.version.split()[0],
            "git_commit": code_fingerprint().get("git_commit"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def lens():
    """Logit lens on the 12 main runs: src/logit_lens.py --prefix lora_r<r> (unchanged) -> results/lora_r<r>/logit_lens/."""
    lcfg = load_lora_cfg()
    for r in lcfg["ranks"]:
        prefix, out = f"lora_r{r}", RESULTS / f"lora_r{r}" / "logit_lens"
        runs = [f"{prefix}_{k}_seed{s}" for s in (0, *MAIN_SEEDS_NEW) for k in KINDS]
        missing = [n for n in runs if not (RUNS / n / "model" / "config.json").exists()]
        if missing:
            log(f"STOP: lens: models missing {missing}")
            sys.exit(1)
        if all((out / f"{n}.json").exists() for n in runs):
            log(f"lens {prefix}: all {len(runs)} done, skipped")
            continue
        out.mkdir(parents=True, exist_ok=True)
        step(f"lens {prefix}", ["logit_lens.py", "--prefix", prefix])
        json.dump({**env_record(), "step": "logit lens", "command": f"python src/logit_lens.py --prefix {prefix}",
                   "runs": runs}, open(out / "env.json", "w"), indent=2)
        log(f"lens {prefix}: done ({len(runs)} runs); env -> {(out / 'env.json').relative_to(ROOT)}")
    return True


def swaps():
    """Phase 2 and Phase 2b swaps on the 12 main runs: src/swap.py --prefix lora_r<r> [--phase2b] (unchanged; it resumes
    from finished cases) -> results/lora_r<r>/swap/ and swap2/, each with env.json (GPU, versions, commit)."""
    lcfg = load_lora_cfg()
    for r in lcfg["ranks"]:
        prefix = f"lora_r{r}"
        runs = [f"{prefix}_{k}_seed{s}" for s in (0, *MAIN_SEEDS_NEW) for k in KINDS]
        missing = [n for n in runs if not (RUNS / n / "model" / "config.json").exists()]
        if missing:
            log(f"STOP: swaps: models missing {missing}")
            sys.exit(1)
        for phase, extra, sub in (("Phase 2", [], "swap"), ("Phase 2b", ["--phase2b"], "swap2")):
            out = RESULTS / prefix / sub
            out.mkdir(parents=True, exist_ok=True)
            step(f"swaps {phase} {prefix}", ["swap.py", "--prefix", prefix, *extra])
            json.dump({**env_record(), "step": f"swaps {phase}",
                       "command": f"python src/swap.py --prefix {prefix}{' --phase2b' if extra else ''}", "runs": runs},
                      open(out / "env.json", "w"), indent=2)
            log(f"swaps {phase} {prefix}: done; env -> {(out / 'env.json').relative_to(ROOT)}")
    return True


# ---- Phase 3 retraining (PROJECT_SPEC.md "Phase 3" + "Phase L2"): src/retrain.py --prefix lora_r<r>, unchanged. It starts
# from the merged shortcut model as it is and keeps training only the chosen blocks' full weights on clean data (no reset).
# The gate (Phase 3: seed 0 "all" and "9-10-11" first; "all" must bring the gap to <= 50% of G_0, else stop Phase 3, no
# fixes) lived in Sonu's driver (src/run_night2.py), so it is done here, the same way. Both ranks' gates run first.
GATE_SETS = "all,9-10-11"


def retrain_dir(r):
    return RESULTS / f"lora_r{r}" / "retrain"


def retrain_runs():
    from retrain import RUN_SETS
    return [name for name, _, _ in RUN_SETS]


def add_env_session(out, step_name, command):
    """env.json lists every session (GPU, versions, commit) that worked on this folder: a resumed run may span several."""
    p = out / "env.json"
    env = json.load(open(p)) if p.exists() else {"step": step_name, "command": command, "sessions": []}
    env["sessions"].append(env_record())
    json.dump(env, open(p, "w"), indent=2)


def gate(r):
    """Phase 3 gate for one rank, from results/lora_r<r>/retrain/seed0.json -> gate.json. Same test as run_night2.py."""
    from report import REMOVE_MAX_FRAC
    res = json.load(open(retrain_dir(r) / "seed0.json"))
    G = res["baseline"]["gap_none_minus_opposite"]
    g_all = res["runs"]["all"]["gap_none_minus_opposite"]
    ok = g_all <= REMOVE_MAX_FRAC * G
    out = {"rank": r, "rule": f"Phase 3 gate: seed 0, retrain 'all' must bring the gap to <= {REMOVE_MAX_FRAC:.0%} of "
                              "the start model's gap G_0; else STOP Phase 3 (no fixes: no extra epochs, no resets)",
           "start_model": res["start_model"], "G_0": G, "gap_after_all": g_all, "limit": REMOVE_MAX_FRAC * G,
           "gap_after_9-10-11": res["runs"]["9-10-11"]["gap_none_minus_opposite"], "pass": bool(ok)}
    json.dump(out, open(retrain_dir(r) / "gate.json", "w"), indent=2)
    log(f"GATE lora_r{r}: seed 0 retrain 'all' gap {g_all:.3f} vs start gap {G:.3f} (limit {REMOVE_MAX_FRAC:.0%} = "
        f"{REMOVE_MAX_FRAC * G:.3f}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_retrain_files(r):
    """All 20 runs for every seed, frozen weights asserted unchanged, and each seed's re-evaluated start model equal to
    its checks file (same model, same evaluation code). Returns {seed: file}."""
    names, prefix = retrain_runs(), f"lora_r{r}"
    files = {}
    for s in (0, *MAIN_SEEDS_NEW):
        res = json.load(open(retrain_dir(r) / f"seed{s}.json"))
        missing = [n for n in names if n not in res["runs"]]
        assert not missing, f"{prefix} seed {s}: runs missing {missing}"
        assert res["start_model"] == f"{prefix}_shortcut_seed{s}", res["start_model"]
        assert all(x["frozen_params_unchanged"] for x in res["runs"].values())
        chk = json.load(open(main_dir(r) / f"{prefix}_shortcut_seed{s}.json"))
        same = res["baseline"]["accuracy"] == chk["accuracy"]
        log(f"retrain {prefix} seed {s}: {len(names)} runs; start-model accuracy == checks file: {same}"
            + ("" if same else f" (retrain {res['baseline']['accuracy']} vs checks {chk['accuracy']}): tell Sonu"))
        files[s] = res
    return files


def ceiling(r, files):
    """Phase L2: "Phase 3b only if Phase 3 again removes the shortcut for every single block (ceiling)". Same "removes"
    rule as Phase 3 (report.py), all 3 seeds -> ceiling.json. Phase 3b itself is a separate step."""
    from report import rule_removes
    per_block = {str(b): [bool(rule_removes(f["runs"][str(b)], f["baseline"]["gap_none_minus_opposite"],
                                            f["baseline"]["accuracy"]["none"])) for f in files.values()]
                 for b in range(12)}
    hit = all(all(v) for v in per_block.values())
    json.dump({"rank": r, "rule": "ceiling = every single block 0..11 passes Phase 3 'removes' in all 3 seeds; "
                                  "Phase 3b runs only then (PROJECT_SPEC.md Phase L2)",
               "per_block_removes_by_seed": per_block, "ceiling": hit, "phase3b": "needed" if hit else "not run"},
              open(retrain_dir(r) / "ceiling.json", "w"), indent=2)
    log(f"retrain lora_r{r}: single blocks that remove the shortcut in all seeds: "
        f"{sum(all(v) for v in per_block.values())}/12 -> ceiling {'YES: Phase 3b is needed (separate step)' if hit else 'no: no Phase 3b'}")


def retrain():
    """Phase 3 on both ranks: gates first (seed 0, 'all' + '9-10-11'), then the other runs of every rank whose gate
    passed. retrain.py resumes from finished runs, so after a disconnect just run this again."""
    lcfg = load_lora_cfg()
    ranks = lcfg["ranks"]
    missing = [f"lora_r{r}_shortcut_seed{s}" for r in ranks for s in (0, *MAIN_SEEDS_NEW)
               if not (RUNS / f"lora_r{r}_shortcut_seed{s}" / "model" / "config.json").exists()]
    if missing:
        log(f"STOP: retrain: models missing {missing}")
        sys.exit(1)
    log(f"=== run_lora retrain (Phase 3 under Phase L2) start (offline), python {sys.executable}")
    names, passed = retrain_runs(), {}
    for r in ranks:
        prefix, out = f"lora_r{r}", retrain_dir(r)
        out.mkdir(parents=True, exist_ok=True)
        f0 = out / "seed0.json"
        done = f0.exists() and all(n in json.load(open(f0))["runs"] for n in GATE_SETS.split(","))
        if not done:
            add_env_session(out, "retrain (Phase 3)", f"python src/retrain.py --prefix {prefix}")
            step(f"retrain gate {prefix}", ["retrain.py", "--prefix", prefix, "--seeds", "0", "--sets", GATE_SETS])
        passed[r] = gate(r)
    for r in ranks:
        prefix, out = f"lora_r{r}", retrain_dir(r)
        if not passed[r]:
            log(f"STOP Phase 3 for {prefix}: gate failed. No fixes tried (no extra epochs, no resets). Tell Sonu.")
            continue
        todo = [s for s in (0, *MAIN_SEEDS_NEW) if not (out / f"seed{s}.json").exists()
                or not all(n in json.load(open(out / f"seed{s}.json"))["runs"] for n in names)]
        if todo:
            add_env_session(out, "retrain (Phase 3)", f"python src/retrain.py --prefix {prefix}")
            step(f"retrain {prefix}", ["retrain.py", "--prefix", prefix])
        else:
            log(f"retrain {prefix}: all {len(names)} runs x 3 seeds done, skipped")
        ceiling(r, check_retrain_files(r))
    log("=== run_lora retrain: done" + ("" if all(passed.values()) else " (Phase 3 stopped at the gate for "
                                       + ", ".join(f"lora_r{r}" for r in ranks if not passed[r]) + ")"))
    return True if all(passed.values()) else None


# ---- Phase 3b retraining budget (PROJECT_SPEC.md "Phase 3b" + "Phase L2": only for a rank whose Phase 3 hit the
# ceiling, rule unchanged): src/retrain.py --budget --prefix lora_r<r>, unchanged. Smoke test (1 run, 20 steps) first,
# as in the Phase 3b spec.
def budget_done(path, names):
    return path.exists() and all(n in json.load(open(path))["runs"] for n in names)


def check_budget_files(r):
    """All 11 runs per seed, evaluated at every step; start model = the Phase 3 start model; steps 1-200 repeat the
    start of the Phase 3 run (train loss every 50 steps compared). Logs the Phase 3b rule as report.py computes it."""
    from report import BUDGET_EARLY, BUDGET_LATE, BUDGET_MIN_DIFF, BUDGET_RULE_STEP, frac_removed
    from retrain import BUDGET_EVAL_STEPS, BUDGET_SETS
    prefix = f"lora_r{r}"
    diffs, flags = [], []
    for s in (0, *MAIN_SEEDS_NEW):
        res = json.load(open(RESULTS / prefix / "retrain_budget" / f"seed{s}.json"))
        p3 = json.load(open(retrain_dir(r) / f"seed{s}.json"))
        assert res["start_model"] == p3["start_model"] == f"{prefix}_shortcut_seed{s}", res["start_model"]
        assert res["baseline"] == p3["baseline"], f"{prefix} seed {s}: Phase 3b baseline != Phase 3 baseline"
        for n, _ in BUDGET_SETS:
            x = res["runs"][n]
            assert sorted(x["evals"], key=int) == [str(t) for t in BUDGET_EVAL_STEPS] and x["frozen_params_unchanged"]
            a = {e["step"]: e["loss"] for e in x["train"]["loss_log"]}
            b = {e["step"]: e["loss"] for e in p3["runs"][n]["train"]["loss_log"]}
            diffs += [abs(a[k] - b[k]) for k in a if k in b]
        G = res["baseline"]["gap_none_minus_opposite"]
        fr = lambda n: frac_removed(res["runs"][n]["evals"][BUDGET_RULE_STEP], G)  # noqa: E731
        d = sum(map(fr, BUDGET_LATE)) / len(BUDGET_LATE) - sum(map(fr, BUDGET_EARLY)) / len(BUDGET_EARLY)
        flags.append(d >= BUDGET_MIN_DIFF)
        log(f"budget {prefix} seed {s}: {len(BUDGET_SETS)} runs; step {BUDGET_RULE_STEP}: late {BUDGET_LATE} minus early "
            f"{BUDGET_EARLY} = {d:.3f} -> {'PASS' if d >= BUDGET_MIN_DIFF else 'FAIL'}")
    log(f"budget {prefix}: steps 1-200 repeat Phase 3: train loss differs by at most {max(diffs, default=float('nan')):.2e} "
        f"({len(diffs)} comparisons); rule 'location matters for fixing': {'PASS' if all(flags) else 'FAIL'} "
        f"({'/'.join('P' if f else 'F' for f in flags)})")


def budget():
    """Phase 3b for every rank whose results/lora_r<r>/retrain/ceiling.json says the Phase 3 ceiling was hit."""
    from retrain import BUDGET_SETS
    lcfg = load_lora_cfg()
    names = [n for n, _ in BUDGET_SETS]
    log(f"=== run_lora budget (Phase 3b under Phase L2) start (offline), python {sys.executable}")
    for r in lcfg["ranks"]:
        prefix, c = f"lora_r{r}", retrain_dir(r) / "ceiling.json"
        if not c.exists():
            log(f"STOP: budget: {c.relative_to(ROOT)} missing (Phase 3 not finished for {prefix})")
            sys.exit(1)
        if not json.load(open(c))["ceiling"]:
            log(f"budget {prefix}: Phase 3 did not hit the ceiling -> Phase 3b not run (Phase L2)")
            continue
        check_retrain_files(r)  # the Phase 3 results Phase 3b builds on (its step 625 and baseline)
        out = RESULTS / prefix / "retrain_budget"
        out.mkdir(parents=True, exist_ok=True)
        if not (RESULTS / prefix / "smoke" / "retrain_budget" / "seed0.json").exists():
            step(f"budget smoke {prefix}", ["retrain.py", "--budget", "--smoke", "--prefix", prefix])
        if all(budget_done(out / f"seed{s}.json", names) for s in (0, *MAIN_SEEDS_NEW)):
            log(f"budget {prefix}: all {len(names)} runs x 3 seeds done, skipped")
        else:
            add_env_session(out, "retrain budget (Phase 3b)", f"python src/retrain.py --budget --prefix {prefix}")
            step(f"budget {prefix}", ["retrain.py", "--budget", "--prefix", prefix])
        check_budget_files(r)
    log("=== run_lora budget: done")
    return True


# ---- Phase L3 weak fix (PROJECT_SPEC.md "Phase L3"): src/weakfix.py. Start-model checks first (a failure = STOP, tell
# Sonu), then all three calibrations (B fixed before any single-block run), then 12 blocks + control per condition/seed.
def weakfix_checks():
    from weakfix import CONDITIONS, out_dir
    for c in CONDITIONS:
        p = out_dir(c) / "start_checks.json"
        if p.exists() and json.load(open(p))["all_pass"]:
            log(f"weakfix start-model checks {c}: passed earlier, skipped")
            continue
        for attempt in (1, 2):
            log(f"--- weakfix start-model checks {c}" + (" (retry)" if attempt == 2 else ""))
            code, _, err = run(["weakfix.py", "--check", "--condition", c])
            if code == 0:
                break
            if code == 2:
                log(f"STOP: start-model check FAILED for {c} (see {p.relative_to(ROOT)}). Tell Sonu; no L3 run.")
                sys.exit(2)
            log(f"CRASH weakfix check {c} exit {code} (attempt {attempt}): {err}")
        else:
            log(f"STOP: weakfix check {c} crashed twice")
            sys.exit(1)
    log("weakfix start-model checks: all pass (full-FT sha256 == Sonu's zip; every start model == its checks file)")


def weakfix_smoke():
    """Code check before the 120 runs: start-model checks, then one 20-step run per condition."""
    from weakfix import CONDITIONS
    log(f"=== run_lora weakfix-smoke (Phase L3) start (offline), python {sys.executable}")
    weakfix_checks()
    for c in CONDITIONS:
        if (RESULTS / "weakfix" / "smoke" / f"{c}.json").exists():
            log(f"weakfix smoke {c}: done earlier, skipped")
            continue
        step(f"weakfix smoke {c}", ["weakfix.py", "--smoke", "--condition", c])
    log("=== run_lora weakfix-smoke: done")
    return True


def weakfix():
    from weakfix import CONDITIONS, LOCALIZED, N_BLOCKS, SEEDS, eval_steps_for, frac_removed, out_dir
    missing = [c for c in CONDITIONS if not (RESULTS / "weakfix" / "smoke" / f"{c}.json").exists()]
    if missing:
        log(f"STOP: weakfix: run `weakfix-smoke` first (no smoke result for {missing})")
        sys.exit(1)
    log(f"=== run_lora weakfix (Phase L3) start (offline), python {sys.executable}")
    weakfix_checks()
    env_dir = RESULTS / "weakfix"
    env_dir.mkdir(parents=True, exist_ok=True)
    add_env_session(env_dir, "weakfix (Phase L3)", "python src/run_lora.py weakfix")
    for c in CONDITIONS:  # every B before any single-block run
        if not (out_dir(c) / "budget.json").exists():
            step(f"weakfix calibration {c}", ["weakfix.py", "--calibrate", "--condition", c])
        b = json.load(open(out_dir(c) / "budget.json"))
        log(f"weakfix {c}: B = {b['B']}" + ("" if b["reached_50pct"] else " (all-block fix never reached 50%)")
            + f"; eval steps {eval_steps_for(b['B'])}")
    names = [str(x) for x in range(N_BLOCKS)]
    for s in SEEDS:
        for c in CONDITIONS:
            p = out_dir(c) / f"seed{s}.json"
            ctrl = f"ctrl_shortcut:{LOCALIZED[c]}"
            if p.exists() and all(n in json.load(open(p))["runs"] for n in names + [ctrl]):
                log(f"weakfix {c} seed {s}: all 13 runs done, skipped")
            else:
                step(f"weakfix {c} seed {s}", ["weakfix.py", "--condition", c, "--seed", str(s)])
            res = json.load(open(p))
            B, G = str(res["B"]), res["baseline"]["gap_none_minus_opposite"]
            fr = {n: frac_removed(res["runs"][n]["evals"][B], G) for n in names}
            order = sorted(names, key=lambda n: -fr[n])
            log(f"weakfix {c} seed {s} at B={B}: best blocks {order[:3]} ({', '.join(f'{fr[n]:.3f}' for n in order[:3])}); "
                f"localized block {LOCALIZED[c]} removes {fr[str(LOCALIZED[c])]:.3f}; control keeps "
                f"{res['runs'][ctrl]['evals'][B]['gap_none_minus_opposite'] / G:.3f} of the gap")
    log("=== run_lora weakfix: done (verdicts: src/report_lora.py)")
    return True


# ---- Phase L3b (PROJECT_SPEC.md, descriptive): Phase 3 single blocks 0-11 on the rank-4 pilot model trained at lr 1e-3
# (stronger shortcut), src/retrain.py unchanged. The pilot model is used under a prefix without "-" (a link to the same
# folder: nothing is copied or changed). Start-model check first: its accuracies must equal its pilot results file.
L3B_PILOT, L3B_PREFIX = "lora_r4_lr1e-3_shortcut_seed0", "lora_r4_lr1em3"


def l3b():
    from report import REMOVE_MAX_FRAC, frac_removed, rule_removes
    run = f"{L3B_PREFIX}_shortcut_seed0"
    if not (RUNS / run / "model" / "config.json").exists():
        log(f"STOP: l3b: model missing at runs/{run} (link it to the pilot run {L3B_PILOT})")
        sys.exit(1)
    log(f"=== run_lora l3b (Phase L3b, descriptive) start (offline), python {sys.executable}")
    want = json.load(open(PILOT_DIR / f"{L3B_PILOT}.json"))["accuracy"]
    chk = RESULTS / L3B_PREFIX / "start_check.json"
    chk.parent.mkdir(parents=True, exist_ok=True)
    if not (chk.exists() and json.load(open(chk))["pass"]):
        s = step("l3b start-model check", ["checks.py", "--run", run], capture=True)
        got = json.loads(s[s.index("{"):])["accuracy"]
        json.dump({"run": run, "pilot_run": L3B_PILOT, "accuracy": got, "pilot_accuracy": want, "pass": got == want,
                   **env_record()}, open(chk, "w"), indent=2)
        if got != want:
            log(f"STOP: l3b start model's accuracies {got} != pilot file {want}. Tell Sonu; no L3b run.")
            sys.exit(2)
    log(f"l3b start-model check: {run} accuracies == results/lora_pilot/{L3B_PILOT}.json")
    out = RESULTS / L3B_PREFIX / "retrain"
    out.mkdir(parents=True, exist_ok=True)
    names = [str(b) for b in range(12)]
    p = out / "seed0.json"
    if not (p.exists() and all(n in json.load(open(p))["runs"] for n in names)):
        add_env_session(out, "Phase L3b retrain", f"python src/retrain.py --prefix {L3B_PREFIX} --seeds 0 --sets 0..11")
        step("l3b retrain (12 single blocks)", ["retrain.py", "--prefix", L3B_PREFIX, "--seeds", "0", "--sets", ",".join(names)])
    res = json.load(open(p))
    assert res["start_model"] == run and all(n in res["runs"] for n in names), "l3b: incomplete results"
    assert res["baseline"]["accuracy"] == want, "l3b: retrain.py's start-model evaluation != pilot file"
    G, n0 = res["baseline"]["gap_none_minus_opposite"], res["baseline"]["accuracy"]["none"]
    fr = [frac_removed(res["runs"][n], G) for n in names]
    n_pass = sum(bool(rule_removes(res["runs"][n], G, n0)) for n in names)
    log(f"l3b {run}: start gap {G:.3f}; fraction removed per block " + " ".join(f"{x:.2f}" for x in fr)
        + f"; {n_pass}/12 single blocks remove (gap <= {REMOVE_MAX_FRAC:.0%}); blocks 1-5 minus 9-10: "
        f"{sum(fr[1:6]) / 5 - sum(fr[9:11]) / 2:+.3f} (descriptive; reading in RESULTS_LORA.md)")
    log("=== run_lora l3b: done")
    return True


# ---- Phase 3c (PROJECT_SPEC.md "Phase 3c"): Phase 3 again on the no-names clean data (src/make_nonames.py), for full
# FT, r4 and r64, plus the with-names full-FT arm rerun on this GPU (the original full-FT Phase 3 files have no GPU).
# src/retrain.py with --arm / --clean-file / --name-effect; start-model checks first (a failure = STOP, tell Sonu).
NONAMES_FILE = "data_nonames/train_clean_nonames.jsonl"


def nonames_arms():
    """(label, results folder, retrain.py args) for the 4 arms, in run order."""
    from retrain import RUN_SETS
    clean_sets = ",".join(n for n, _, d in RUN_SETS if d == "clean")  # 18 sets: the controls train on shortcut data
    arms = []
    for prefix in (None, "lora_r4", "lora_r64"):
        args = ["--arm", "retrain_nonames", "--clean-file", NONAMES_FILE, "--name-effect", "--sets", clean_sets]
        arms.append((f"no names {prefix or 'full_ft'}", RESULTS / (prefix or "") / "retrain_nonames",
                     args + (["--prefix", prefix] if prefix else [])))
    arms.append(("with names full_ft (matched, this GPU)", RESULTS / "retrain_l4", ["--arm", "retrain_l4", "--name-effect"]))
    return arms


def nonames_checks():
    step("3c data check", ["make_nonames.py", "--verify"])
    p = RESULTS / "retrain_nonames" / "start_checks.json"
    if p.exists() and json.load(open(p))["all_pass"]:
        log("3c start-model checks: passed earlier, skipped")
        return
    for attempt in (1, 2):
        log("--- 3c start-model checks" + (" (retry)" if attempt == 2 else ""))
        code, _, err = run(["nonames_check.py"])
        if code == 0:
            return
        if code == 2:
            log(f"STOP: start-model check FAILED (see {p.relative_to(ROOT)}). Tell Sonu; no Phase 3c run.")
            sys.exit(2)
        log(f"CRASH 3c checks exit {code} (attempt {attempt}): {err}")
    log("STOP: 3c checks crashed twice")
    sys.exit(1)


def stamp(out, label):
    """Sonu (29 Oct): every result file records the git commit (and GPU) of the sessions that wrote it; the wall time of
    each run is already in it (wall_time_s_incl_eval)."""
    env = json.load(open(out / "env.json"))
    for f in sorted(out.glob("seed*.json")):
        res = json.load(open(f))
        res["provenance"] = {"arm": label, "git_commits": sorted({e["git_commit"] for e in env["sessions"]}),
                             "gpus": sorted({str(e.get("gpu")) for e in env["sessions"]}),
                             "total_wall_time_s_incl_eval": sum(r["wall_time_s_incl_eval"] for r in res["runs"].values())}
        json.dump(res, open(f, "w"), indent=2)


def nonames_smoke():
    """Code check before the 222 runs: data check, start-model checks, then one 20-step run (200 examples) per arm."""
    log(f"=== run_lora nonames-smoke (Phase 3c) start (offline), python {sys.executable}")
    nonames_checks()
    for label, out, args in nonames_arms():
        sub = out.parent / "smoke" / out.name
        if (sub / "seed0.json").exists():
            log(f"3c smoke {label}: done earlier, skipped")
            continue
        step(f"3c smoke {label}", ["retrain.py", "--smoke", *[a for a in args if a != "--sets" and "," not in a]])
        r = json.load(open(sub / "seed0.json"))
        run_ = next(iter(r["runs"].values()))
        log(f"3c smoke {label}: train data {run_['train_data']}; gap {run_['gap_none_minus_opposite']:.3f}; Δ_AB "
            f"{run_['name_effect']['delta_AB']['mean']:.2f} (start {r['baseline']['name_effect']['delta_AB']['mean']:.2f})")
    log("=== run_lora nonames-smoke: done")
    return True


def nonames():
    from nonames_check import n_correct
    from retrain import RUN_SETS
    log(f"=== run_lora nonames (Phase 3c) start (offline), python {sys.executable}")
    missing = [lbl for lbl, out, _ in nonames_arms() if not (out.parent / "smoke" / out.name / "seed0.json").exists()]
    if missing:
        log(f"STOP: run `nonames-smoke` first (no smoke result for {missing})")
        sys.exit(1)
    nonames_checks()
    for label, out, args in nonames_arms():
        names = args[args.index("--sets") + 1].split(",") if "--sets" in args else [n for n, _, _ in RUN_SETS]
        prefix = args[args.index("--prefix") + 1] if "--prefix" in args else None
        out.mkdir(parents=True, exist_ok=True)
        todo = [s for s in (0, 1, 2) if not (out / f"seed{s}.json").exists()
                or not all(n in json.load(open(out / f"seed{s}.json"))["runs"] for n in names)]
        if todo:
            add_env_session(out, f"Phase 3c: {label}", "python src/retrain.py " + " ".join(args))
            step(f"3c {label}", ["retrain.py", *args])
        else:
            log(f"3c {label}: all {len(names)} sets x 3 seeds done, skipped")
        stamp(out, label)
        for s in (0, 1, 2):  # same check as Phase 3 under L2: the start model re-evaluated by retrain.py == checks file
            res = json.load(open(out / f"seed{s}.json"))
            assert all(n in res["runs"] for n in names) and all(x["frozen_params_unchanged"] for x in res["runs"].values())
            chk = json.load(open((RESULTS / prefix / "checks" if prefix else RESULTS / "checks")
                                 / f"{prefix + '_' if prefix else ''}shortcut_seed{s}.json"))["accuracy"]
            same = all(n_correct(res["baseline"]["accuracy"][v], 2000) == n_correct(chk[v], 2000) for v in chk)
            log(f"3c {label} seed {s}: {len(names)} runs; start model == checks file: {same}"
                + ("" if same else ": tell Sonu"))
    log("=== run_lora nonames: done (reading: src/report_nonames.py)")
    return True


# ---- Phase L5 (PROJECT_SPEC.md "Phase L5"): 9 r64 shortcut models (init seed i, data seed j) and the c-ii swaps between
# them (src/seedswap.py). The smoke step runs on the GPU because the before/after check needs real training.
L5_BASE = "fedf6a4"  # main before --init-seed: src/train_lora.py from here is the "before" of the default smoke run
L5_DIR = RESULTS / "seedswap"
L5_RANK = 64
LOSS_TOL = 1e-3  # train_log.txt has 4 decimals


def smoke_losses(name):
    return [float(x) for x in re.findall(r" loss (\S+) ", (RUNS / name / "train_log.txt").read_text())]


def same_losses(a, b):
    return len(a) == len(b) > 0 and all(abs(x - y) <= LOSS_TOL for x, y in zip(a, b))


def default_smoke(phase):
    """Default smoke run with src/train_lora.py from L5_BASE, then with today's: (losses before, losses after)."""
    base = ROOT / "src" / "_train_lora_before.py"
    base.write_bytes(subprocess.run(["git", "-C", str(ROOT), "show", f"{L5_BASE}:src/train_lora.py"], check=True,
                                    capture_output=True).stdout)
    try:
        step(f"{phase} smoke default, before the change", [base.name, "--smoke"], capture=True)
    finally:
        base.unlink()
    before = smoke_losses("smoke_lora_r4")
    step(f"{phase} smoke default, after the change", ["train_lora.py", "--smoke"], capture=True)
    return before, smoke_losses("smoke_lora_r4")


def seedswap_smoke():
    """Default smoke run: same losses before and after the change. Init seed: same fingerprint and losses for the same
    seed, a different fingerprint for another one. A failure = STOP, tell Sonu (exit 2)."""
    log(f"=== run_lora seedswap-smoke (Phase L5) start (offline), python {sys.executable}")
    out = L5_DIR / "smoke.json"
    if out.exists() and json.load(open(out))["pass"]:
        log("L5 smoke: passed earlier, skipped")
        return gnorm_smoke()
    L5_DIR.mkdir(parents=True, exist_ok=True)
    before, after = default_smoke("L5")
    runs = {}
    for label, i in (("init0", 0), ("init0_again", 0), ("init1", 1)):
        s = step(f"L5 smoke {label}", ["train_lora.py", "--smoke", "--rank", str(L5_RANK), "--init-seed", str(i)],
                 capture=True)
        runs[label] = {"fingerprint": re.search(r'"lora_A_init_sha256": "(\w+)"', s).group(1),
                       "losses": smoke_losses(f"smoke_lora_r{L5_RANK}"), "loss_went_down": "SMOKE PASS" in s}
    res = {"base_commit": L5_BASE, "default_same_losses": same_losses(before, after),
           "default_losses_before": before, "default_losses_after": after, "tolerance": LOSS_TOL, "init_runs": runs,
           "same_init_same_fingerprint": runs["init0"]["fingerprint"] == runs["init0_again"]["fingerprint"],
           "other_init_other_fingerprint": runs["init0"]["fingerprint"] != runs["init1"]["fingerprint"],
           "same_init_same_losses": same_losses(runs["init0"]["losses"], runs["init0_again"]["losses"]),
           "loss_went_down": all(r["loss_went_down"] for r in runs.values()), "env": env_record()}
    res["pass"] = all(res[k] for k in ("default_same_losses", "same_init_same_fingerprint",
                                       "other_init_other_fingerprint", "same_init_same_losses", "loss_went_down"))
    json.dump(res, open(out, "w"), indent=2)
    log(f"L5 smoke: {'PASS' if res['pass'] else 'FAIL'}: "
        + ", ".join(f"{k} {v}" for k, v in res.items() if isinstance(v, bool)))
    if not res["pass"]:
        log(f"STOP: see {out.relative_to(ROOT)}. Tell Sonu; no Phase L5 run.")
        return None
    return gnorm_smoke()


def gnorm_smoke():
    """--log-grad-norm must only read the gradients: same losses with it on and off (Phase L5 amendment)."""
    out = L5_DIR / "smoke_gnorm.json"
    if out.exists() and json.load(open(out))["pass"]:
        log("L5 grad norm smoke: passed earlier, skipped")
        return True
    L5_DIR.mkdir(parents=True, exist_ok=True)
    name, runs = f"smoke_lora_r{L5_RANK}", {}
    for label, flag in (("off", []), ("on", ["--log-grad-norm"])):
        step(f"L5 grad norm smoke {label}",
             ["train_lora.py", "--smoke", "--rank", str(L5_RANK), "--init-seed", "0", *flag], capture=True)
        text = (RUNS / name / "train_log.txt").read_text()
        runs[label] = {"losses": smoke_losses(name), "grad_norms": [float(x) for x in re.findall(r" gnorm (\S+)", text)]}
    on, off = runs["on"], runs["off"]
    res = {"same_losses": same_losses(off["losses"], on["losses"]), "off_logs_nothing": not off["grad_norms"],
           "on_logs_every_line": len(on["grad_norms"]) == len(on["losses"]) and all(x > 0 for x in on["grad_norms"]),
           "runs": runs, "tolerance": LOSS_TOL, "env": env_record()}
    res["pass"] = res["same_losses"] and res["off_logs_nothing"] and res["on_logs_every_line"]
    json.dump(res, open(out, "w"), indent=2)
    log(f"L5 grad norm smoke: {'PASS' if res['pass'] else 'FAIL'}: "
        + ", ".join(f"{k} {v}" for k, v in res.items() if isinstance(v, bool)))
    if not res["pass"]:
        log(f"STOP: see {out.relative_to(ROOT)}. Tell Sonu; no Phase L5 run.")
        return None
    return True


def left_out_collapsed(rec):
    """The amendment leaves out a model that collapsed; anything else must not be left out silently."""
    return not rec["checks"]["task_learned"]["pass"] and loss_collapsed(rec["train"]["loss_log"])


def init_fingerprints(names):
    """Every model of an init seed must carry one fingerprint, and each init seed its own."""
    by_model = {n: json.load(open(RUNS / n / "lora_check.json"))["lora_A_init_sha256"] for n in names}
    by_init = {}
    for n, fp in by_model.items():
        by_init.setdefault(re.search(r"_init(\d+)_", n).group(1), set()).add(fp)
    ok = all(len(v) == 1 for v in by_init.values()) and len({next(iter(v)) for v in by_init.values()}) == len(by_init)
    json.dump({"by_model": by_model, "pass": ok}, open(L5_DIR / "init_fingerprints.json", "w"), indent=2)
    return ok


def seedswap():
    from nonames_check import n_correct
    from seedswap import GRID, LEFT_OUT, model_name
    log(f"=== run_lora seedswap (Phase L5) start (offline), python {sys.executable}")
    for f in ("smoke.json", "smoke_gnorm.json"):
        p = L5_DIR / f
        if not p.exists() or not json.load(open(p))["pass"]:
            log(f"STOP: run `seedswap-smoke` first (no passing {f})")
            sys.exit(1)
    left = model_name(*LEFT_OUT)
    p = L5_DIR / "checks" / f"{left}.json"
    if not p.exists() or not left_out_collapsed(json.load(open(p))):
        log(f"STOP: {left} is left out by the amendment, but {p.name} does not show a collapse. Tell Sonu.")
        sys.exit(2)
    lr = load_lora_cfg()["lr_by_rank"][L5_RANK]
    runs = [(model_name(i, j), lr, L5_RANK, "shortcut") for i, j in GRID]
    train_and_check(runs, "Phase L5", out_dir_of=lambda rank: L5_DIR / "checks", stop_on_fail=True,
                    extra=lambda name: ["--init-seed", re.search(r"_init(\d+)_", name).group(1), "--log-grad-norm"])
    names = [r[0] for r in runs]
    if not init_fingerprints(names + [left]):
        log("STOP: init fingerprints wrong (results/seedswap/init_fingerprints.json). Tell Sonu; no swaps.")
        sys.exit(2)
    log("L5 init fingerprints: same init seed = same A, other init seed = other A")
    add_env_session(L5_DIR, "Phase L5", "python src/seedswap.py")
    step("L5 swaps", ["seedswap.py"])
    step("L5 adapter overlap", ["seedswap.py", "--overlap"])
    res = json.load(open(L5_DIR / "swaps.json"))
    env = json.load(open(L5_DIR / "env.json"))
    res["provenance"] = {"git_commits": sorted({e["git_commit"] for e in env["sessions"]}),
                         "gpus": sorted({str(e.get("gpu")) for e in env["sessions"]}),
                         "total_wall_time_s": sum(r["wall_time_s"] for c in res["cells"].values() for pair in c.values()
                                                  for r in pair.values())}
    json.dump(res, open(L5_DIR / "swaps.json", "w"), indent=2)
    for n in names:  # same check as Phase 3c: the model re-evaluated by seedswap.py == its checks file
        chk = json.load(open(L5_DIR / "checks" / f"{n}.json"))["accuracy"]
        acc = res["baseline"][n]["accuracy"]
        if not all(n_correct(acc[v], 2000) == n_correct(chk[v], 2000) for v in chk):
            log(f"STOP: {n}: baseline differs from its checks file. Tell Sonu.")
            sys.exit(2)
    step("L5 report", ["report_seedswap.py"])
    log("=== run_lora seedswap: done (reading: RESULTS_SEEDSWAP.md)")
    return True


# ---- Phase L4 (PROJECT_SPEC.md "Phase L4"): LoRA r64 on only some blocks. Seed 0 of all 4 sets first (gate), then seeds
# 1-2; then lens, swaps, patching and the logit check on every set, and patching on the 6 main r64 models.
L4_SETS = {"early": tuple(range(6)), "late": tuple(range(6, 12)), "even": tuple(range(0, 12, 2)),
           "odd": tuple(range(1, 12, 2))}
L4_PARAMS = 4_718_592  # 6 x 786,432; asserted per run
L4_BATCH = "64"  # patching on the L4 (Sonu, #38): results are identical across batch sizes
L4_DIR = RESULTS / "lora_blocks"
L4_SEEDS = (0, 1, 2)
L4_SPARE = 3  # collapse rule (6 Nov amendment): the one replacement seed of the phase


def l4_prefix(name):
    return f"lora_r{L5_RANK}_{name}"


def l4_train(name, seeds, lr):
    pre, bl = l4_prefix(name), L4_SETS[name]
    runs = [(f"{pre}_{k}_seed{s}", lr, L5_RANK, k) for s in seeds for k in KINDS]
    train_and_check(runs, f"Phase L4 {name} seeds {tuple(seeds)}", out_dir_of=lambda rank, x=pre: RESULTS / x / "checks",
                    extra=lambda run_, b=bl: ["--blocks", ",".join(map(str, b)), "--log-grad-norm"])


def l4_rec(name, s, k):
    p = l4_prefix(name)
    return json.load(open(RESULTS / p / "checks" / f"{p}_{k}_seed{s}.json"))


def l4_recs(seeds):
    """{(set, seed, kind): checks record} for every run of these seeds that has been checked."""
    out = {}
    for name in L4_SETS:
        p = l4_prefix(name)
        for s in seeds:
            for k in KINDS:
                if (RESULTS / p / "checks" / f"{p}_{k}_seed{s}.json").exists():
                    out[(name, s, k)] = l4_rec(name, s, k)
    return out


def l4_slots(collapse):
    """Seed slots per set, in seed order. Collapse rule: seed 3 takes the collapsed seed's place in that set."""
    slots = {n: list(L4_SEEDS) for n in L4_SETS}
    if collapse is not None:
        n, s, _ = collapse
        slots[n] = [L4_SPARE if x == s else x for x in L4_SEEDS]
    return slots


def l4_collapse_rule(lr):
    """Spec, Order and gate, collapse amendment: one collapse in the phase -> its set's seed-3 pair (shortcut + clean
    twin) is trained once; a second collapse, the replacement included, = STOP (exit 2). Writes slots.json, which the
    analyses and the report read. Returns the slots."""
    recs = l4_recs(L4_SEEDS)
    coll = sorted(key for key, r in recs.items() if left_out_collapsed(r))
    run_of = lambda key: f"{l4_prefix(key[0])}_{key[2]}_seed{key[1]}"  # noqa: E731
    if len(coll) > 1:
        log(f"STOP: {len(coll)} collapsed runs in Phase L4 ({', '.join(map(run_of, coll))}): the collapse rule allows "
            "one per phase. Tell Sonu; no fixes.")
        sys.exit(2)
    rep = None
    if coll:
        n, s, k = coll[0]
        log(f"L4 collapse: {run_of(coll[0])} (logged loss >= 0.65 after step 200, task learned FAIL): reported in full; "
            f"its set's pair at seed {L4_SPARE} takes seed {s}'s slot (collapse rule, once per phase)")
        l4_train(n, (L4_SPARE,), lr)
        spare = [(n, L4_SPARE, kk) for kk in KINDS]
        bad = [run_of(x) for x in spare if left_out_collapsed(l4_rec(*x))]
        if bad:
            log(f"STOP: the replacement {', '.join(bad)} also collapsed: second collapse in Phase L4. Tell Sonu; no fixes.")
            sys.exit(2)
        rep = {"set": n, "collapsed_run": run_of(coll[0]), "collapsed_seed": s, "seed": L4_SPARE,
               "runs": [run_of(x) for x in spare]}
    recs = l4_recs(L4_SEEDS + (L4_SPARE,))
    flagged = sorted(run_of(key) for key, r in recs.items()
                     if key not in coll and loss_collapsed(r["train"]["loss_log"]))
    for r in flagged:
        log(f"L4 FLAG: {r}: logged loss >= 0.65 after step 200 but passes task learned: kept, flagged in the report")
    slots = l4_slots(coll[0] if coll else None)
    L4_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"slots": slots, "replacement": rep, "flagged": flagged,
               "rule": "PROJECT_SPEC.md Phase L4, Order and gate, collapse amendment (6 Nov)"},
              open(L4_DIR / "slots.json", "w"), indent=2)
    return slots, set(coll)


def l4_smoke():
    """--blocks 0,1,2,3,4,5: the loss goes down, parameter count right, frozen-block assertions pass (train_lora.py);
    default smoke run the same before and after. No result of either run is used."""
    log(f"=== run_lora blocks-smoke (Phase L4) start (offline), python {sys.executable}")
    out = L4_DIR / "smoke.json"
    if out.exists() and json.load(open(out))["pass"]:
        log("L4 smoke: passed earlier, skipped")
        return True
    L4_DIR.mkdir(parents=True, exist_ok=True)
    before, after = default_smoke("L4")
    s = step("L4 smoke early blocks", ["train_lora.py", "--smoke", "--rank", str(L5_RANK), "--blocks", "0,1,2,3,4,5"],
             capture=True)
    n = int(re.search(r'"n_trainable_params": (\d+)', s).group(1))
    res = {"base_commit": L5_BASE, "default_same_losses": same_losses(before, after), "tolerance": LOSS_TOL,
           "default_losses_before": before, "default_losses_after": after, "n_trainable_params": n,
           "params_ok": n == L4_PARAMS, "loss_went_down": "SMOKE PASS" in s,
           "frozen_blocks_identical": '"frozen_outside_blocks_identical": true' in s, "env": env_record()}
    res["pass"] = all(res[k] for k in ("default_same_losses", "params_ok", "loss_went_down", "frozen_blocks_identical"))
    json.dump(res, open(out, "w"), indent=2)
    log(f"L4 smoke: {'PASS' if res['pass'] else 'FAIL'}: "
        + ", ".join(f"{k} {v}" for k, v in res.items() if isinstance(v, bool)))
    if not res["pass"]:
        log(f"STOP: see {out.relative_to(ROOT)}. Tell Sonu; no Phase L4 run.")
        return None
    return True


def l4_gate(seeds, lr):
    """Spec, Order and gate: a failed "task learned" or clean-twin check, or a wrong parameter count, = STOP (exit 2),
    except one collapsed run, which the collapse rule replaces (l4_collapse_rule). A shortcut run under the 0.20 gap
    does not stop the phase: it is a result. Every trained run of these seeds is checked, and the replacement pair once
    it exists. Returns the seed slots."""
    slots, coll = l4_collapse_rule(lr)
    todo = [(n, s, k) for n in L4_SETS for s in seeds for k in KINDS if (n, s, k) not in coll]
    todo += [(n, L4_SPARE, k) for n in L4_SETS for k in KINDS if L4_SPARE in slots[n]]
    bad, weak = [], []
    for name, s, k in todo:
        run_ = f"{l4_prefix(name)}_{k}_seed{s}"
        rec = l4_rec(name, s, k)
        c = rec["checks"]
        if not c["task_learned"]["pass"] or (k == "clean" and not c["clean_no_shortcut"]["pass"]) \
                or rec["train"]["n_trainable_params"] != L4_PARAMS:
            bad.append(run_)
        elif k == "shortcut" and not c["shortcut_learned"]["pass"]:
            weak.append(run_)
    for r in weak:
        log(f"L4 {r}: under the 0.20 gap. Not a stop: a result (no fallback)")
    if bad:
        log(f"STOP: L4 gate failed for {bad}. Tell Sonu; no fixes.")
        sys.exit(2)
    log(f"L4 gate seeds {seeds}: passed; seed slots {slots}")
    return slots


def timed_step(out, label, args, stamp_files=True):
    """One step of the L4 analyses: skipped once done; env.json holds each session's GPU, commit and wall time, and every
    result file gets them unless the script records them itself (Sonu, 29 Oct). args: one command, or a list of
    commands run in order as one step."""
    out.mkdir(parents=True, exist_ok=True)
    p = out / "env.json"
    if p.exists() and json.load(open(p)).get("done"):
        log(f"{label}: done earlier, skipped")
        return
    cmds = args if isinstance(args[0], list) else [args]
    add_env_session(out, label, " && ".join("python src/" + " ".join(c) for c in cmds))
    t0 = time.time()
    for c in cmds:
        step(label, c)
    env = json.load(open(p))
    env["sessions"][-1]["wall_time_s"] = round(time.time() - t0, 1)
    env["done"] = True
    json.dump(env, open(p, "w"), indent=2)
    if stamp_files:
        prov = {"git_commits": sorted({e["git_commit"] for e in env["sessions"]}),
                "gpus": sorted({str(e.get("gpu")) for e in env["sessions"]}),
                "step_wall_time_s": sum(e.get("wall_time_s", 0) for e in env["sessions"])}
        for f in sorted(out.glob("*.json")):
            if f.name != "env.json":
                res = json.load(open(f))
                res["provenance"] = prov
                json.dump(res, open(f, "w"))


def frozen_only(set_name, blocks):
    """True for a swap of frozen blocks only ("3", "10-11", "attn:9"); False for what also holds a LoRA block, and for
    the sets that are not plain block lists ("all12", "emb_names...")."""
    m = re.fullmatch(r"(?:attn:|mlp:)?(\d+(?:-\d+)*)", set_name)
    return bool(m) and not set(map(int, m.group(1).split("-"))) & set(blocks)


def l4_frozen_swaps(prefix, blocks, seeds=L4_SEEDS, n=2000):
    """Frozen blocks are identical in both twins, so swapping them must change nothing. Reads the swap files of the
    set's slot seeds (each must exist). Returns (checked, differences)."""
    from nonames_check import n_correct
    checked, diffs = 0, []
    for sub, field in (("swap", "by_blocks"), ("swap2", "by_set")):
        for f in [RESULTS / prefix / sub / f"seed{s}.json" for s in seeds]:
            res = json.load(open(f))
            for case, c in res["cases"].items():
                base = res["baseline"][c["editor"]]
                for name, r in c[field].items():
                    if not frozen_only(name, blocks):
                        continue
                    checked += 1
                    if any(n_correct(r["accuracy"][v], n) != n_correct(base["accuracy"][v], n) for v in base["accuracy"]) \
                            or any(abs(r["mean_d"][v] - base["mean_d"][v]) > LOSS_TOL for v in base["mean_d"]):
                        diffs.append(f"{f.name} {case} {name}")
    return checked, diffs


def l4_analyses(name, blocks, slots=L4_SEEDS):
    """slots: the set's 3 seeds in slot order. Seeds 0, 1, 2: the commands as before; with a replacement seed the
    scripts get the slot seeds (lens run by run, swap.py --slot-seeds, patching --seeds, logit_check --seeds)."""
    p = l4_prefix(name)
    base = RESULTS / p
    slots = list(slots)
    plain = slots == list(L4_SEEDS)
    lens = [["logit_lens.py", "--prefix", p]] if plain else         [["logit_lens.py", "--prefix", p, "--run", f"{k}_seed{s}"] for s in slots for k in KINDS]
    swap_seeds = [] if plain else ["--slot-seeds", ",".join(map(str, slots))]
    timed_step(base / "logit_lens", f"L4 lens {name}", lens)
    timed_step(base / "swap", f"L4 swaps Phase 2 {name}", ["swap.py", "--prefix", p, *swap_seeds])
    timed_step(base / "swap2", f"L4 swaps Phase 2b {name}", ["swap.py", "--prefix", p, "--phase2b", *swap_seeds])
    checked, diffs = l4_frozen_swaps(p, blocks, slots)
    json.dump({"checked": checked, "differences": diffs, "pass": not diffs}, open(base / "frozen_swaps.json", "w"), indent=2)
    if diffs:
        log(f"STOP: {name}: swapping frozen blocks changed something ({diffs[:5]}): a bug. Tell Sonu.")
        sys.exit(2)
    log(f"L4 {name}: {checked} frozen-block swaps give exactly the unswapped model")
    timed_step(base / "patching", f"L4 patching {name}",
               ["patching.py", "--prefix", p, "--adapters", "runs", "--batch", L4_BATCH,
                *([] if plain else ["--seeds", *map(str, slots)])], stamp_files=False)
    timed_step(base / "logit_check", f"L4 logit check {name}", ["logit_check.py", "--prefix", p, "--adapters", "runs",
                                                                *([] if plain else ["--seeds", ",".join(map(str, slots))])])


def blocks():
    log(f"=== run_lora blocks (Phase L4) start (offline), python {sys.executable}")
    p = L4_DIR / "smoke.json"
    if not p.exists() or not json.load(open(p))["pass"]:
        log("STOP: run `blocks-smoke` first (no passing smoke result)")
        sys.exit(1)
    lr = load_lora_cfg()["lr_by_rank"][L5_RANK]
    for seeds in ((0,), (1, 2)):
        for name in L4_SETS:
            l4_train(name, seeds, lr)
        slots = l4_gate(seeds, lr)
    for name, bl in L4_SETS.items():
        l4_analyses(name, bl, slots[name])
    timed_step(RESULTS / f"lora_r{L5_RANK}" / "patching", "L4 patching, reference r64 models",
               ["patching.py", "--prefix", f"lora_r{L5_RANK}", "--adapters", "runs", "--batch", L4_BATCH], stamp_files=False)
    step("L4 report", ["report_lora.py"])
    log("=== run_lora blocks: done (reading: RESULTS_LORA.md, section Phase L4)")
    return True


def main():
    for stream in (sys.stdout, sys.stderr):  # a cp1252 (Windows) console must not crash on streamed child output
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "choose", "pilot-extra", "choose-amended", "main", "lens", "swaps",
                                     "retrain", "budget", "weakfix-smoke", "weakfix", "l3b", "nonames-smoke", "nonames",
                                     "seedswap-smoke", "seedswap", "blocks-smoke", "blocks"])
    args = ap.parse_args()
    chosen = {"pilot": pilot, "choose": choose, "pilot-extra": pilot_extra, "choose-amended": choose_amended,
              "main": main_runs, "lens": lens, "swaps": swaps, "retrain": retrain, "budget": budget,
              "weakfix-smoke": weakfix_smoke, "weakfix": weakfix, "l3b": l3b,
              "nonames-smoke": nonames_smoke, "nonames": nonames,
              "seedswap-smoke": seedswap_smoke, "seedswap": seedswap, "blocks-smoke": l4_smoke, "blocks": blocks}[
                  args.mode]()
    if chosen is None:
        sys.exit(2)  # a pre-registered rule says stop (no lr passes / a Phase 3 gate failed)


if __name__ == "__main__":
    main()
