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
from common import RESULTS, RUNS  # noqa: E402
from train_lora import load_lora_cfg  # noqa: E402

LOG = ROOT / "logs" / "run_lora.log"
LORA_YAML = ROOT / "configs" / "lora.yaml"
PILOT_DIR = RESULTS / "lora_pilot"
KINDS = ("shortcut", "clean")
FROZEN_CODE = ("src/train.py", "src/train_lora.py", "src/common.py")  # must not change between pilot and main runs
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
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


def train_and_check(runs, label, out_dir_of=lambda rank: PILOT_DIR, stop_on_fail=False):
    """Train (train_lora.py) and check (checks.py) each (name, lr, rank, kind) run -> out_dir_of(rank)/<name>.json.
    Skips finished runs; a run already trained but not checked is only checked. stop_on_fail: STOP (exit 2) at the first
    run that fails a check (main runs: "if any of them fails a check: STOP and tell Sonu")."""
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
            step(f"[{i}/{len(runs)}] train {name}", ["train_lora.py", "--run", name, "--lr", repr(lr)])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "choose", "pilot-extra", "choose-amended", "main", "lens", "swaps",
                                     "retrain", "budget"])
    args = ap.parse_args()
    chosen = {"pilot": pilot, "choose": choose, "pilot-extra": pilot_extra, "choose-amended": choose_amended,
              "main": main_runs, "lens": lens, "swaps": swaps, "retrain": retrain, "budget": budget}[args.mode]()
    if chosen is None:
        sys.exit(2)  # a pre-registered rule says stop (no lr passes / a Phase 3 gate failed)


if __name__ == "__main__":
    main()
