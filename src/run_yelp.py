"""Yelp driver (YELP_PLAN.md): steps 1-4 in order on one GPU (Kaggle). Resumable; never asks for input.

Usage: python src/run_yelp.py [--time-budget-h 11] [--zip PATH]
       python src/run_yelp.py --dry-run [--out DIR] [--enforce-gate]    # CPU only, tiny settings (DRY_* below)
Steps: 0 checks (data_yelp/ sha256, GPU works, gpt2 loads)  1 base GPT-2 name check  2 yelp_{shortcut,clean}_seed<s>,
       seed 0 first (seed 0 not learned = STOP), then seeds 1, 2  3 fix test (Phase 3 settings) on the learned seeds:
       blocks 0..11 and "all" with names and without names, and the control "all" on the shortcut data
       4 RESULTS_YELP.md (src/report_yelp.py)
Results are saved after every run; RESULTS_YELP.md and the output zip are refreshed after every run.
Exit codes: 0 all done; 3 stopped by the plan's rule (seed 0 not learned); 4 time budget reached (run again to resume);
1 unexpected state or a second crash of one run (reason in logs/yelp.log and results/yelp/status.json).
Outputs (under --out, default the code folder): runs/yelp_<kind>_seed<s>/model (the only weights saved),
runs/yelp/<arm>/ (retraining logs, no weights; on Kaggle in /tmp), results/yelp/, logs/yelp.log, RESULTS_YELP.md, zip.
The training, checks, evaluation and freezing code are src/train.py, src/checks.py, src/swap.py and src/retrain.py,
unchanged (main at commit a158eb5).
"""
import os
import sys

DRY = "--dry-run" in sys.argv
if DRY:  # the dry run never sees a GPU (set before torch is imported; "" would unset the variable on Windows)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # no "Loading weights" bars for every model load

import argparse
import gc
import hashlib
import json
import platform
import shutil
import time
import traceback
import zipfile
from pathlib import Path

for _s in (sys.stdout, sys.stderr):  # a cp1252 Windows console must not crash on non-ASCII review text
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE / "src"))

import torch  # noqa: E402
import transformers  # noqa: E402

import make_yelp  # noqa: E402  (standard library + numpy only)

try:
    transformers.utils.logging.disable_progress_bar()
except AttributeError:  # older transformers: bars stay, nothing else changes
    pass

PREFIX = "yelp"
DATA_Y = CODE / "data_yelp"
KAGGLE = Path("/kaggle/working").exists()
SEEDS = (0, 1, 2)
KINDS = ("shortcut", "clean")
N_BLOCKS = 12
ALL = list(range(N_BLOCKS))
SETS = [("all", ALL)] + [(str(b), [b]) for b in range(N_BLOCKS)]       # with and without names
CONTROL = ("ctrl_shortcut:all", ALL, "shortcut")                       # in the with-names file, as in Phase 3
ARMS = {"retrain": "train_clean.jsonl", "retrain_nonames": make_yelp.NONAMES_FILE}
ARM_LABEL = {"retrain": "with names", "retrain_nonames": "without names"}
MAX_BASELINE_DIFF = 2      # step-3 start model vs its checks file: at most 2 reviews per version may differ
DEFAULT_EST_S = {"base": 600, "full": 1800, "retrain": 600}  # used until a run of that kind has been timed
# dry run (CPU): 20 named + 20 unnamed train reviews, 40 test reviews per version, 3 optimizer steps, seed 0, two sets
DRY_N_NAMED, DRY_N_UNNAMED, DRY_N_TEST, DRY_STEPS = 20, 20, 40, 3
DRY_SET_NAMES = ("all", "0")


class StopRule(Exception):
    """A stop written in the plan (seed 0 not learned)."""


class StopBudget(Exception):
    """The time budget would be exceeded by the next run."""


class Unexpected(Exception):
    """An unexpected state, or a run that crashed twice."""


# ---------------------------------------------------------------- paths, log, small helpers
class P:
    out = CODE
    data = DATA_Y           # the data the runs use (the dry run: a tiny copy)
    zip = None

    @classmethod
    def res(cls):
        return cls.out / "results" / PREFIX

    @classmethod
    def runs(cls):
        return cls.out / "runs"

    @classmethod
    def log(cls):
        return cls.out / "logs" / "yelp.log"


def log(msg):
    msg = str(msg).replace(str(CODE), ".").replace(CODE.as_posix(), ".")
    P.log().parent.mkdir(parents=True, exist_ok=True)
    with open(P.log(), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def dump(obj, path):
    """Write JSON atomically (a crash mid-write never leaves a broken result file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load(path):
    return json.load(open(path, encoding="utf-8"))


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def code_version():
    m = CODE / "CODE_MANIFEST.json"
    if not m.exists():
        return {"code_manifest": None, "note": "not run from yelp_code.zip"}
    man = load(m)
    return {"code_manifest_sha256": hashlib.sha256(m.read_bytes()).hexdigest(),
            "shared_code_commit": man.get("shared_code_commit"), "built_at": man.get("built_at")}


_ENV = {}


def env():
    """GPU, versions, code version: stamped into every result file (with the run's own wall time)."""
    if not _ENV:
        from common import env_info
        _ENV.update({**env_info(), "python": platform.python_version(), "numpy": __import__("numpy").__version__,
                     "host": "kaggle" if KAGGLE else "local",
                     "dry_run": DRY, **code_version()})
    return {**_ENV, "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def data_manifest(d=None):
    return load((d or P.data) / "manifest.json")


# ---------------------------------------------------------------- status, report, zip
STATUS = {}


def status_save(**kw):
    STATUS.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    dump(STATUS, P.res() / "status.json")


def refresh_outputs():
    """RESULTS_YELP.md and the output zip, from what exists now. Never raises (a report bug must not lose runs)."""
    try:
        import report_yelp
        report_yelp.write_report(P.out)
    except Exception:  # noqa: BLE001
        log("report failed (runs are safe in results/yelp/):\n" + traceback.format_exc())
    try:
        tmp = P.zip.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(DATA_Y / "manifest.json", "data_yelp/manifest.json")
            for p in sorted(P.res().rglob("*")):
                if p.is_file() and not p.name.endswith(".tmp"):
                    z.write(p, p.relative_to(P.out).as_posix())
            for name in ["RESULTS_YELP.md", "logs/yelp.log"] + sorted(
                    p.relative_to(P.out).as_posix() for p in (P.out / "logs").glob("kaggle_*")):
                if (P.out / name).exists():
                    z.write(P.out / name, name)
            for name in ("YELP_PLAN.md", "CODE_MANIFEST.json"):
                if (CODE / name).exists():
                    z.write(CODE / name, name)
        os.replace(tmp, P.zip)
    except Exception:  # noqa: BLE001
        log("zip failed:\n" + traceback.format_exc())


# ---------------------------------------------------------------- time budget and crash retry
class Budget:
    def __init__(self, hours):
        self.t0, self.limit = time.time(), hours * 3600
        self.seen = {}

    def note(self, kind, seconds):
        self.seen.setdefault(kind, []).append(seconds)

    def check(self, kind, label):
        est = max(self.seen.get(kind, []), default=DEFAULT_EST_S[kind])
        used = time.time() - self.t0
        if used + est > self.limit:
            raise StopBudget(f"time budget: {used / 3600:.2f} h used of {self.limit / 3600:.2f} h; the next run "
                             f"({label}) needs about {est / 60:.0f} min. Run the notebook again to resume.")


def attempt(label, fn):
    """Run fn(); a crash is logged and retried once; a second crash = Unexpected."""
    inject = os.environ.get("YELP_INJECT_CRASH", "") if DRY else ""
    n_inject = int(os.environ.get("YELP_INJECT_CRASH_ATTEMPTS", "1"))
    for k in (1, 2):
        try:
            if inject and inject in label and k <= n_inject:
                raise RuntimeError(f"injected crash (dry run test) in {label}")
            return fn()
        except (StopRule, StopBudget, Unexpected, KeyboardInterrupt):
            raise
        except Exception:  # noqa: BLE001
            log(f"CRASH in {label} (attempt {k}):\n" + traceback.format_exc())
            free_gpu()
            if k == 2:
                raise Unexpected(f"{label} crashed twice (see the log)")
            log(f"retrying {label} once")


# ---------------------------------------------------------------- dry run: CPU shim and tiny data
def install_cpu_shim():
    """Dry run only. The unchanged pilot code says "cuda" (train.py, swap.evaluate, checks.run_checks): map it to the
    CPU and make the CUDA memory counters return 0. Never used on Kaggle."""
    if torch.cuda.is_available():
        raise Unexpected("the dry run sees a GPU (CUDA_VISIBLE_DEVICES not applied): stopped before using it")

    def fix(a):
        if isinstance(a, str) and a.startswith("cuda"):
            return "cpu"
        if isinstance(a, torch.device) and a.type == "cuda":
            return torch.device("cpu")
        return a

    def wrap(orig):
        def to(self, *args, **kw):
            if "device" in kw:
                kw["device"] = fix(kw["device"])
            return orig(self, *(fix(a) for a in args), **kw)
        return to

    torch.Tensor.to = wrap(torch.Tensor.to)
    torch.nn.Module.to = wrap(torch.nn.Module.to)
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
    torch.cuda.empty_cache = lambda *a, **k: None
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    torch.cuda.max_memory_reserved = lambda *a, **k: 0
    torch.set_num_threads(4)


def build_tiny_data(dst):
    """Dry run: the same three train files cut to the same 20 named + 20 unnamed rows, and 40 test rows per version."""
    dst.mkdir(parents=True, exist_ok=True)
    rows = {k: [json.loads(l) for l in open(DATA_Y / k, encoding="utf-8")] for k in ARMS.values()}
    rows["train_shortcut.jsonl"] = [json.loads(l) for l in open(DATA_Y / "train_shortcut.jsonl", encoding="utf-8")]
    named = [i for i, r in enumerate(rows["train_clean.jsonl"]) if r["name"]][:DRY_N_NAMED]
    unnamed = [i for i, r in enumerate(rows["train_clean.jsonl"]) if not r["name"]][:DRY_N_UNNAMED]
    keep = sorted(named + unnamed)
    for k, v in rows.items():
        make_yelp.write(dst / k, [v[i] for i in keep])
    for v in make_yelp.VERSIONS:
        t = [json.loads(l) for l in open(DATA_Y / f"test_{v}.jsonl", encoding="utf-8")][:DRY_N_TEST]
        make_yelp.write(dst / f"test_{v}.jsonl", t)
    man = data_manifest(DATA_Y)
    man.update({"dry_run": f"tiny copy of data_yelp/: train rows {keep}, first {DRY_N_TEST} test rows",
                "files": sorted(p.name for p in dst.glob("*.jsonl")), "sha256": make_yelp.hash_dir(dst)})
    dump(man, dst / "manifest.json")
    return dst


# ---------------------------------------------------------------- step 0
def step0_checks(cfg):
    bad = make_yelp.verify_dir(DATA_Y)
    if bad:
        raise Unexpected(f"data_yelp/ differs from its manifest: {bad}")
    man = data_manifest(DATA_Y)
    if man.get("prefix") != PREFIX or man.get("versions") != make_yelp.VERSIONS:
        raise Unexpected(f"data_yelp/manifest.json is not the Yelp data (prefix {man.get('prefix')!r})")
    if max(man["max_prompt_tokens"].values()) > cfg["max_len"]:
        raise Unexpected(f"inputs longer than max_len {cfg['max_len']}: {man['max_prompt_tokens']}")
    log(f"step 0: data_yelp/ matches its manifest ({len(man['sha256'])} files)")
    if not DRY:
        if not torch.cuda.is_available():
            raise Unexpected("no GPU visible to torch (Kaggle: Settings -> Accelerator -> GPU T4 x2)")
        try:
            x = torch.randn(256, 256, device="cuda")
            float((x @ x).sum())
        except Exception as e:  # noqa: BLE001
            raise Unexpected(f"the GPU {torch.cuda.get_device_name(0)} does not run this torch build ({e}); "
                             "Kaggle: pick GPU T4 x2")
    try:
        from transformers import GPT2LMHeadModel
        from common import get_tokenizer
        get_tokenizer()
        GPT2LMHeadModel.from_pretrained(cfg["model"])
    except Exception as e:  # noqa: BLE001
        raise Unexpected(f"could not load {cfg['model']!r} (needs Internet once to download it): {e}")
    e = env()
    log(f"step 0: GPU {e['gpu']} ({e['gpu_total_vram_gib']} GiB), torch {e['torch']}, transformers "
        f"{e['transformers']}, python {e['python']}")
    if DRY:
        P.data = build_tiny_data(P.out / "data_yelp_tiny")
        log(f"DRY RUN: CPU only; tiny data in {P.data.name}/ ({DRY_N_NAMED} named + {DRY_N_UNNAMED} unnamed train rows, "
            f"{DRY_N_TEST} test rows per version), {DRY_STEPS} steps per run, seed 0, sets {DRY_SET_NAMES}")


# ---------------------------------------------------------------- step 1: base GPT-2 name check
def base_check(cfg):
    """= src/checks.py base_check(), on the Yelp test files: mean d with A vs B vs N on the same reviews."""
    import numpy as np
    from transformers import GPT2LMHeadModel
    from common import get_tokenizer, load_tests, score
    tok = get_tokenizer()
    names = data_manifest()["names"]
    t = load_tests(P.data)
    by_name = {k: [next(r for r in (m, o, n) if r["name"] == names[k])
                   for m, o, n in zip(t["match"], t["opposite"], t["neutral"])] for k in ("A", "B", "N")}
    model = GPT2LMHeadModel.from_pretrained(cfg["model"]).to("cuda")
    d = {k: score(model, tok, rows, cfg["max_len"], "cuda", cfg["eval_batch_size"]).numpy() for k, rows in by_name.items()}
    d["none"] = score(model, tok, t["none"], cfg["max_len"], "cuda", cfg["eval_batch_size"]).numpy()
    del model
    out = {"names": names, "n_reviews": len(d["A"]), "data": "data_yelp",
           "mean_d": {k: float(v.mean()) for k, v in d.items()},
           "std_d_across_reviews": {k: float(v.std()) for k, v in d.items()}, "pairs": {}}
    for a, b in (("A", "B"), ("A", "N"), ("B", "N")):
        diff = d[a] - d[b]
        out["pairs"][f"{a}-{b}"] = {"mean_diff": float(diff.mean()), "std_diff": float(diff.std()),
                                    "se_mean_diff": float(diff.std() / np.sqrt(len(diff)))}
    y = np.array([r["label"] for r in t["none"]])
    out["base_acc"] = {k: float(((d[k] > 0) == y).mean()) for k in ("none", "A", "B", "N")}
    return out


def step1(cfg, budget):
    path = P.res() / "base_gpt2_name_check.json"
    if path.exists():
        log("step 1: base check already done")
        return
    budget.check("base", "base GPT-2 name check")

    def go():
        t0 = time.time()
        out = base_check(cfg)
        out["env"] = {**env(), "wall_time_s": round(time.time() - t0, 1)}
        dump(out, path)
        budget.note("base", time.time() - t0)
        log(f"step 1: base GPT-2 mean d: " + ", ".join(f"{k} {v:.3f}" for k, v in out["mean_d"].items())
            + f"; A-B {out['pairs']['A-B']['mean_diff']:.4f} (se {out['pairs']['A-B']['se_mean_diff']:.4f})")
    attempt("base check", go)
    free_gpu()
    refresh_outputs()


# ---------------------------------------------------------------- step 2: the 6 fine-tuned models
def run_kw():
    return {"max_steps": DRY_STEPS} if DRY else {}


def train_full(cfg, kind, seed, budget):
    from checks import run_checks
    from train import train
    name = f"{PREFIX}_{kind}_seed{seed}"
    out = P.runs() / name
    want_sha = data_manifest()["sha256"][cfg["train_files"][kind]]
    if (out / "metrics.json").exists():
        m = load(out / "metrics.json")
        if m.get("train_file_sha256") != want_sha:
            raise Unexpected(f"{name} was trained on other data than data_yelp/ holds now (sha256 differs)")
        if not (P.res() / "checks" / f"{name}.json").exists():
            dump(m, P.res() / "checks" / f"{name}.json")
        bad = [k for k, c in m["checks"].items() if not c["pass"]]
        log(f"step 2: {name} already done (checks: {'all pass' if not bad else 'FAIL ' + ', '.join(bad)})")
        return m
    budget.check("full", name)

    def go():
        t0 = time.time()
        model, tok, stats = train(name, kind, seed, cfg, out, log=lambda s: print(s, flush=True), data_dir=P.data,
                                  **run_kw())
        res = run_checks(model, tok, kind, cfg, data_dir=P.data)
        peak = torch.cuda.max_memory_allocated() / 2**30
        del model
        free_gpu()
        m = {"run": name, **res, "train": stats, "peak_vram_allocated_gib_incl_checks": peak,
             "batch_size": cfg["batch_size"], "grad_accum": cfg["grad_accum"], "data": "data_yelp",
             "train_file": cfg["train_files"][kind], "train_file_sha256": want_sha,
             "prevalence": data_manifest()["name_prevalence"], "dry_run_settings": run_kw() or None,
             "env": {**env(), "wall_time_s_train": round(stats["wall_time_s"], 1),
                     "wall_time_s_incl_checks": round(time.time() - t0, 1)}}
        dump(m, P.res() / "checks" / f"{name}.json")
        dump(m, out / "metrics.json")  # written last = run complete (the model is saved by train())
        budget.note("full", time.time() - t0)
        a = res["accuracy"]
        log(f"step 2: {name}: none {a['none']:.3f} match {a['match']:.3f} opp {a['opposite']:.3f} neutral "
            f"{a['neutral']:.3f} gap {res['gap_none_minus_opposite']:.3f}; train {stats['wall_time_s'] / 60:.1f} min")
        for k, c in res["checks"].items():
            log(f"step 2: {name} CHECK {k}: {'PASS' if c['pass'] else 'FAIL'} value={c['value']:.4f} ({c['rule']})")
        return m
    m = attempt(name, go)
    refresh_outputs()
    return m


def failed(ms):
    return [f"{m['run']} {k} = {c['value']:.4f} ({c['rule']})" for m in ms for k, c in m["checks"].items()
            if not c["pass"]]


def step2(cfg, budget, enforce_gate):
    learned_path = P.res() / "learned.json"
    learned = load(learned_path) if learned_path.exists() else {
        "rule": "seed learned = shortcut model gap acc(none) - acc(opposite) >= 0.20 and acc(none) >= 0.80, clean twin "
                "|gap| <= 0.05 and acc(none) >= 0.80 (src/checks.py run_checks, configs/pilot.yaml). Seed 0 not "
                "learned = STOP; seed 1 or 2 not learned = left out of the fix test (YELP_PLAN.md)", "seeds": {}}
    seeds = [0] if DRY else list(SEEDS)
    for s in seeds:
        ms = [train_full(cfg, k, s, budget) for k in KINDS]
        fails = failed(ms)
        learned["seeds"][str(s)] = {"learned": not fails, "failed_checks": fails,
                                    "gap_shortcut": ms[0]["gap_none_minus_opposite"],
                                    "gap_clean": ms[1]["gap_none_minus_opposite"]}
        if s == 0 and fails:
            if DRY and not enforce_gate:
                learned["seeds"]["0"]["dry_run_gate_ignored"] = True
                log(f"DRY RUN: seed 0 not learned ({'; '.join(fails)}) -> gate NOT enforced in the dry run")
            else:
                learned.update(fix_test_seeds=[], env=env())
                dump(learned, learned_path)
                raise StopRule("seed 0 not learned: " + "; ".join(fails))
        elif fails:
            log(f"step 2: seed {s} not learned ({'; '.join(fails)}) -> left out of the fix test, flagged")
        else:
            log(f"step 2: seed {s} learned")
        learned["fix_test_seeds"] = [int(k) for k, v in learned["seeds"].items()
                                     if v["learned"] or v.get("dry_run_gate_ignored")]
        learned["env"] = env()
        dump(learned, learned_path)
    refresh_outputs()
    return learned["fix_test_seeds"]


# ---------------------------------------------------------------- step 3: the fix test
def retrain_log_dir(arm, name_s):
    """train()'s per-run files (config.json, train_log.txt, train_stats.json; no weights). Their content is also in the
    result files, so on Kaggle they go to /tmp: the notebook output stays at a few dozen files."""
    return (Path("/tmp/yelp_retrain_logs") if KAGGLE else P.runs() / PREFIX) / arm / name_s


def correct_counts(acc, n):
    return {v: round(a * n) for v, a in acc.items()}


def step3_seed(cfg, s, budget, tok, tests):
    from transformers import GPT2LMHeadModel
    from retrain import frozen_check
    from swap import evaluate
    from train import train
    start = P.runs() / f"{PREFIX}_shortcut_seed{s}" / "model"
    if not (start / "config.json").exists():
        raise Unexpected(f"start model {start.relative_to(P.out)} missing (attach the previous output with its runs/)")
    sets = [x for x in SETS if not DRY or x[0] in DRY_SET_NAMES]
    plan = [(arm, n, b, "clean") for n, b in sets for arm in ARMS] + [("retrain", CONTROL[0], CONTROL[1], CONTROL[2])]
    paths = {arm: P.res() / arm / f"seed{s}.json" for arm in ARMS}
    files = {arm: load(p) if p.exists() else None for arm, p in paths.items()}
    todo = [x for x in plan if not (files[x[0]] and x[1] in files[x[0]]["runs"])]
    if not todo:
        log(f"step 3: seed {s} already done")
        return
    init = GPT2LMHeadModel.from_pretrained(start)
    init_sd = {k: v.detach().clone() for k, v in init.state_dict().items()}
    checks = load(P.res() / "checks" / f"{PREFIX}_shortcut_seed{s}.json")
    n = len(tests["none"])
    for arm in ARMS:
        if files[arm] and files[arm]["baseline"]:
            continue
        other = next((f for f in files.values() if f and f["baseline"]), None)
        if other:  # one baseline per seed, shared by both arms
            base = other["baseline"]
        else:
            budget.check("base", f"seed {s} baseline")

            def go():
                t0 = time.time()
                m = init.to("cuda").eval()
                b = evaluate(m, tok, tests, cfg)
                init.to("cpu")
                diff = {v: correct_counts(b["accuracy"], n)[v] - correct_counts(checks["accuracy"], n)[v]
                        for v in b["accuracy"]}
                b.update({"start_model_vs_checks_file_correct_reviews_diff": diff,
                          "env": {**env(), "wall_time_s": round(time.time() - t0, 1)}})
                if max(abs(x) for x in diff.values()) > MAX_BASELINE_DIFF:
                    raise Unexpected(f"start model yelp_shortcut_seed{s} gives other numbers than its checks file "
                                     f"(correct reviews differ by {diff}); stop and ask")
                return b
            base = attempt(f"seed {s} baseline", go)
            free_gpu()
        files[arm] = files[arm] or {
            "seed": s, "start_model": f"{PREFIX}_shortcut_seed{s}", "arm": ARM_LABEL[arm],
            "train_file": f"data_yelp/{ARMS[arm]}", "data": "data_yelp",
            "note": "Phase 3 settings: only transformer.h.<b>.* of the chosen blocks trained; wte, wpe, ln_f, lm_head "
                    "and other blocks frozen (asserted bit-identical after training). 1 epoch, pilot settings. No "
                    "weights saved." + (" Control ctrl_shortcut:all trains on data_yelp/train_shortcut.jsonl."
                                        if arm == "retrain" else ""),
            "baseline": {}, "runs": {}}
        files[arm]["baseline"] = base
        dump(files[arm], paths[arm])
    G = files["retrain"]["baseline"]["gap_none_minus_opposite"]
    log(f"step 3: seed {s}: start gap G = {G:.3f}, acc(none) {files['retrain']['baseline']['accuracy']['none']:.3f}; "
        f"{len(todo)} runs to do")
    for arm, name, blocks, kind in todo:
        label = f"seed {s} {ARM_LABEL[arm]} {name}"
        budget.check("retrain", label)
        train_file = ARMS[arm] if kind == "clean" else cfg["train_files"]["shortcut"]

        def go():
            t0 = time.time()
            cfg1 = {**cfg, "epochs": 1, "train_files": {**cfg["train_files"], "clean": ARMS[arm]}}
            name_s = f"retrain_seed{s}_{name.replace(':', '_')}"
            model, _, stats = train(name_s, kind, s, cfg1, retrain_log_dir(arm, name_s), save=False,
                                    log=lambda m: print(m, flush=True), init_from=start, trainable_blocks=blocks,
                                    data_dir=P.data, **run_kw())
            n_changed = frozen_check(model, init_sd, blocks)
            r = evaluate(model, tok, tests, cfg)
            peak_total = torch.cuda.max_memory_allocated() / 2**30
            del model
            free_gpu()
            r.update({"blocks": blocks, "train_data": f"data_yelp/{train_file}", "seed": s,
                      "config": {"lr": cfg1["lr"], "epochs": 1, "batch_size": cfg1["batch_size"],
                                 "grad_accum": cfg1["grad_accum"], "warmup_frac": cfg1["warmup_frac"],
                                 "weight_decay": cfg1["weight_decay"], "precision": cfg1["precision"],
                                 "n_examples": None, "max_steps": run_kw().get("max_steps")},
                      "train": stats, "peak_vram_allocated_gib_incl_eval": peak_total,
                      "frozen_params_unchanged": True, "n_trained_tensors_changed": n_changed,
                      "wall_time_s_incl_eval": time.time() - t0, "env": {**env(), "wall_time_s": round(time.time() - t0, 1)}})
            return r
        r = attempt(label, go)
        files[arm]["runs"][name] = r
        dump(files[arm], paths[arm])
        budget.note("retrain", r["wall_time_s_incl_eval"])
        a = r["accuracy"]
        frac = f"{(G - r['gap_none_minus_opposite']) / G:.2f}" if G else "n/a (G = 0)"
        log(f"step 3: {label}: none {a['none']:.3f} opp {a['opposite']:.3f} gap {r['gap_none_minus_opposite']:.3f} "
            f"(G {G:.3f}, fraction removed {frac}) {r['wall_time_s_incl_eval'] / 60:.1f} min")
        refresh_outputs()
    del init, init_sd
    free_gpu()


def step3(cfg, budget, seeds):
    from common import get_tokenizer, load_tests
    tok = get_tokenizer()
    tests = load_tests(P.data)
    for s in seeds:
        step3_seed(cfg, s, budget, tok, tests)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="CPU only, tiny settings, never the GPU")
    ap.add_argument("--out", help="output folder (default: the code folder)")
    ap.add_argument("--zip", help="output zip (default: <out>/yelp_results.zip)")
    ap.add_argument("--time-budget-h", type=float, default=11.0)
    ap.add_argument("--enforce-gate", action="store_true", help="dry run only: apply the seed-0 stop as on Kaggle")
    ap.add_argument("--refresh-only", action="store_true", help="only rewrite RESULTS_YELP.md and the zip; no runs")
    args = ap.parse_args()
    P.out = Path(args.out).resolve() if args.out else CODE
    P.zip = Path(args.zip).resolve() if args.zip else P.out / "yelp_results.zip"
    P.res().mkdir(parents=True, exist_ok=True)
    if args.refresh_only:
        refresh_outputs()
        print(f"refreshed {P.out / 'RESULTS_YELP.md'} and {P.zip}")
        return
    ref = CODE / "results" / PREFIX / "imdb_reference.json"
    if ref.exists() and not (P.res() / "imdb_reference.json").exists():
        shutil.copy2(ref, P.res() / "imdb_reference.json")
    budget = Budget(args.time_budget_h)
    t_start = time.time()
    if (P.res() / "status.json").exists():
        STATUS.update(load(P.res() / "status.json"))
    session = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(), "time_budget_h": args.time_budget_h}
    STATUS.setdefault("sessions", []).append(session)
    status_save(state="running", reason="", exit_code=None)
    log(f"=== run_yelp start{' (DRY RUN, CPU)' if DRY else ''}: out {P.out}, zip {P.zip}, budget {args.time_budget_h} h")
    code, state, reason = 0, "done", "all steps finished"
    try:
        if DRY:
            install_cpu_shim()
        from common import load_cfg
        cfg = load_cfg(CODE / "configs" / "pilot.yaml")
        step0_checks(cfg)
        session["env"] = env()
        status_save()
        step1(cfg, budget)
        seeds = step2(cfg, budget, args.enforce_gate)
        flagged = sorted(set(([0] if DRY else list(SEEDS))) - set(seeds))
        log(f"step 3: fix test on seeds {seeds}" + (f" (left out, not learned: {flagged})" if flagged else ""))
        step3(cfg, budget, seeds)
    except StopRule as e:
        code, state, reason = 3, "stopped_by_rule", str(e)
    except StopBudget as e:
        code, state, reason = 4, "stopped_time_budget", str(e)
    except Unexpected as e:
        code, state, reason = 1, "stopped_unexpected", str(e)
    except BaseException as e:  # noqa: BLE001
        code, state = (130, "interrupted") if isinstance(e, KeyboardInterrupt) else (1, "crashed")
        reason = f"{type(e).__name__}: {e}"
        log("CRASH (driver):\n" + traceback.format_exc())
    log(f"=== {state}: {reason}")
    session.update({"end": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_time_s": round(time.time() - t_start, 1),
                    "exit_code": code, "state": state})
    status_save(state=state, reason=reason, exit_code=code)
    refresh_outputs()  # step 4: RESULTS_YELP.md + zip, whatever happened
    log(f"=== run_yelp end (exit {code}), {(time.time() - t_start) / 3600:.2f} h; report {P.out / 'RESULTS_YELP.md'}, "
        f"zip {P.zip}")
    sys.exit(code)


if __name__ == "__main__":
    main()
