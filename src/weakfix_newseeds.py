"""Phase L7 (PROJECT_SPEC.md "Phase L7"): does the repair-speed pattern hold on new seeds? Phase L3's weak fix on two new
full fine-tuning shortcut models (seeds 3 and 4; seed 5 replaces a seed that fails its training checks, once).

The weak fix is Phase L3's, unchanged: src/weakfix.py's run_fix() and setup() are imported and called as they are (rank-4
adapter on one block, 49,152 parameters, lr 5e-4, 1-epoch schedule of 625 steps, everything else asserted bit-identical,
no weights saved). B = 100 is read from results/weakfix/full_ft/budget.json (Phase L3), not recalibrated.

Usage:
  python src/weakfix_newseeds.py --check --seeds 3,4   # start-model checks -> results/weakfix_newseeds/start_checks.json
  python src/weakfix_newseeds.py --smoke --seed 3      # 1 run, block 11, 200 examples, 20 steps (code check only)
  python src/weakfix_newseeds.py --seed 3              # 12 single blocks + the control -> results/weakfix_newseeds/full_ft/seed3.json
Resumable: finished runs are skipped. Exit code 2 = a start-model check failed (STOP and tell Sonu).
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import sys
import time

from transformers import GPT2LMHeadModel

from common import DATA, RESULTS, RUNS, VERSIONS, env_info, get_tokenizer, load_cfg, load_jsonl, run_name
from swap import evaluate
from weakfix import N_BLOCKS, eval_steps_for, free_gpu, git_commit, run_fix, setup, sha256
from weakfix import out_dir as l3_out_dir

COND = "full_ft"           # Phase L7 tests full fine-tuning only
NEW_SEEDS = (3, 4)         # the two new seeds
REPLACEMENT_SEED = 5       # replaces a seed that fails its training checks, once
ALL_SEEDS = (*NEW_SEEDS, REPLACEMENT_SEED)
B_REGISTERED = 100         # PROJECT_SPEC.md Phase L7: B = 100, reused from Phase L3
CTRL_BLOCK = 11            # the Phase L3 control block for full FT (its localized block)
OUT = RESULTS / "weakfix_newseeds"


def seed_dir():
    return OUT / COND


def start_model(s):
    return RUNS / run_name("shortcut", s, None) / "model"


def checks_file(s):
    return RESULTS / "checks" / f"{run_name('shortcut', s, None)}.json"


def reading_B():
    """B from Phase L3's budget file, asserted equal to the registered 100; eval steps as in Phase L3."""
    B = json.load(open(l3_out_dir(COND) / "budget.json"))["B"]
    assert B == B_REGISTERED, f"Phase L3 budget.json says B = {B}, Phase L7 registered B = {B_REGISTERED}"
    steps = eval_steps_for(B)
    assert steps == [50, 100, 200, 625], steps
    return B, steps


def counts(acc, tests):
    """Accuracies as correct-review counts (each test version has a fixed size), so equality is exact."""
    return {v: round(acc[v] * len(tests[v])) for v in tests}


def check(seeds):
    """Start-model checks (Phase L7): record the sha256 of each new model; its accuracies on the 4 test versions must
    equal its checks file (as correct counts). Exit 2 on any mismatch."""
    assert all(s in ALL_SEEDS for s in seeds), seeds
    cfg = load_cfg()
    tok = get_tokenizer()
    tests = {v: load_jsonl(DATA / f"test_{v}.jsonl") for v in VERSIONS}
    p = OUT / "start_checks.json"
    res = json.load(open(p)) if p.exists() else {"condition": COND, "seeds": {}}
    res["env"], res["git_commit"] = env_info(), git_commit()
    ok = True
    t0 = time.time()
    for s in seeds:
        rec = {"start_model": run_name("shortcut", s, None),
               "sha256": sha256(start_model(s) / "model.safetensors")}
        model = GPT2LMHeadModel.from_pretrained(start_model(s)).to("cuda").eval()
        e = evaluate(model, tok, tests, cfg)
        del model
        free_gpu()
        chk = json.load(open(checks_file(s)))
        want = chk["accuracy"]
        rec.update(baseline=e, checks_accuracy=want, checks_all_pass=chk["all_pass"],
                   accuracy_matches_checks=counts(e["accuracy"], tests) == counts(want, tests))
        ok &= rec["accuracy_matches_checks"] and rec["checks_all_pass"]
        res["seeds"][str(s)] = rec
        print(f"[check seed{s}] accuracy == checks file: {rec['accuracy_matches_checks']}; training checks pass: "
              f"{rec['checks_all_pass']}; sha256 {rec['sha256'][:12]}...", flush=True)
    res["checked_seeds"] = sorted(int(k) for k in res["seeds"])
    res["all_pass"] = bool(ok) and all(r["accuracy_matches_checks"] and r["checks_all_pass"]
                                       for r in res["seeds"].values())
    res["wall_time_s"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(p, "w"), indent=2)
    if not res["all_pass"]:
        print("START-MODEL CHECK FAILED (Phase L7): STOP and tell Sonu", flush=True)
        sys.exit(2)


def baseline(s):
    c = json.load(open(OUT / "start_checks.json"))
    assert c["all_pass"] and str(s) in c["seeds"], f"start-model checks did not pass for seed {s}"
    rec = c["seeds"][str(s)]
    h = sha256(start_model(s) / "model.safetensors")
    assert h == rec["sha256"], f"seed {s}: the model changed since its start-model check"
    return rec["baseline"], h


def main_runs(s):
    """12 single blocks on clean data + the control (block 11, shortcut data) -> full_ft/seed<s>.json."""
    assert s in ALL_SEEDS, s
    B, steps = reading_B()
    _, cfg1, tok, tests = setup(COND)
    base, h = baseline(s)
    path = seed_dir() / f"seed{s}.json"
    res = json.load(open(path)) if path.exists() else {
        "phase": "L7", "condition": COND, "seed": s, "start_model": run_name("shortcut", s, None),
        "start_model_sha256": h, "baseline": base, "B": B, "eval_steps": steps, "control_block": CTRL_BLOCK,
        "git_commit_at_start": git_commit(),
        "note": "Phase L7: Phase L3's weak fix (src/weakfix.py run_fix, unchanged): rank-4 LoRA adapter (49,152 params) "
                "on one block of the new start model as it is; lr 5e-4, 1-epoch schedule (625 steps); everything else "
                "asserted bit-identical; eval on none/opposite at eval_steps. B = 100 from Phase L3. No weights saved.",
        "runs": {}}
    assert res["B"] == B and res["eval_steps"] == steps and res["start_model_sha256"] == h
    seed_dir().mkdir(parents=True, exist_ok=True)
    todo = [(str(b), [b], "clean") for b in range(N_BLOCKS)] + [(f"ctrl_shortcut:{CTRL_BLOCK}", [CTRL_BLOCK], "shortcut")]
    G = base["gap_none_minus_opposite"]
    for name, blocks, data in todo:
        if name in res["runs"]:
            continue
        rn = f"l7_weakfix_seed{s}_{name.replace(':', '_')}"
        res["runs"][name] = run_fix(COND, s, blocks, data, steps, rn, tok, tests, cfg1)
        json.dump(res, open(path, "w"), indent=2)
        r = res["runs"][name]
        print(f"[L7 seed{s}] {name} done: " + ", ".join(
            f"step {st} removes {(G - r['evals'][str(st)]['gap_none_minus_opposite']) / G:.3f}" for st in steps)
            + f"; wall {r['wall_time_s_incl_eval']:.0f}s", flush=True)
    print(f"L7 seed{s} done", flush=True)


def smoke(s):
    """1 run: block 11, 200 examples, 20 steps, eval at steps 10 and 20 (code check only, not a result)."""
    _, cfg1, tok, tests = setup(COND)
    r = run_fix(COND, s, [CTRL_BLOCK], "clean", [10, 20], f"l7_smoke_seed{s}_{CTRL_BLOCK}", tok, tests, cfg1,
                n_examples=200, max_steps=20)
    (OUT / "smoke").mkdir(parents=True, exist_ok=True)
    json.dump({"seed": s, "block": CTRL_BLOCK, "run": r, "env": env_info(), "git_commit": git_commit()},
              open(OUT / "smoke" / f"seed{s}.json", "w"), indent=2)
    print(f"SMOKE L7 seed{s}: loss {r['train']['loss_first10']:.4f} -> {r['train']['loss_last10']:.4f}, "
          f"{r['n_trainable_params']:,} trainable params, frozen weights unchanged, adapter moved", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=ALL_SEEDS)
    ap.add_argument("--seeds", help="with --check: comma-separated seeds")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check([int(x) for x in a.seeds.split(",")])
    assert a.seed is not None, "--seed is needed"
    if a.smoke:
        return smoke(a.seed)
    return main_runs(a.seed)


if __name__ == "__main__":
    main()
