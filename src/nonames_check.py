"""Phase 3c start-model checks (PROJECT_SPEC.md "Phase 3c", as in Phase L3): the full-FT shortcut models have the sha256
of Sonu's zip; every start model's number of correct reviews on the 4 test versions equals its checks file.
-> results/retrain_nonames/start_checks.json (git commit, GPU, time). Exit 2 on any mismatch: STOP and tell Sonu.

Usage: python src/nonames_check.py
"""
import json
import subprocess
import sys
import time

import torch
from transformers import GPT2LMHeadModel

from common import DATA, RESULTS, ROOT, VERSIONS, env_info, get_tokenizer, load_cfg, load_jsonl, run_name
from swap import evaluate
from weakfix import CONDITIONS, FULL_FT_SHA256, SEEDS, checks_file, sha256, start_model

OUT = RESULTS / "retrain_nonames" / "start_checks.json"


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or None
    except OSError:
        return None


def n_correct(acc, n):
    """Accuracy as a count of correct reviews (files store float32 or float64 values of the same count)."""
    return round(acc * n)


def main():
    t0 = time.time()
    cfg, tok = load_cfg(), get_tokenizer()
    tests = {v: load_jsonl(DATA / f"test_{v}.jsonl") for v in VERSIONS}
    res = {"phase": "3c", "env": env_info(), "git_commit": git_commit(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "rule": "full-FT sha256 == Sonu's zip; correct reviews per test version == the checks file", "models": {}}
    ok = True
    for cond in CONDITIONS:
        for s in SEEDS:
            name = run_name("shortcut", s, CONDITIONS[cond])
            rec = {"condition": cond, "seed": s}
            if cond == "full_ft":
                h = sha256(start_model(cond, s) / "model.safetensors")
                rec.update(sha256=h, sha256_expected=FULL_FT_SHA256[s], sha256_match=h == FULL_FT_SHA256[s])
                ok &= rec["sha256_match"]
            model = GPT2LMHeadModel.from_pretrained(start_model(cond, s)).to("cuda").eval()
            e = evaluate(model, tok, tests, cfg)
            del model
            torch.cuda.empty_cache()
            want = json.load(open(checks_file(cond, s)))["accuracy"]
            got_n = {v: n_correct(e["accuracy"][v], len(tests[v])) for v in VERSIONS}
            want_n = {v: n_correct(want[v], len(tests[v])) for v in VERSIONS}
            rec.update(accuracy=e["accuracy"], checks_accuracy=want, correct=got_n, checks_correct=want_n,
                       correct_matches_checks=got_n == want_n, checks_file=str(checks_file(cond, s).relative_to(ROOT)))
            ok &= rec["correct_matches_checks"]
            res["models"][name] = rec
            print(f"[3c check] {name}: correct reviews == checks file: {rec['correct_matches_checks']}"
                  + (f"; sha256 == zip: {rec['sha256_match']}" if cond == "full_ft" else ""), flush=True)
    res["all_pass"], res["wall_time_s"] = bool(ok), time.time() - t0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    if not ok:
        print("START-MODEL CHECK FAILED: STOP and tell Sonu (no Phase 3c run)", flush=True)
        sys.exit(2)
    print(f"[3c check] all {len(res['models'])} start models pass -> {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
