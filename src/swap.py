"""Phase 2: removal map. Swap transformer blocks between saved models and re-measure the shortcut.

Usage: python src/swap.py [--seeds 0,1,2]
Writes results/swap/seed<s>.json (resumable: finished cases are skipped).
Forward passes only. Embeddings, ln_f and lm_head always stay from the model being edited.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import time

import torch
from transformers import GPT2LMHeadModel

from common import DATA, RESULTS, RUNS, VERSIONS, accuracy, get_tokenizer, load_cfg, load_jsonl, score

N_BLOCKS = 12
BLOCK_SETS = [[b] for b in range(N_BLOCKS)] + [[10, 11], [9, 10, 11]]


def other_seed(s, seeds):
    return seeds[(seeds.index(s) + 1) % len(seeds)]


def cases_for(s, seeds):
    o = other_seed(s, seeds)
    return {
        "a_shortcut_from_clean_same": (f"shortcut_seed{s}", f"clean_seed{s}"),
        "b_clean_from_shortcut_same": (f"clean_seed{s}", f"shortcut_seed{s}"),
        "ci_shortcut_from_clean_other": (f"shortcut_seed{s}", f"clean_seed{o}"),
        "cii_shortcut_from_shortcut_other": (f"shortcut_seed{s}", f"shortcut_seed{o}"),
    }


def block_keys(sd, blocks):
    pre = tuple(f"transformer.h.{b}." for b in blocks)
    return [k for k in sd if k.startswith(pre)]


def key(blocks):
    return "-".join(map(str, blocks))


def evaluate(model, tok, tests, cfg):
    acc, mean_d = {}, {}
    for v, rows in tests.items():
        d = score(model, tok, rows, cfg["max_len"], "cuda", cfg["eval_batch_size"])
        acc[v] = accuracy(d, rows)
        mean_d[v] = float(d.mean())
    return {"accuracy": acc, "mean_d": mean_d, "gap_none_minus_opposite": acc["none"] - acc["opposite"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds")
    args = ap.parse_args()
    cfg = load_cfg()
    seeds = cfg["seeds"]
    run_seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else seeds
    tok = get_tokenizer()
    tests = {v: load_jsonl(DATA / f"test_{v}.jsonl") for v in VERSIONS}
    out_dir = RESULTS / "swap"
    out_dir.mkdir(parents=True, exist_ok=True)

    # all 6 state dicts on CPU (~0.5 GB each)
    names = [f"{k}_seed{s}" for k in ("shortcut", "clean") for s in seeds]
    sds = {}
    for n in names:
        m = GPT2LMHeadModel.from_pretrained(RUNS / n / "model")
        sds[n] = {k: v.detach().clone() for k, v in m.state_dict().items()}
        del m
    model = GPT2LMHeadModel.from_pretrained(RUNS / names[0] / "model").to("cuda").eval()

    for s in run_seeds:
        path = out_dir / f"seed{s}.json"
        res = json.load(open(path)) if path.exists() else {
            "seed": s, "other_seed": other_seed(s, seeds), "block_sets": [key(b) for b in BLOCK_SETS],
            "note": "editor keeps its embeddings, ln_f and lm_head; only transformer.h.<b>.* comes from the donor",
            "baseline": {}, "cases": {}}
        for n in (f"shortcut_seed{s}", f"clean_seed{s}"):
            if n not in res["baseline"]:
                model.load_state_dict(sds[n])
                res["baseline"][n] = evaluate(model, tok, tests, cfg)
                json.dump(res, open(path, "w"), indent=2)
        for case, (editor, donor) in cases_for(s, seeds).items():
            res["cases"].setdefault(case, {"editor": editor, "donor": donor, "by_blocks": {}})
            for blocks in BLOCK_SETS:
                if key(blocks) in res["cases"][case]["by_blocks"]:
                    continue
                t0 = time.time()
                sd = dict(sds[editor])
                ks = block_keys(sd, blocks)
                assert len(ks) == 12 * len(blocks), (len(ks), blocks)  # 12 tensors per GPT-2 block
                for k in ks:
                    sd[k] = sds[donor][k]
                model.load_state_dict(sd)
                r = evaluate(model, tok, tests, cfg)
                r["n_tensors_swapped"] = len(ks)
                res["cases"][case]["by_blocks"][key(blocks)] = r
                json.dump(res, open(path, "w"), indent=2)
                a = r["accuracy"]
                print(f"seed{s} {case} blocks {key(blocks)}: none {a['none']:.3f} match {a['match']:.3f} "
                      f"opp {a['opposite']:.3f} neutral {a['neutral']:.3f} gap {r['gap_none_minus_opposite']:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print("swap done")


if __name__ == "__main__":
    main()
