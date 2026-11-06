"""Phase L5 (PROJECT_SPEC.md "Phase L5", with its 6 Nov amendment): the c-ii swap between the LoRA r64 models of the 3x3
grid (init seed i, data seed j), to separate LoRA's random init from the training path. 8 of the 9 models: (2, 0)
collapsed in training and is left out. Swap as in Phase 2 (src/swap.py evaluate, block_keys): only transformer.h.<b>.*
comes from the donor. Forward passes only. Results: results/seedswap/.

Usage: python src/seedswap.py          # all 56 editor/donor pairs x 5 block sets -> swaps.json (resumable)
       python src/seedswap.py --overlap   # adapter overlap per pair (no GPU) -> overlap.json
"""
import argparse
import json
import time

import torch
from transformers import GPT2LMHeadModel

from adapter_overlap import MODULES, lora_factors, lowrank_dirs, overlap
from common import DATA, RESULTS, RUNS, VERSIONS, get_tokenizer, load_cfg, load_jsonl, run_env, run_name
from swap import block_keys, evaluate, key

IDS = (0, 1, 2)
LEFT_OUT = (2, 0)  # init 2, data 0 collapsed in training: in no swap, reported openly (Phase L5 amendment)
GRID = [(i, j) for i in IDS for j in IDS if (i, j) != LEFT_OUT]
RANK, ALPHA = 64, 128  # Phase L1 amendment; train_lora.py asserts the rest
BLOCK_SETS = [[9], [10], [11], [10, 11], [9, 10, 11]]
PRIMARY = "9-10-11"
OUT = RESULTS / "seedswap"


def model_name(i, j):
    return run_name("shortcut", j, f"lora_r64_init{i}")


def pair_key(edit, donor):
    return f"{edit[0]}{edit[1]}<-{donor[0]}{donor[1]}"


def cell_pairs():
    """{cell: [(editor, donor)]} over the 8 models, each model = (init seed, data seed). A: same init, different data
    (14 pairs); B: different init, same data (14); C: both different (28)."""
    out = {"A": [], "B": [], "C": []}
    for e in GRID:
        for d in GRID:
            same_init, same_data = e[0] == d[0], e[1] == d[1]
            if same_init != same_data:
                out["A" if same_init else "B"].append((e, d))
            elif not same_init:
                out["C"].append((e, d))
    return out


def replicate_pairs():
    """{cell: [(editor, donor)]} of the reading rule, replicates k = 0, 1, 2: editor (k, k); indices mod 3. Amendment:
    A's k = 2 donor is (2, 1), since (2, 0) is left out; C's donors are (init k+2, data k+1), new models, because
    (k, k) are the main models and (k+1, k+1) would repeat the main c-ii pairs."""
    n = len(IDS)
    a = [((k, k), (k, (k + 1) % n)) for k in IDS]
    a[2] = ((2, 2), (2, 1))
    return {"A": a, "B": [((k, k), ((k + 1) % n, k)) for k in IDS],
            "C": [((k, k), ((k + 2) % n, (k + 1) % n)) for k in IDS]}


def original_pairs():
    """The first registered cell C pairs, the main c-ii pairs: reported next to C, no verdict. Part of cell_pairs()."""
    n = len(IDS)
    return [((k, k), ((k + 1) % n, (k + 1) % n)) for k in IDS]


def run_swaps(sds, model, tok, tests, cfg, path, evaluate_fn=evaluate):
    """sds[(i, j)] = state dict. Every (cell, pair, block set) once; saved after each evaluation, so a restart resumes."""
    res = json.load(open(path)) if path.exists() else {
        "note": "editor keeps its embeddings, ln_f and lm_head; only transformer.h.<b>.* comes from the donor",
        "block_sets": [key(b) for b in BLOCK_SETS], "baseline": {}, "cells": {}}
    for m, sd in sds.items():
        name = model_name(*m)
        if name not in res["baseline"]:
            model.load_state_dict(sd)
            res["baseline"][name] = evaluate_fn(model, tok, tests, cfg)
            json.dump(res, open(path, "w"), indent=2)
    for cell, pairs in cell_pairs().items():
        for edit, donor in pairs:
            entry = res["cells"].setdefault(cell, {}).setdefault(pair_key(edit, donor), {})
            for blocks in BLOCK_SETS:
                if key(blocks) in entry:
                    continue
                t0 = time.time()
                sd = dict(sds[edit])
                ks = block_keys(sd, blocks)
                assert len(ks) == 12 * len(blocks), (len(ks), blocks)  # 12 tensors per GPT-2 block
                for k in ks:
                    sd[k] = sds[donor][k]
                model.load_state_dict(sd)
                r = evaluate_fn(model, tok, tests, cfg)
                r.update({"n_tensors_swapped": len(ks), "wall_time_s": round(time.time() - t0, 1)})
                entry[key(blocks)] = r
                json.dump(res, open(path, "w"), indent=2)
                print(f"{cell} {pair_key(edit, donor)} blocks {key(blocks)}: gap {r['gap_none_minus_opposite']:.3f} "
                      f"({r['wall_time_s']:.0f}s)", flush=True)
    return res


def pair_overlaps(factors, edit, donor):
    """Adapter overlap of two models over all 12 blocks: {block: {side: mean over the 4 matrices}}, whole update, k = r
    (results/adapter_overlap/NOTES.md, view i; view ii needs clean twins, which Phase L5 does not train)."""
    out = {}
    for b in sorted({bm[0] for bm in factors[edit]}):
        vals = {"input": [], "output": []}
        for m in MODULES:
            da = lowrank_dirs(*factors[edit][(b, m)], (RANK,))[RANK]
            db = lowrank_dirs(*factors[donor][(b, m)], (RANK,))[RANK]
            for si, side in enumerate(("input", "output")):
                vals[side].append(overlap(da[si], db[si]))
        out[str(b)] = {s: sum(v) / len(v) for s, v in vals.items()}
    return out


def overlaps(factors):
    """{cell: {pair: pair_overlaps}}; a pair is counted once (i, j) with (i', j') symmetric, editor listed first."""
    res = {}
    for cell, pairs in cell_pairs().items():
        seen = set()
        for edit, donor in pairs:
            if (donor, edit) in seen:
                continue
            seen.add((edit, donor))
            res.setdefault(cell, {})[pair_key(edit, donor)] = pair_overlaps(factors, edit, donor)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.overlap:
        torch.set_num_threads(min(8, torch.get_num_threads()))
        factors = {m: lora_factors(RUNS / model_name(*m) / "adapter" / "adapter_model.safetensors", RANK, ALPHA)
                   for m in GRID}
        t0 = time.time()
        res = {"method": "results/adapter_overlap/NOTES.md, view i (whole update), k = 64", "modules": MODULES,
               "cells": overlaps(factors)}
        res.update({"wall_time_s": round(time.time() - t0, 1), "env": run_env()})
        json.dump(res, open(OUT / "overlap.json", "w"), indent=1)
        print(f"wrote {OUT / 'overlap.json'}")
        return
    cfg, tok = load_cfg(), get_tokenizer()
    tests = {v: load_jsonl(DATA / f"test_{v}.jsonl") for v in VERSIONS}
    sds = {}  # the 8 state dicts on CPU (~0.5 GB each)
    for m in GRID:
        g = GPT2LMHeadModel.from_pretrained(RUNS / model_name(*m) / "model")
        sds[m] = {k: v.detach().clone() for k, v in g.state_dict().items()}
        del g
    model = GPT2LMHeadModel.from_pretrained(RUNS / model_name(*GRID[0]) / "model").to("cuda").eval()
    run_swaps(sds, model, tok, tests, cfg, OUT / "swaps.json")
    print("seedswap done")


if __name__ == "__main__":
    main()
