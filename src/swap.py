"""Phase 2: removal map. Swap transformer blocks between saved models and re-measure the shortcut.

Usage: python src/swap.py [--seeds 0,1,2]            # Phase 2  -> results/swap/seed<s>.json
       python src/swap.py --phase2b [--seeds 0,1,2]  # Phase 2b -> results/swap2/seed<s>.json
       python src/swap.py --phase2b --smoke          # 2 cases, seed 0 -> results/smoke/swap2/
       python src/swap.py --prefix lora_r4 [...]     # lora_r4_{shortcut,clean}_seed<s> -> results/lora_r4/swap/ etc.
Resumable: finished cases are skipped. Forward passes only. Embeddings, ln_f and lm_head stay from the
model being edited, except Phase 2b "emb_names", which swaps only the name-token rows of wte.
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

from common import (DATA, RUNS, VERSIONS, accuracy, get_tokenizer, label_ids, load_cfg, load_jsonl, prefix_arg,
                    results_dir, run_name, score)

N_BLOCKS = 12
BLOCK_SETS = [[b] for b in range(N_BLOCKS)] + [[10, 11], [9, 10, 11]]


def other_seed(s, seeds):
    return seeds[(seeds.index(s) + 1) % len(seeds)]


def cases_for(s, seeds, prefix=None):
    o = other_seed(s, seeds)
    sc, cl = (lambda k: run_name("shortcut", k, prefix)), (lambda k: run_name("clean", k, prefix))
    return {
        "a_shortcut_from_clean_same": (sc(s), cl(s)),
        "b_clean_from_shortcut_same": (cl(s), sc(s)),
        "ci_shortcut_from_clean_other": (sc(s), cl(o)),
        "cii_shortcut_from_shortcut_other": (sc(s), sc(o)),
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


# ---- Phase 2b (PROJECT_SPEC.md "Phase 2b: follow-up swaps")
PH2B_WINDOWS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [7, 8]]
PH2B_HALF_SETS = [[9], [10], [11], [10, 11], [9, 10, 11]]
HALF_PARTS = {"attn": ("ln_1.", "attn."), "mlp": ("ln_2.", "mlp.")}
EMB_CASES = ("a_shortcut_from_clean_same", "b_clean_from_shortcut_same")
HALF_CASES = ("a_shortcut_from_clean_same", "cii_shortcut_from_shortcut_other")


def spec_name(kind, blocks):
    if kind == "blocks":
        return "all12" if blocks == list(range(N_BLOCKS)) else key(blocks)
    if kind == "emb_names":
        return "emb_names" + (f"+{key(blocks)}" if blocks else "")
    return f"{kind}:{key(blocks)}"


def phase2b_specs(case):
    specs = [("blocks", b) for b in PH2B_WINDOWS]
    if case in EMB_CASES:
        specs += [("emb_names", []), ("emb_names", [9, 10, 11]), ("blocks", list(range(N_BLOCKS)))]
    if case in HALF_CASES:
        specs += [(h, b) for b in PH2B_HALF_SETS for h in ("attn", "mlp")]
    return specs


def half_keys(sd, blocks, half):
    pre = tuple(f"transformer.h.{b}.{p}" for b in blocks for p in HALF_PARTS[half])
    return [k for k in sd if k.startswith(pre)]


def edited_state(sds, editor, donor, kind, blocks, name_ids):
    """Editor's state dict with the given part taken from the donor. Returns (sd, swapped keys)."""
    sd = dict(sds[editor])
    if kind in HALF_PARTS:
        ks = half_keys(sd, blocks, kind)
        assert len(ks) == 6 * len(blocks), (len(ks), kind, blocks)  # 6 tensors per half-block
    else:
        ks = block_keys(sd, blocks)
        assert len(ks) == 12 * len(blocks), (len(ks), blocks)
    for k in ks:
        sd[k] = sds[donor][k]
    if kind == "emb_names":
        w = sds[editor]["transformer.wte.weight"].clone()
        w[name_ids] = sds[donor]["transformer.wte.weight"][name_ids]
        sd["transformer.wte.weight"] = w
        sd["lm_head.weight"] = w  # tied: both keys must carry the same matrix, or load order undoes the edit
    return sd, ks


def check_emb_edit(model, sds, editor, donor, name_ids, label_rows):
    """Assert only the name rows of wte changed (label rows untouched, still tied). Returns L2 size of the change."""
    wte = model.transformer.wte.weight.detach().cpu()
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr(), "wte/lm_head untied"
    assert torch.equal(wte[label_rows], sds[editor]["lm_head.weight"][label_rows]), "label rows changed"
    assert torch.equal(wte[name_ids], sds[donor]["transformer.wte.weight"][name_ids]), "name rows not swapped"
    other = torch.ones(wte.size(0), dtype=torch.bool)
    other[name_ids] = False
    assert torch.equal(wte[other], sds[editor]["transformer.wte.weight"][other]), "non-name rows changed"
    return float((sds[editor]["transformer.wte.weight"][name_ids]
                  - sds[donor]["transformer.wte.weight"][name_ids]).norm())


def run_phase2b(sds, model, tok, tests, cfg, seeds, run_seeds, out_dir, smoke=False, prefix=None):
    manifest = json.load(open(DATA / "manifest.json"))
    name_ids = sorted({i for n in ("A", "B", "N") for i in manifest["name_token_ids"][n]})
    pos_id, neg_id = label_ids(tok)
    label_rows = [pos_id, neg_id]
    assert not set(label_rows) & set(name_ids), "a label token is also a name token"
    for s in run_seeds:
        path = out_dir / f"seed{s}.json"
        res = json.load(open(path)) if path.exists() else {
            "seed": s, "other_seed": other_seed(s, seeds), "name_token_ids": name_ids,
            "label_token_ids": {"positive": pos_id, "negative": neg_id},
            "note": "Phase 2b. Editor keeps its embeddings, ln_f and lm_head, except emb_names (name rows of "
                    "wte only; wte is tied to lm_head; label rows asserted unchanged)",
            "baseline": {}, "cases": {}}
        if not smoke:
            for n in (run_name("shortcut", s, prefix), run_name("clean", s, prefix)):
                if n not in res["baseline"]:
                    model.load_state_dict(sds[n])
                    res["baseline"][n] = evaluate(model, tok, tests, cfg)
                    json.dump(res, open(path, "w"), indent=2)
        for case, (editor, donor) in cases_for(s, seeds, prefix).items():
            specs = phase2b_specs(case)
            if smoke:
                specs = {"b_clean_from_shortcut_same": [("emb_names", [])],
                         "a_shortcut_from_clean_same": [("attn", [9])]}.get(case, [])
            if not specs:
                continue
            res["cases"].setdefault(case, {"editor": editor, "donor": donor, "by_set": {}})
            for kind, blocks in specs:
                name = spec_name(kind, blocks)
                if name in res["cases"][case]["by_set"]:
                    continue
                t0 = time.time()
                sd, ks = edited_state(sds, editor, donor, kind, blocks, name_ids)
                model.load_state_dict(sd)
                extra = {}
                if kind == "emb_names":
                    extra = {"n_wte_rows_swapped": len(name_ids), "label_rows_unchanged": True,
                             "name_rows_l2_diff_editor_vs_donor":
                                 check_emb_edit(model, sds, editor, donor, name_ids, label_rows)}
                r = evaluate(model, tok, tests, cfg)
                r.update({"kind": kind, "blocks": blocks, "n_tensors_swapped": len(ks), **extra})
                res["cases"][case]["by_set"][name] = r
                json.dump(res, open(path, "w"), indent=2)
                a = r["accuracy"]
                print(f"seed{s} {case} {name}: none {a['none']:.3f} match {a['match']:.3f} "
                      f"opp {a['opposite']:.3f} neutral {a['neutral']:.3f} gap {r['gap_none_minus_opposite']:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print(f"phase2b done; peak VRAM allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds")
    ap.add_argument("--phase2b", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prefix", type=prefix_arg, help="run-name prefix, e.g. lora_r4 (default: none)")
    args = ap.parse_args()
    cfg = load_cfg()
    seeds = cfg["seeds"]
    run_seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else seeds
    tok = get_tokenizer()
    tests = {v: load_jsonl(DATA / f"test_{v}.jsonl") for v in VERSIONS}
    out_dir = results_dir("smoke/swap2" if args.smoke else "swap2" if args.phase2b else "swap", args.prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        run_seeds = run_seeds[:1]

    # all 6 state dicts on CPU (~0.5 GB each)
    names = [run_name(k, s, args.prefix) for k in ("shortcut", "clean") for s in seeds]
    sds = {}
    for n in names:
        m = GPT2LMHeadModel.from_pretrained(RUNS / n / "model")
        sds[n] = {k: v.detach().clone() for k, v in m.state_dict().items()}
        del m
    model = GPT2LMHeadModel.from_pretrained(RUNS / names[0] / "model").to("cuda").eval()
    if args.phase2b:
        run_phase2b(sds, model, tok, tests, cfg, seeds, run_seeds, out_dir, smoke=args.smoke, prefix=args.prefix)
        return

    for s in run_seeds:
        path = out_dir / f"seed{s}.json"
        res = json.load(open(path)) if path.exists() else {
            "seed": s, "other_seed": other_seed(s, seeds), "block_sets": [key(b) for b in BLOCK_SETS],
            "note": "editor keeps its embeddings, ln_f and lm_head; only transformer.h.<b>.* comes from the donor",
            "baseline": {}, "cases": {}}
        for n in (run_name("shortcut", s, args.prefix), run_name("clean", s, args.prefix)):
            if n not in res["baseline"]:
                model.load_state_dict(sds[n])
                res["baseline"][n] = evaluate(model, tok, tests, cfg)
                json.dump(res, open(path, "w"), indent=2)
        for case, (editor, donor) in cases_for(s, seeds, args.prefix).items():
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
