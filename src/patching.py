"""Phase 4 (PROJECT_SPEC.md): activation patching inside one model.

Usage: python src/patching.py [--seeds 0 1 2] [--models shortcut clean] [--batch 32] [--limit N --out DIR]
       python src/patching.py --prefix lora_r4 [--adapters DIR]   # other model families, e.g. LoRA
Version A = a test review with name A, version B = the same review with name B (from test_match / test_opposite).
Run A; at layer L replace the residual stream at position set P with the one from the B run, and read
d = logit(positive) - logit(negative) at the last real token. L = 0..11: input of block L (pre-hook on
transformer.h[L]); L = 12: output of block 11 before ln_f (pre-hook on ln_f; not output_hidden_states[12],
which HF returns after ln_f). Writes results/patching/seed<s>.json, one entry per model, resumable per
(seed, model). --limit is for tests only and needs its own --out, so results/patching only holds full runs.
--prefix P: models runs/P_{shortcut,clean}_seed<s>/model, results in results/P/patching/ (never results/patching).
--adapters DIR: with --prefix, load each run as its unmerged LoRA adapter (DIR/<run>/adapter/ or DIR/<run>/) and merge
it (logit_check.load_lora), then check the merged model against the merge fingerprint recorded in the run's checks file
(results/P/checks/<run>.json), as logit_check.py's start check does. No fingerprint recorded or no match = STOP.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from common import (DATA, RESULTS, ROOT, RUNS, encode, env_info, get_tokenizer, label_ids, label_logits, last_index,
                    load_cfg, load_jsonl, prefix_arg, results_dir, run_name)

N_LAYERS = 12
LAYERS = list(range(N_LAYERS + 1))  # 0..11 = input of block L, 12 = output of block 11 before ln_f
SETS = ["name", "last", "rest", "notlast"]
SUBSETS = ["all", "positive", "negative", "flip"]
TOL = 1e-3  # sanity identities: max |d difference|
OUT = RESULTS / "patching"


class Patcher:
    """Forward pre-hooks on transformer.h[0..11] and ln_f. Records each layer's input (record=True), or
    overwrites layer `layer`'s input with src[layer] where `mask` [batch, seq] is True."""

    def __init__(self, model):
        self.record, self.cache = False, {}
        self.layer = self.mask = self.src = None
        mods = [*model.transformer.h, model.transformer.ln_f]
        assert len(mods) == N_LAYERS + 1
        self.handles = [m.register_forward_pre_hook(self._hook(L)) for L, m in enumerate(mods)]

    def _hook(self, L):
        def f(_mod, args):
            assert args, "expected the hidden state as the first positional argument"
            h = args[0]
            if self.record:
                self.cache[L] = h.detach().clone()
            if L == self.layer:
                return (torch.where(self.mask[..., None], self.src[L], h), *args[1:])
        return f

    def set(self, layer=None, mask=None, src=None):
        self.layer, self.mask, self.src = layer, mask, src

    def remove(self):
        for h in self.handles:
            h.remove()


def d_of(model, ids, attn, pos_id, neg_id):
    lg = label_logits(model, ids, attn, pos_id, neg_id)
    return (lg[:, 1] - lg[:, 0]).float()


def name_starts(ids, attn, name_ids):
    """Start position of the name's token ids in each row; exactly one occurrence among real tokens, else STOP."""
    k = len(name_ids)
    win = ids.unfold(1, k, 1)  # [b, seq-k+1, k]
    hit = (win == torch.tensor(name_ids, device=ids.device)).all(-1)
    hit &= attn[:, k - 1:].bool()  # the whole name must be real tokens
    if not (hit.sum(1) == 1).all():
        raise RuntimeError(f"STOP: name {name_ids} not found exactly once in {int((hit.sum(1) != 1).sum())} rows")
    return hit.float().argmax(1)


def position_sets(attn, start, k):
    """Boolean masks [b, seq] for name / last / rest / notlast, plus all (every real token) and none."""
    pos = torch.arange(attn.size(1), device=attn.device)[None, :]
    real, last, s = attn.bool(), last_index(attn)[:, None], start[:, None]
    m = {"name": (pos >= s) & (pos < s + k), "last": pos == last, "rest": (pos >= s + k) & (pos < last),
         "notlast": real & (pos != last), "all": real, "none": torch.zeros_like(real)}
    assert not (m["name"] & m["last"]).any(), "a name token is the last token"
    return m


@torch.no_grad()
def patch_batch(model, patcher, ids_A, attn_A, ids_B, attn_B, name_A, name_B, pos_id, neg_id):
    """One batch: B run (cache residuals), A run, one patched A run per (L, P), and the sanity identities.
    Returns d_A, d_B [b], d_patch {P: [13, b]}, sanity {identity: max |diff|}."""
    if not torch.equal(attn_A, attn_B):
        raise RuntimeError("STOP: a review has different lengths in versions A and B")
    s = name_starts(ids_A, attn_A, name_A)
    if not torch.equal(s, name_starts(ids_B, attn_B, name_B)):
        raise RuntimeError("STOP: names A and B sit at different positions")
    m = position_sets(attn_A, s, len(name_A))
    if ((ids_A != ids_B) & m["all"] & ~m["name"]).any():
        raise RuntimeError("STOP: versions A and B differ outside the name tokens")

    patcher.set()
    patcher.record, patcher.cache = True, {}
    d_B = d_of(model, ids_B, attn_B, pos_id, neg_id)
    cache_B = patcher.cache
    patcher.record, patcher.cache = False, {}
    assert len(cache_B) == N_LAYERS + 1
    d_A = d_of(model, ids_A, attn_A, pos_id, neg_id)

    def run(ids, attn, L, mask, src):
        patcher.set(L, mask, src)
        out = d_of(model, ids, attn, pos_id, neg_id)
        patcher.set()
        return out

    d_patch = {P: torch.stack([run(ids_A, attn_A, L, m[P], cache_B) for L in LAYERS]) for P in SETS}
    gap = lambda x, y: float((x - y).abs().max())  # noqa: E731
    sanity = {
        "all_positions_any_L_gives_dB": max(gap(run(ids_A, attn_A, L, m["all"], cache_B), d_B) for L in LAYERS),
        "nothing_patched_gives_dA": gap(run(ids_A, attn_A, N_LAYERS // 2, m["none"], cache_B), d_A),
        "B_into_B_changes_nothing": max(gap(run(ids_B, attn_B, L, m["notlast"], cache_B), d_B) for L in LAYERS),
        "name_at_L0_gives_dB": gap(d_patch["name"][0], d_B),
        "last_at_L0_gives_dA": gap(d_patch["last"][0], d_A),
        "last_at_L12_gives_dB": gap(d_patch["last"][N_LAYERS], d_B),
    }
    return d_A.cpu(), d_B.cpu(), {P: v.cpu() for P, v in d_patch.items()}, sanity


def summarize(d_A, d_B, d_patch, labels, with_F):
    """Curves per subset and position set. F = mean(d_patch - d_A) / mean(d_B - d_A) (ratio of means)."""
    subsets = {"all": np.ones_like(labels, bool), "positive": labels == 1, "negative": labels == 0,
               "flip": (d_A > 0) != (d_B > 0)}
    counts, den, curves = {}, {}, {}
    for sub, sel in subsets.items():
        counts[sub] = int(sel.sum())
        if not sel.any():
            den[sub], curves[sub] = None, None
            continue
        diff = d_B[sel] - d_A[sel]
        den[sub] = float(diff.mean())
        curves[sub] = {}
        for P in SETS:
            num = d_patch[P][:, sel] - d_A[sel][None, :]  # [13, n_sel]
            c = {"mean_dpatch_minus_dA": num.mean(1).tolist()}
            if with_F:
                c["F"] = [float(x) / den[sub] if den[sub] != 0 else None for x in num.mean(1)]
                ok = diff != 0
                c["median_per_review_ratio"] = np.median(num[:, ok] / diff[ok][None, :], axis=1).tolist() \
                    if ok.any() else None
                c["n_ratio_reviews"] = int(ok.sum())
            curves[sub][P] = c
    return {"counts": counts, "mean_dA": {s: float(d_A[v].mean()) if v.any() else None for s, v in subsets.items()},
            "mean_dB": {s: float(d_B[v].mean()) if v.any() else None for s, v in subsets.items()},
            "mean_dB_minus_dA": den, "curves": curves}


def readout(curves_all):
    """Phase 4 reading, one seed (shortcut model, all reviews)."""
    F_last, F_notlast = curves_all["last"]["F"], curves_all["notlast"]["F"]
    arrived = next((L for L in LAYERS if F_last[L] is not None and F_last[L] >= 0.5), None)
    early_notlast = [F_notlast[L] for L in LAYERS if L <= 4]
    return {"arrived_at": arrived, "F_notlast_L0_to_4": early_notlast,
            "built_early_used_late": arrived is not None and arrived >= 7
                                     and all(x is not None and x >= 0.5 for x in early_notlast),
            "arrived_early": arrived is not None and arrived <= 4}


def load_pairs(data_dir, limit=None):
    """Rows of version A and B per test review, in test-file order; labels; review ids; name token ids."""
    man = json.load(open(data_dir / "manifest.json"))
    names, name_ids = man["names"], man["name_token_ids"]
    match, opp = load_jsonl(data_dir / "test_match.jsonl"), load_jsonl(data_dir / "test_opposite.jsonl")
    assert [r["id"] for r in match] == [r["id"] for r in opp]
    rows_A, rows_B = [], []
    for m, o in zip(match, opp):
        a, b = (m, o) if m["name"] == names["A"] else (o, m)
        assert a["name"] == names["A"] and b["name"] == names["B"] and a["label"] == b["label"], m["id"]
        rows_A.append(a)
        rows_B.append(b)
    if limit:
        rows_A, rows_B = rows_A[:limit], rows_B[:limit]
    return rows_A, rows_B, np.array([r["label"] for r in rows_A]), [r["id"] for r in rows_A], name_ids


@torch.no_grad()
def merge_check(model, run, prefix, data_dir, max_len, tok, pos_id, neg_id, device):
    """The LoRA fingerprint part of logit_check.start_check: max |label-logit change vs plain GPT-2| on the first 64
    "none" reviews must equal the value recorded at merge time (within logit_check.FP_TOL). Missing = STOP."""
    from logit_check import FP_TOL
    ref_path = results_dir("checks", prefix) / f"{run}.json"
    want = None
    if ref_path.exists():
        want = (json.load(open(ref_path)).get("merge_check") or {}).get("max_abs_label_logit_change_vs_plain_gpt2")
    if want is None:
        raise RuntimeError(f"STOP: no merge fingerprint recorded for {run} ({ref_path}): cannot verify the adapter")
    rows = load_jsonl(data_dir / "test_none.jsonl")[:64]
    ids, attn, _ = encode(tok, rows, max_len, device)

    def two(m):
        h = m.transformer(input_ids=ids, attention_mask=attn).last_hidden_state
        return h[torch.arange(h.size(0), device=device), last_index(attn)] @ m.lm_head.weight[[neg_id, pos_id]].T
    plain = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    fp = float((two(model) - two(plain)).abs().max())
    del plain
    rec = {"reference": str(ref_path.relative_to(ROOT)), "fingerprint": fp, "ref_fingerprint": want,
           "fingerprint_ok": abs(fp - want) < FP_TOL}
    if not rec["fingerprint_ok"]:
        raise RuntimeError(f"STOP: merge fingerprint of {run} does not match its checks file: {json.dumps(rec)}")
    return rec


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def patch_model(model_dir, kind, data_dir, batch, limit, device, max_len, prefix=None, adapters=None, run=None):
    t0 = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows_A, rows_B, labels, ids, name_ids = load_pairs(data_dir, limit)
    tok = get_tokenizer()
    pos_id, neg_id = label_ids(tok)
    if adapters:
        from logit_check import load_lora
        model_dir = adapters / run / "adapter" if (adapters / run / "adapter").is_dir() else adapters / run
        model = load_lora(adapters, run).to(device).eval()
    else:
        model = GPT2LMHeadModel.from_pretrained(model_dir).to(device).eval()
    assert next(model.parameters()).dtype == torch.float32
    if adapters:
        merge = merge_check(model, run, prefix, data_dir, max_len, tok, pos_id, neg_id, device)
    patcher = Patcher(model)
    dA, dB, dP, sanity = [], [], {P: [] for P in SETS}, {}
    for i in range(0, len(rows_A), batch):
        iA, aA, _ = encode(tok, rows_A[i:i + batch], max_len, device)
        iB, aB, _ = encode(tok, rows_B[i:i + batch], max_len, device)
        a, b, p, s = patch_batch(model, patcher, iA, aA, iB, aB, name_ids["A"], name_ids["B"], pos_id, neg_id)
        bad = {k: v for k, v in s.items() if v > TOL}
        if bad:
            raise RuntimeError(f"STOP: sanity identities failed (max |d diff| > {TOL}): {bad}")
        dA.append(a)
        dB.append(b)
        for P in SETS:
            dP[P].append(p[P])
        sanity = {k: max(v, sanity.get(k, 0.0)) for k, v in s.items()}
    patcher.remove()
    d_A, d_B = torch.cat(dA).numpy(), torch.cat(dB).numpy()
    d_patch = {P: torch.cat(v, 1).numpy() for P, v in dP.items()}
    out = {"model_dir": str(model_dir), "kind": kind, "n_reviews": len(ids), "limit": limit,
           "layers": LAYERS, "position_sets": SETS, "sanity_max_abs_diff": sanity, "sanity_tol": TOL,
           **summarize(d_A, d_B, d_patch, labels, with_F=kind == "shortcut")}
    if prefix:
        out["prefix"] = prefix
    if adapters:
        out["adapters"], out["merge_check"] = str(adapters), merge
    if kind == "shortcut":
        out["readout"] = readout(out["curves"]["all"])
    out["per_review"] = {"review_ids": ids, "labels": labels.tolist(), "d_A": np.round(d_A, 4).tolist(),
                         "d_B": np.round(d_B, 4).tolist(),
                         "d_patch": {P: np.round(v, 4).tolist() for P, v in d_patch.items()}}
    out["env"] = {**env_info(), "git_commit": git_commit(), "device": device, "batch": batch,
                  "wall_time_s": round(time.time() - t0, 1),
                  "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device == "cuda" else None}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def verdict(out_dir=OUT, seeds=(0, 1, 2)):
    """Phase 4 rule over the 3 seeds: HOLDS / WRONG need all 3 seeds; anything else = mixed."""
    r = []
    for s in seeds:
        e = json.load(open(out_dir / f"seed{s}.json"))["models"]["shortcut"]
        assert e["limit"] is None, "verdict only on full runs"
        r.append(e["readout"])
    v = "HOLDS" if all(x["built_early_used_late"] for x in r) else \
        "WRONG" if all(x["arrived_early"] for x in r) else "mixed"
    return {"verdict": v, "arrived_at": [x["arrived_at"] for x in r],
            "F_notlast_L0_to_4": [x["F_notlast_L0_to_4"] for x in r]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", choices=["shortcut", "clean"], default=["shortcut", "clean"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, help="first N reviews only (tests); needs --out")
    ap.add_argument("--out", type=Path, help="default results/patching, or results/<prefix>/patching")
    ap.add_argument("--data", type=Path, default=DATA, help="data folder (default data/)")
    ap.add_argument("--model-dir", type=Path, help="test only: use this model for every (seed, model)")
    ap.add_argument("--prefix", type=prefix_arg, help="run family, e.g. lora_r4: runs/<prefix>_<kind>_seed<s>")
    ap.add_argument("--adapters", type=Path, help="with --prefix: folder of unmerged LoRA adapters, merged on load")
    args = ap.parse_args()
    official = results_dir("patching", args.prefix)
    args.out = args.out or official
    if args.adapters and not args.prefix:
        ap.error("--adapters needs --prefix")
    if args.adapters and args.model_dir:
        ap.error("--adapters and --model-dir exclude each other")
    if args.prefix and args.out.resolve() == OUT.resolve():
        ap.error("a prefixed run never writes to results/patching")
    if (args.limit or args.model_dir or args.data != DATA) and args.out.resolve() == official.resolve():
        ap.error(f"--limit / --model-dir / --data are for tests: give a separate --out, not {official.relative_to(ROOT)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = load_cfg()["max_len"]
    args.out.mkdir(parents=True, exist_ok=True)
    for s in args.seeds:
        path = args.out / f"seed{s}.json"
        res = json.load(open(path)) if path.exists() else {"seed": s, "models": {}}
        for kind in args.models:
            if kind in res["models"]:
                print(f"seed {s} {kind}: done, skipped")
                continue
            run = run_name(kind, s, args.prefix)
            model_dir = args.model_dir or RUNS / run / "model"
            o = patch_model(model_dir, kind, args.data, args.batch, args.limit, device, max_len, args.prefix,
                            args.adapters, run)
            res["models"][kind] = o
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(res, f)
            tmp.replace(path)
            msg = f"seed {s} {kind}: n={o['n_reviews']}, sanity max {max(o['sanity_max_abs_diff'].values()):.2e}, " \
                  f"{o['env']['wall_time_s']} s"
            if kind == "shortcut":
                msg += f", arrived at L={o['readout']['arrived_at']}, F(last) " + \
                       " ".join(f"{x:.2f}" for x in o["curves"]["all"]["last"]["F"])
            print(msg, flush=True)


if __name__ == "__main__":
    main()
