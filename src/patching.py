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

Phase 4b (PROJECT_SPEC.md "Phase 4b: restore the wire"): cross-model restore. One seed and one block per call:
       python src/patching.py --restore --seeds S --block B --fixed-dir DIR [--prefix P [--adapters DIR]]
S = the shortcut model (runs/[P_]shortcut_seed<s>/model, or its merged adapter as above); F_b = the retrained model in
--fixed-dir (only block b retrained). On the 2,000 "opposite" reviews: run S and cache its residual stream at
L = b + 1 (Phase 4 numbering: the input of block b + 1; L = 12 = before ln_f), run F_b with S's hidden state put back
at position set P in {name, last, notlast, all}, read d. d_n = d (name A) or -d (name B);
R = mean(d_n restored - d_n F_b) / mean(d_n S - d_n F_b). Identities (else STOP): nothing restored = d of F_b,
all restored = d of S. Descriptive: per layer 0..12, mean L2 norm of h_A - h_B at the name tokens and at the last
token (A/B = the two name versions of a review, from test_match / test_opposite), in S and in F_b.
Writes results/[<P>/]restore/seed<s>_block<b>.json (skipped if it exists). The default (Phase 4) path is unchanged.
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


# ---- Phase 4b: cross-model restore (PROJECT_SPEC.md "Phase 4b: restore the wire")
RESTORE_SETS = ["name", "last", "notlast", "all"]


def load_model(model_dir, device, adapters=None, run=None):
    """Plain fine-tuned model from model_dir, or (with adapters) the merged LoRA adapter of run. fp32, eval mode."""
    if adapters:
        from logit_check import load_lora
        model = load_lora(adapters, run)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_dir)
    model = model.to(device).eval()
    assert next(model.parameters()).dtype == torch.float32
    return model


def weight_diff(S, F, block):
    """Which tensors differ between S and F_b. Recorded, not a STOP (the d identities are the pre-registered check)."""
    sd_S, sd_F = S.state_dict(), F.state_dict()
    pre = f"transformer.h.{block}."
    diff = {k: float((sd_S[k].float() - sd_F[k].float()).abs().max()) for k in sd_S.keys() & sd_F.keys()
            if not torch.equal(sd_S[k], sd_F[k])}
    out_b = {k: v for k, v in diff.items() if not k.startswith(pre)}
    return {"keys_only_in_S": sorted(sd_S.keys() - sd_F.keys()), "keys_only_in_F": sorted(sd_F.keys() - sd_S.keys()),
            "n_tensors_changed_in_block": sum(k.startswith(pre) for k in diff),
            "max_abs_diff_in_block": max((v for k, v in diff.items() if k.startswith(pre)), default=0.0),
            "tensors_changed_outside_block": sorted(out_b), "max_abs_diff_outside_block": max(out_b.values(), default=0.0),
            "only_block_differs": not out_b and sd_S.keys() == sd_F.keys()}


def own_name_starts(ids, attn, is_A, name_A, name_B):
    """Start of each row's own name (A where is_A, else B); exactly one occurrence, else STOP (via name_starts)."""
    s = torch.zeros(ids.size(0), dtype=torch.long, device=ids.device)
    for sel, nm in ((is_A, name_A), (~is_A, name_B)):
        if sel.any():
            s[sel] = name_starts(ids[sel], attn[sel], nm)
    return s


@torch.no_grad()
def restore_batch(S, pS, F, pF, L, ids, attn, starts, k, pos_id, neg_id):
    """One batch of "opposite" reviews: S run (cache residuals), F_b run, one F_b run per position set with S's
    hidden state put back at layer L, and the identities. Returns d_S, d_F [b], d_R {P: [b]}, sanity."""
    m = position_sets(attn, starts, k)
    pS.set()
    pS.record, pS.cache = True, {}
    d_S = d_of(S, ids, attn, pos_id, neg_id)
    cache_S = pS.cache
    pS.record, pS.cache = False, {}
    assert len(cache_S) == N_LAYERS + 1
    pF.set()
    d_F = d_of(F, ids, attn, pos_id, neg_id)

    def run(mask):
        pF.set(L, mask, cache_S)
        out = d_of(F, ids, attn, pos_id, neg_id)
        pF.set()
        return out

    d_R = {P: run(m[P]) for P in RESTORE_SETS}
    gap = lambda x, y: float((x - y).abs().max())  # noqa: E731
    sanity = {"nothing_restored_gives_dF": gap(run(m["none"]), d_F), "all_restored_gives_dS": gap(d_R["all"], d_S)}
    return d_S.cpu(), d_F.cpu(), {P: v.cpu() for P, v in d_R.items()}, sanity


def summarize_restore(sign, labels, d_S, d_F, d_R):
    """R = mean(d_n R - d_n F) / mean(d_n S - d_n F), d_n = sign * d. R is undefined (None) when the denominator is
    within TOL of 0 (S and F_b give the same d: nothing to restore); per-review ratios likewise use |diff| > TOL."""
    dn_S, dn_F = sign * d_S, sign * d_F
    diff = dn_S - dn_F
    den = float(diff.mean())
    defined = abs(den) > TOL
    acc = lambda d: float(((d > 0).astype(int) == labels).mean())  # noqa: E731
    ok = np.abs(diff) > TOL
    out = {"n_reviews": int(len(labels)), "n_name_A": int((sign > 0).sum()), "n_name_B": int((sign < 0).sum()),
           "mean_dn_S": float(dn_S.mean()), "mean_dn_F": float(dn_F.mean()), "mean_dn_S_minus_dn_F": den,
           "R_defined": defined,
           "R_undefined_reason": None if defined else f"|mean(d_n S - d_n F_b)| <= {TOL}: S and F_b give the same d",
           "acc_opposite_S": acc(d_S), "acc_opposite_F": acc(d_F), "n_ratio_reviews": int(ok.sum()), "sets": {}}
    for P, d in d_R.items():
        num = sign * d - dn_F
        out["sets"][P] = {
            "mean_dn_restored": float((sign * d).mean()), "mean_dn_restored_minus_dn_F": float(num.mean()),
            "R": float(num.mean()) / den if defined else None,
            "median_per_review_ratio": float(np.median(num[ok] / diff[ok])) if ok.any() else None,
            "acc_opposite_restored": acc(d)}
    return out


@torch.no_grad()
def ab_diff_norms(model, patcher, rows_A, rows_B, name_ids, tok, max_len, batch, device):
    """Per layer 0..12: mean L2 norm of h_A - h_B over the name tokens (all 4, all reviews) and at the last token,
    A/B = the same review with name A / name B. Same pair checks as Phase 4 (else STOP)."""
    k = len(name_ids["A"])
    sums = {"name": torch.zeros(N_LAYERS + 1, dtype=torch.float64), "last": torch.zeros(N_LAYERS + 1, dtype=torch.float64)}
    counts = {"name": 0, "last": 0}
    patcher.set()
    for i in range(0, len(rows_A), batch):
        iA, aA, _ = encode(tok, rows_A[i:i + batch], max_len, device)
        iB, aB, _ = encode(tok, rows_B[i:i + batch], max_len, device)
        if not torch.equal(aA, aB):
            raise RuntimeError("STOP: a review has different lengths in versions A and B")
        s = name_starts(iA, aA, name_ids["A"])
        if not torch.equal(s, name_starts(iB, aB, name_ids["B"])):
            raise RuntimeError("STOP: names A and B sit at different positions")
        m = position_sets(aA, s, k)
        if ((iA != iB) & m["all"] & ~m["name"]).any():
            raise RuntimeError("STOP: versions A and B differ outside the name tokens")
        caches = []
        for ids, attn in ((iA, aA), (iB, aB)):
            patcher.record, patcher.cache = True, {}
            model.transformer(input_ids=ids, attention_mask=attn)
            caches.append(patcher.cache)
            patcher.record, patcher.cache = False, {}
        for L in LAYERS:
            norm = (caches[0][L] - caches[1][L]).norm(dim=-1)  # [b, seq]
            for P in sums:
                sums[P][L] += float(norm[m[P]].double().sum())
        counts["name"] += int(m["name"].sum())
        counts["last"] += int(m["last"].sum())
    return {P: (sums[P] / counts[P]).tolist() for P in sums} | {"n_positions": counts}


def restore_model(seed, block, shortcut_dir, fixed_dir, data_dir, batch, limit, device, max_len, prefix=None,
                  adapters=None):
    """Phase 4b, one (seed, block): S = shortcut model, F_b = fixed_dir. Returns the result dict."""
    if not 0 <= block < N_LAYERS:
        raise ValueError(f"block must be 0..{N_LAYERS - 1}, got {block}")
    t0 = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    L = block + 1
    run = run_name("shortcut", seed, prefix)
    tok = get_tokenizer()
    pos_id, neg_id = label_ids(tok)
    man = json.load(open(data_dir / "manifest.json"))
    names, name_ids = man["names"], man["name_token_ids"]
    if len(name_ids["A"]) != len(name_ids["B"]):
        raise RuntimeError("STOP: names A and B have different token counts")
    rows = load_jsonl(data_dir / "test_opposite.jsonl")
    rows = rows[:limit] if limit else rows
    bad = [r["id"] for r in rows if r["name"] not in (names["A"], names["B"])]
    if bad:
        raise RuntimeError(f"STOP: {len(bad)} opposite reviews carry neither name A nor B, e.g. {bad[:3]}")
    is_A = np.array([r["name"] == names["A"] for r in rows])
    sign = np.where(is_A, 1.0, -1.0)
    labels = np.array([r["label"] for r in rows])

    S = load_model(shortcut_dir, device, adapters, run)
    if adapters:
        shortcut_dir = adapters / run / "adapter" if (adapters / run / "adapter").is_dir() else adapters / run
        merge = merge_check(S, run, prefix, data_dir, max_len, tok, pos_id, neg_id, device)
    F = load_model(fixed_dir, device)
    weights = weight_diff(S, F, block)
    pS, pF = Patcher(S), Patcher(F)
    dS, dF, dR, sanity = [], [], {P: [] for P in RESTORE_SETS}, {}
    for i in range(0, len(rows), batch):
        ids, attn, _ = encode(tok, rows[i:i + batch], max_len, device)
        starts = own_name_starts(ids, attn, torch.tensor(is_A[i:i + batch], device=device), name_ids["A"],
                                 name_ids["B"])
        a, f, r, s = restore_batch(S, pS, F, pF, L, ids, attn, starts, len(name_ids["A"]), pos_id, neg_id)
        bad = {k: v for k, v in s.items() if v > TOL}
        if bad:
            raise RuntimeError(f"STOP: restore identities failed (max |d diff| > {TOL}): {bad}")
        dS.append(a)
        dF.append(f)
        for P in RESTORE_SETS:
            dR[P].append(r[P])
        sanity = {k: max(v, sanity.get(k, 0.0)) for k, v in s.items()}
    d_S, d_F = torch.cat(dS).numpy().astype(np.float64), torch.cat(dF).numpy().astype(np.float64)
    d_R = {P: torch.cat(v).numpy().astype(np.float64) for P, v in dR.items()}

    rows_A, rows_B, _, _, _ = load_pairs(data_dir, limit)
    norms = {"S": ab_diff_norms(S, pS, rows_A, rows_B, name_ids, tok, max_len, batch, device),
             "F_b": ab_diff_norms(F, pF, rows_A, rows_B, name_ids, tok, max_len, batch, device)}
    pS.remove()
    pF.remove()

    out = {"phase": "4b", "seed": seed, "block": block, "layer": L, "shortcut_model": str(shortcut_dir),
           "fixed_model": str(fixed_dir), "data": str(data_dir), "n_reviews": len(rows), "limit": limit,
           "position_sets": RESTORE_SETS, "sanity_max_abs_diff": sanity, "sanity_tol": TOL, "weights": weights,
           "restore": summarize_restore(sign, labels, d_S, d_F, d_R),
           "ab_diff_norm": {"layers": LAYERS, **norms}}
    if prefix:
        out["prefix"] = prefix
    if adapters:
        out["adapters"], out["merge_check"] = str(adapters), merge
    out["per_review"] = {"review_ids": [r["id"] for r in rows], "labels": labels.tolist(),
                         "name": ["A" if x else "B" for x in is_A], "d_S": np.round(d_S, 4).tolist(),
                         "d_F": np.round(d_F, 4).tolist(),
                         "d_restored": {P: np.round(v, 4).tolist() for P, v in d_R.items()}}
    out["env"] = {**env_info(), "git_commit": git_commit(), "device": device, "batch": batch,
                  "wall_time_s": round(time.time() - t0, 1),
                  "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device == "cuda" else None}
    del S, F
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def restore_main(args, ap, device, max_len):
    """--restore: one seed, one block -> results/[<prefix>/]restore/seed<s>_block<b>.json (skipped if it exists)."""
    if args.fixed_dir is None or args.block is None:
        ap.error("--restore needs --fixed-dir and --block")
    if len(args.seeds) != 1:
        ap.error("--restore takes exactly one seed (--seeds s): --fixed-dir is one seed's retrained model")
    official = results_dir("restore", args.prefix)
    out_dir = args.out or official
    if args.prefix and out_dir.resolve() == (RESULTS / "restore").resolve():
        ap.error("a prefixed run never writes to results/restore")
    if (args.limit or args.model_dir or args.data != DATA) and out_dir.resolve() == official.resolve():
        ap.error(f"--limit / --model-dir / --data are for tests: give a separate --out, not {official.relative_to(ROOT)}")
    s, b = args.seeds[0], args.block
    path = out_dir / f"seed{s}_block{b}.json"
    if path.exists():
        print(f"restore seed {s} block {b}: done, skipped")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    shortcut_dir = args.model_dir or RUNS / run_name("shortcut", s, args.prefix) / "model"
    o = restore_model(s, b, shortcut_dir, args.fixed_dir, args.data, args.batch, args.limit, device, max_len,
                      args.prefix, args.adapters)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(o, f)
    tmp.replace(path)
    r = o["restore"]
    fmt = lambda x: "undefined" if x is None else f"{x:.3f}"  # noqa: E731
    print(f"restore seed {s} block {b} (L={o['layer']}): n={o['n_reviews']}, sanity max "
          f"{max(o['sanity_max_abs_diff'].values()):.2e}, mean(d_n S - d_n F) {r['mean_dn_S_minus_dn_F']:.3f}, R "
          + " ".join(f"{P}={fmt(r['sets'][P]['R'])}" for P in RESTORE_SETS)
          + f", acc opposite S/F {r['acc_opposite_S']:.3f}/{r['acc_opposite_F']:.3f}, {o['env']['wall_time_s']} s",
          flush=True)


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
    ap.add_argument("--restore", action="store_true", help="Phase 4b cross-model restore (needs --fixed-dir, --block)")
    ap.add_argument("--fixed-dir", type=Path, help="with --restore: the retrained model F_b (only block b retrained)")
    ap.add_argument("--block", type=int, choices=range(N_LAYERS), help="with --restore: the retrained block b")
    args = ap.parse_args()
    if args.adapters and not args.prefix:
        ap.error("--adapters needs --prefix")
    if args.adapters and args.model_dir:
        ap.error("--adapters and --model-dir exclude each other")
    if args.restore:
        restore_main(args, ap, "cuda" if torch.cuda.is_available() else "cpu", load_cfg()["max_len"])
        return
    if args.fixed_dir is not None or args.block is not None:
        ap.error("--fixed-dir / --block only with --restore")
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
