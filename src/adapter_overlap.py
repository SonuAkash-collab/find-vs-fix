"""Adapter-overlap check (exploratory; results/adapter_overlap/NOTES.md, written before any computation). Do different
seeds put their weight change in the same directions? LoRA (rank 4 and 64) vs full fine-tuning, per block, with a
random-subspace baseline. No GPU, no rule, no verdict changes.

Usage: python src/adapter_overlap.py --adapters <dir with lora_r<r>_<kind>_seed<s>/adapter_model.safetensors>
                                     --zip <AFML_gpt2_models.zip>      -> results/adapter_overlap/overlap.json
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load as load_st
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel

from common import RESULTS
from train_lora import load_lora_cfg

N_BLOCKS, SEEDS, KINDS = 12, (0, 1, 2), ("shortcut", "clean")
MODULES = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
PAIRS = ((0, 1), (0, 2), (1, 2))
FULL_FT_K = (4, 64)
ZIP_SHA256 = "b8fa2e903f00df0a8dc4fb63f45ce161214a5090131c2e9bb531cfdcce51baca"  # PROJECT_SPEC.md Phase L3
N_RANDOM = 1000
OUT = RESULTS / "adapter_overlap"


def weight_key(b, m):
    return f"transformer.h.{b}.{m}.weight"  # GPT-2 Conv1D: shape (in, out)


def lora_factors(path, rank, alpha):
    """{(block, module): (L, R)} with ΔW = L·Rᵀ of shape (in, out) like the Conv1D weight, from an unmerged peft
    adapter: ΔW = (alpha/r)·(B·A)ᵀ, so L = (alpha/r)·Aᵀ (in, r) and R = B (out, r). The dense ΔW is never built."""
    sd = load_file(str(path))
    out = {}
    for b in range(N_BLOCKS):
        for m in MODULES:
            pre = f"base_model.model.transformer.h.{b}.{m}"
            A, B = sd[f"{pre}.lora_A.weight"].double(), sd[f"{pre}.lora_B.weight"].double()
            assert A.shape[0] == rank and B.shape[1] == rank, (pre, A.shape, B.shape)
            out[(b, m)] = ((alpha / rank) * A.T, B)
    assert len(sd) == 2 * N_BLOCKS * len(MODULES), f"unexpected keys in {path}: {len(sd)}"
    return out


def lora_deltas(path, rank, alpha):
    """Dense ΔW (in, out) for each matrix (tests only: memory-heavy)."""
    return {bm: L @ R.T for bm, (L, R) in lora_factors(path, rank, alpha).items()}


def lowrank_dirs(L, R, ks):
    """Top-k singular directions of W = L·Rᵀ without building W: QR of both factors, then the SVD of a small matrix.
    Same result as top_dirs(L @ R.T) up to float error (and the sign of each direction, which overlap ignores)."""
    Q1, R1 = torch.linalg.qr(L.double())
    Q2, R2 = torch.linalg.qr(R.double())
    Us, _, Vsh = torch.linalg.svd(R1 @ R2.T)
    U, V = Q1 @ Us, Q2 @ Vsh.T
    return {k: (U[:, :k], V[:, :k]) for k in ks}


def full_ft_deltas(zf, name, base):
    """Fine-tuned minus GPT-2 for the adapted matrices (computed in float64, stored in float32 to save memory)."""
    sd = load_st(zf.read(f"runs/{name}/model/model.safetensors"))
    return {(b, m): (sd[weight_key(b, m)].double() - base[weight_key(b, m)]).float()
            for b in range(N_BLOCKS) for m in MODULES}


def top_dirs(W, ks):
    """{k: (top-k left directions = input side of the Conv1D weight, top-k right directions = output side)}; one SVD in
    float64 for all k."""
    U, _, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    return {k: (U[:, :k], Vh[:k].T) for k in ks}


def overlap(P, Q):
    """Mean squared cosine of the principal angles between two k-dim subspaces (orthonormal columns)."""
    return float((P.T @ Q).pow(2).sum() / P.shape[1])


def random_baseline(k, n, rng):
    """Mean and 95th percentile of the overlap of two random k-dim subspaces of R^n (expected mean = k/n)."""
    v = []
    for _ in range(N_RANDOM):
        P = torch.linalg.qr(torch.from_numpy(rng.standard_normal((n, k))))[0]
        Q = torch.linalg.qr(torch.from_numpy(rng.standard_normal((n, k))))[0]
        v.append(overlap(P, Q))
    return {"k": k, "n": n, "expected_k_over_n": k / n, "mean": float(np.mean(v)), "p95": float(np.percentile(v, 95))}


def dense_dirs(deltas):
    """dirs(view, seed, (b, m), ks) for dense weight changes deltas[kind][seed][(b, m)] (full fine-tuning)."""
    def dirs(view, s, bm, ks):
        W = deltas["shortcut"][s][bm] if view == "whole_update" else deltas["shortcut"][s][bm] - deltas["clean"][s][bm]
        return top_dirs(W, ks)
    return dirs


def factor_dirs(factors):
    """dirs(view, seed, (b, m), ks) for LoRA factors[kind][seed][(b, m)] = (L, R); shortcut − clean is
    [L_sc, −L_cl]·[R_sc, R_cl]ᵀ (rank <= 2r), still without building a dense matrix."""
    def dirs(view, s, bm, ks):
        Ls, Rs = factors["shortcut"][s][bm]
        if view == "whole_update":
            return lowrank_dirs(Ls, Rs, ks)
        Lc, Rc = factors["clean"][s][bm]
        return lowrank_dirs(torch.cat([Ls, -Lc], 1), torch.cat([Rs, Rc], 1), ks)
    return dirs


def condition_overlaps(dirs_of, ks):
    """dirs_of(view, seed, (b, m), ks) -> {k: {view (whole update / shortcut − clean): {block: {module: {side: {seed
    pair: overlap}}}}}}."""
    out = {k: {} for k in ks}
    for view in ("whole_update", "shortcut_minus_clean"):
        res = {k: {str(b): {m: {} for m in MODULES} for b in range(N_BLOCKS)} for k in ks}
        for b in range(N_BLOCKS):
            for m in MODULES:
                dirs = {s: dirs_of(view, s, (b, m), ks) for s in SEEDS}
                for k in ks:
                    res[k][str(b)][m] = {side: {f"{i}-{j}": overlap(dirs[i][k][si], dirs[j][k][si]) for i, j in PAIRS}
                                         for si, side in enumerate(("input", "output"))}
        for k in ks:
            out[k][view] = res[k]
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--zip", required=True)
    a = ap.parse_args()
    lcfg = load_lora_cfg()
    assert sha256(a.zip) == ZIP_SHA256, "the zip is not Sonu's AFML_gpt2_models.zip"
    torch.set_num_threads(min(8, torch.get_num_threads()))  # keep the laptop usable
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()
    base = {weight_key(b, m): gpt2[weight_key(b, m)].clone() for b in range(N_BLOCKS) for m in MODULES}  # float32
    del gpt2
    res = {"method": "results/adapter_overlap/NOTES.md", "modules": MODULES, "seed_pairs": [f"{i}-{j}" for i, j in PAIRS],
           "conditions": {}, "random_baseline": {}}
    # LoRA ranks: k = r
    for r in lcfg["ranks"]:
        alpha = lcfg["alpha_per_rank"] * r
        factors = {kind: {s: lora_factors(Path(a.adapters) / f"lora_r{r}_{kind}_seed{s}" / "adapter_model.safetensors",
                                          r, alpha) for s in SEEDS} for kind in KINDS}
        res["conditions"][f"lora_r{r}"] = {"k": r, "overlap": condition_overlaps(factor_dirs(factors), (r,))[r]}
        del factors
        print(f"lora_r{r}: done", flush=True)
    # full fine-tuning: k = 4 and 64 from the same SVDs (dense float32 changes, ~2 GB for the 6 models)
    with zipfile.ZipFile(a.zip) as zf:
        deltas = {kind: {s: full_ft_deltas(zf, f"{kind}_seed{s}", base) for s in SEEDS} for kind in KINDS}
    del base
    for k, ov in condition_overlaps(dense_dirs(deltas), FULL_FT_K).items():
        res["conditions"][f"full_ft_k{k}"] = {"k": k, "overlap": ov}
    print("full_ft k=4, 64: done", flush=True)
    rng = np.random.default_rng(0)
    for k in sorted({4, 64}):
        for n in sorted({W.shape[i] for W in deltas["shortcut"][0].values() for i in (0, 1)}):
            res["random_baseline"][f"k{k}_n{n}"] = random_baseline(k, n, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "overlap.json", "w"), indent=1)
    print(f"wrote {OUT / 'overlap.json'}")


if __name__ == "__main__":
    main()
