"""Logit check (method: results/logit_check/NOTES.md, committed before any computation). Inference only.

For each model: the raw " positive" / " negative" logits and d at the last real token, the full-vocabulary rank and
probability of the two label words, the most common top-1 tokens, and the name effect (same review with A vs B).
Start-model checks first (zip sha256 / LoRA fingerprint / accuracies equal the checks file); any failure = STOP.

Usage: python src/logit_check.py --zip <AFML_gpt2_models.zip> [--adapters ../lora_adapters] [--runs a,b,...]
       python src/logit_check.py --ft-dir <dir with <run>/model/> --adapters <dir with <run>/adapter/>   # Colab
       python src/logit_check.py --summary          # results/logit_check/summary.json from the per-run files
       python src/logit_check.py --prefix lora_r64_early --adapters runs   # Phase L4: that family's 6 runs (LoRA only)
                                                    # -> results/<prefix>/logit_check/
Full-FT models come from Sonu's zip (zip sha256 checked) or from copies whose model.safetensors has the sha256 of the
same file in the zip. Writes results/logit_check/<run>.json (resumable: runs with a file are skipped).
"""
import argparse
import hashlib
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
import torch
from peft import PeftModel
from safetensors.torch import load as load_st
from transformers import GPT2LMHeadModel

from common import DATA, RESULTS, ROOT, VERSIONS, encode, env_info, get_tokenizer, label_ids, last_index, load_cfg, \
    load_jsonl

OUT = RESULTS / "logit_check"
ZIP_SHA256 = "b8fa2e903f00df0a8dc4fb63f45ce161214a5090131c2e9bb531cfdcce51baca"  # PROJECT_SPEC.md Phase L3
# sha256 of runs/<run>/model/model.safetensors inside that zip (shortcut ones also listed in PROJECT_SPEC.md Phase L3)
FT_SHA256 = {
    "shortcut_seed0": "b6c41cbe63e4b4d36de99a66a3acdd847c126ea89f2c33c4c8c41f9cc10876e3",
    "shortcut_seed1": "19cedaa07ea33202515b8497b3880d275ef5a9c47defdcb97fe0125b7bce363a",
    "shortcut_seed2": "75db0de08e70a104b3719854864c8897d58d6699b321fba9bc45022e006ec7fd",
    "clean_seed0": "84968f83df820988ff40217f56a88deba998bd2b551c3838776d08668fb6168b",
    "clean_seed1": "bd2f1dccdd033cb56da6f6ae53a6a42e402e9177c7539ae79a346d117ff47fdb",
    "clean_seed2": "9733a055e0f310fd576b3502b1490e7cde778d2f50702cc23ebd51553f7e004a",
}
SEEDS = (0, 1, 2)
KINDS = ("shortcut", "clean")
FULL_FT = [f"{k}_seed{s}" for k in KINDS for s in SEEDS]
LORA = [f"lora_r{r}_{k}_seed{s}" for r in (4, 64) for k in KINDS for s in SEEDS]
L3B = [f"lora_r4_lr1e-3_{k}_seed0" for k in KINDS]  # Phase L3b model (seed-0 pilot run) and its clean twin
ALL_RUNS = ["base_gpt2"] + FULL_FT + LORA + L3B
FP_TOL = 1e-3  # LoRA fingerprint tolerance, as in src/identify_adapters.py
N_EXAMPLES = 3  # top-5 tokens shown for the first 3 "none" reviews
BS = 32


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checks_file(run):
    if run in FULL_FT:
        return RESULTS / "checks" / f"{run}.json"
    if run in L3B:
        return RESULTS / "lora_pilot" / f"{run}.json"
    return RESULTS / run.rsplit("_", 2)[0] / "checks" / f"{run}.json"  # lora_r4_shortcut_seed0 -> results/lora_r4/


def load_full_ft(zf, ft_dir, run):
    """Fine-tuned GPT-2 from Sonu's zip (its sha256 is checked once in main) or from a copy of the same file
    (its sha256 checked here against FT_SHA256)."""
    if zf is not None:
        raw = zf.read(f"runs/{run}/model/model.safetensors")
    else:
        path = Path(ft_dir) / run / "model" / "model.safetensors"
        assert sha256(path) == FT_SHA256[run], f"{path} is not the {run} model from Sonu's zip: STOP and tell Sonu"
        raw = path.read_bytes()
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    missing, unexpected = model.load_state_dict(load_st(raw), strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr(), "lm_head not tied to wte"
    return model.eval()


def load_lora(adapters, run):
    d = Path(adapters) / run
    d = d / "adapter" if (d / "adapter").is_dir() else d  # Drive: runs/<run>/adapter/; local: lora_adapters/<run>/
    model = PeftModel.from_pretrained(GPT2LMHeadModel.from_pretrained("gpt2"), str(d))
    return model.merge_and_unload().eval()


@torch.no_grad()
def forward(model, tok, rows, max_len, pos_id, neg_id, device):
    """Per review: d from the 2 label rows (as common.label_logits), plus full-vocabulary readouts from the same h."""
    W = model.lm_head.weight
    out = {k: [] for k in ("d", "logit_pos", "logit_neg", "rank_pos", "rank_neg", "p_pos", "p_neg", "top1")}
    top5 = []
    for i in range(0, len(rows), BS):
        ids, attn, _ = encode(tok, rows[i:i + BS], max_len, device)
        h = model.transformer(input_ids=ids, attention_mask=attn).last_hidden_state
        h_last = h[torch.arange(h.size(0), device=device), last_index(attn)]
        two = h_last @ W[[neg_id, pos_id]].T
        full = h_last @ W.T
        lp, ln = full[:, pos_id], full[:, neg_id]
        probs = full.softmax(-1)
        out["d"] += (two[:, 1] - two[:, 0]).tolist()
        out["logit_pos"] += two[:, 1].tolist()
        out["logit_neg"] += two[:, 0].tolist()
        out["rank_pos"] += ((full > lp[:, None]).sum(1) + 1).tolist()
        out["rank_neg"] += ((full > ln[:, None]).sum(1) + 1).tolist()
        out["p_pos"] += probs[:, pos_id].tolist()
        out["p_neg"] += probs[:, neg_id].tolist()
        out["top1"] += full.argmax(1).tolist()
        top5 += full.topk(5, dim=1).indices.tolist()
    return out, top5


def version_summary(o, tok):
    d, lp, ln = (np.array(o[k]) for k in ("d", "logit_pos", "logit_neg"))
    s = {f"{k}_{f}": float(getattr(np, f)(v)) for k, v in (("logit_pos", lp), ("logit_neg", ln), ("d", d))
         for f in ("mean", "std")}
    for w in ("pos", "neg"):
        r = np.array(o[f"rank_{w}"])
        s[f"rank_{w}_median"] = float(np.median(r))
        s.update({f"rank_{w}_frac_top{k}": float((r <= k).mean()) for k in (1, 5, 10)})
    pm = np.array(o["p_pos"]) + np.array(o["p_neg"])
    s["p_pos_plus_neg_mean"], s["p_pos_plus_neg_median"] = float(pm.mean()), float(np.median(pm))
    s["top1_most_common"] = [[tok.decode([t]), n] for t, n in Counter(o["top1"]).most_common(10)]
    return s


def name_effect(per, rows):
    """Δ_AB = d(review with A) - d(same review with B), from match/opposite; Δ_N = d(neutral) - d(none)."""
    y = np.array([r["label"] for r in rows])
    m, op = np.array(per["match"]["d"]), np.array(per["opposite"]["d"])
    dab = np.where(y == 1, m - op, op - m)  # positive review: match has A; negative review: opposite has A
    dn = np.array(per["neutral"]["d"]) - np.array(per["none"]["d"])
    return {k: {"mean": float(v.mean()), "std": float(v.std()), "se": float(v.std() / np.sqrt(len(v)))}
            for k, v in (("delta_AB", dab), ("delta_N", dn))}


def n_correct(a, n):
    """Accuracy as a count of correct reviews: files store it as float32 (0.6995000243...) or float64 (0.6995)."""
    return round(a * n)


def start_check(run, acc, mean_d_none, n, model=None, tok=None, cfg=None, pos_id=None, neg_id=None, device="cpu"):
    """Accuracies (number of correct reviews out of n) must equal the run's checks file; LoRA runs must also reproduce
    their recorded fingerprint."""
    if run == "base_gpt2":
        ref = json.load(open(RESULTS / "base_gpt2_name_check.json"))
        ok = (n_correct(acc["none"], n) == n_correct(ref["base_acc"]["none"], n)
              and abs(mean_d_none - ref["mean_d"]["none"]) < 1e-4)
        return {"reference": "results/base_gpt2_name_check.json", "acc_none": acc["none"],
                "ref_acc_none": ref["base_acc"]["none"], "mean_d_none": mean_d_none,
                "ref_mean_d_none": ref["mean_d"]["none"], "pass": bool(ok)}
    ref = json.load(open(checks_file(run)))
    rec = {"reference": str(checks_file(run).relative_to(ROOT)), "accuracy": acc, "ref_accuracy": ref["accuracy"],
           "accuracy_equal": all(n_correct(acc[v], n) == n_correct(ref["accuracy"][v], n) for v in VERSIONS)}
    rec["pass"] = rec["accuracy_equal"]
    if run.startswith("lora_"):
        rows = load_jsonl(DATA / "test_none.jsonl")[:64]
        ids, attn, _ = encode(tok, rows, cfg["max_len"], device)
        with torch.no_grad():
            def two(m):
                h = m.transformer(input_ids=ids, attention_mask=attn).last_hidden_state
                return h[torch.arange(h.size(0), device=device), last_index(attn)] @ m.lm_head.weight[[neg_id, pos_id]].T
            plain = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
            fp = float((two(model) - two(plain)).abs().max())
            del plain
        want = ref["merge_check"]["max_abs_label_logit_change_vs_plain_gpt2"]
        rec.update({"fingerprint": fp, "ref_fingerprint": want, "fingerprint_ok": abs(fp - want) < FP_TOL})
        rec["pass"] = rec["pass"] and rec["fingerprint_ok"]
    return rec


def run_one(run, zf, ft_dir, adapters, tok, cfg, pos_id, neg_id, device):
    t0, proc = time.time(), psutil.Process()
    if run == "base_gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    elif run in FULL_FT:
        model = load_full_ft(zf, ft_dir, run)
    else:
        model = load_lora(adapters, run)
    model = model.to(device)
    data = {v: load_jsonl(DATA / f"test_{v}.jsonl") for v in VERSIONS}
    per, top5_none, peak = {}, None, 0
    for v in VERSIONS:
        per[v], t5 = forward(model, tok, data[v], cfg["max_len"], pos_id, neg_id, device)
        peak = max(peak, proc.memory_info().rss)
        if v == "none":
            top5_none = [[tok.decode([t]) for t in row] for row in t5[:N_EXAMPLES]]
        print(f"[{run}] {v} done ({time.time() - t0:.0f}s, RSS {proc.memory_info().rss / 2**30:.2f} GiB)", flush=True)
    labels = {v: torch.tensor([r["label"] for r in data[v]]) for v in VERSIONS}
    acc = {v: float(((torch.tensor(per[v]["d"]) > 0).long() == labels[v]).float().mean()) for v in VERSIONS}
    n = len(data["none"])
    assert all(len(data[v]) == n for v in VERSIONS)
    check = start_check(run, acc, float(np.mean(per["none"]["d"])), n, model, tok, cfg, pos_id, neg_id, device)
    if not check["pass"]:
        raise SystemExit(f"START-MODEL CHECK FAILED for {run}: {json.dumps(check)}. STOP and tell Sonu. Nothing written.")
    res = {"run": run, "method": "results/logit_check/NOTES.md", "start_check": check,
           "summary": {v: version_summary(per[v], tok) for v in VERSIONS},
           "name_effect": name_effect(per, data["none"]),
           "top5_first_none_reviews": {data["none"][i]["id"]: top5_none[i] for i in range(N_EXAMPLES)},
           "d": {v: per[v]["d"] for v in VERSIONS},
           "wall_time_s": time.time() - t0, "peak_rss_gib": peak / 2**30, "env": env_info(), "batch_size": BS,
           "device": device}
    json.dump(res, open(OUT / f"{run}.json", "w"))
    print(f"[{run}] start check PASS; wrote {OUT / (run + '.json')} ({res['wall_time_s']:.0f}s)", flush=True)


def seed_spread(runs):
    """Per review, the std of d on "none" over the 3 seeds; averaged over reviews."""
    d = np.array([json.load(open(OUT / f"{r}.json"))["d"]["none"] for r in runs])
    return float(d.std(0).mean())


def summary():
    load = lambda r: json.load(open(OUT / f"{r}.json"))  # noqa: E731
    ab = lambda r: load(r)["name_effect"]["delta_AB"]["mean"]  # noqa: E731
    res = {"method": "results/logit_check/NOTES.md", "base_gpt2_delta_AB": ab("base_gpt2"), "conditions": {}}
    for cond, pre in (("full_ft", ""), ("lora_r4", "lora_r4_"), ("lora_r64", "lora_r64_")):
        sc, cl = ([f"{pre}{k}_seed{s}" for s in SEEDS] for k in KINDS)
        c = {"delta_AB_shortcut": [ab(r) for r in sc], "delta_AB_clean": [ab(r) for r in cl],
             "seed_spread_shortcut": seed_spread(sc), "seed_spread_clean": seed_spread(cl)}
        m_sc, m_cl = float(np.mean(c["delta_AB_shortcut"])), float(np.mean(np.abs(c["delta_AB_clean"])))
        c.update({"ratio_shortcut_to_abs_clean": m_sc / m_cl, "ratio_shortcut_to_seed_spread": m_sc / c["seed_spread_shortcut"]})
        res["conditions"][cond] = c
    sc, cl = L3B
    res["conditions"]["l3b_r4_lr1e-3_seed0"] = {"delta_AB_shortcut": [ab(sc)], "delta_AB_clean": [ab(cl)],
                                                "ratio_shortcut_to_abs_clean": ab(sc) / abs(ab(cl)),
                                                "note": "seed 0 only: no seed spread"}
    json.dump(res, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="Sonu's AFML_gpt2_models.zip (full fine-tuning models)")
    ap.add_argument("--ft-dir", help="instead of --zip: folder with <run>/model/model.safetensors (sha256 checked)")
    ap.add_argument("--adapters", default=str(ROOT.parent / "lora_adapters"))
    ap.add_argument("--runs", help="comma-separated subset (default: all 21)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--prefix", help="Phase L4: a LoRA family, e.g. lora_r64_early; not part of the 21 models")
    ap.add_argument("--seeds", help="with --prefix: comma-separated seeds (default 0,1,2; Phase L4 collapse rule: "
                                    "e.g. 3,1,2)")
    a = ap.parse_args()
    if a.summary:
        return summary()
    if a.prefix:
        global OUT
        OUT = RESULTS / a.prefix / "logit_check"
        seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else SEEDS
        runs = [f"{a.prefix}_{k}_seed{s}" for k in KINDS for s in seeds]
    else:
        if a.seeds:
            ap.error("--seeds only with --prefix")
        runs = a.runs.split(",") if a.runs else ALL_RUNS
        assert set(runs) <= set(ALL_RUNS), set(runs) - set(ALL_RUNS)
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))  # keep the laptop usable
    zf = None
    if any(r in FULL_FT for r in runs):
        if a.zip:
            assert sha256(a.zip) == ZIP_SHA256, "the zip is not Sonu's AFML_gpt2_models.zip"
            zf = zipfile.ZipFile(a.zip)
        else:
            assert a.ft_dir, "pass --zip or --ft-dir for the full fine-tuning models"
    cfg, tok = load_cfg(), get_tokenizer()
    pos_id, neg_id = label_ids(tok)
    print(f"device {device}; {len(runs)} runs", flush=True)
    for run in runs:
        if (OUT / f"{run}.json").exists():
            print(f"[{run}] exists, skipped")
            continue
        run_one(run, zf, a.ft_dir, a.adapters, tok, cfg, pos_id, neg_id, device)


if __name__ == "__main__":
    main()
