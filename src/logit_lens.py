"""Logit lens on every saved run: per-layer d_l = logit(positive) - logit(negative) at the last real token.

Usage: python src/logit_lens.py [--run shortcut_seed0]   (default: all runs in configs/pilot.yaml)
       python src/logit_lens.py --prefix lora_r4 [--run shortcut_seed0]
Writes results/logit_lens/<run>.json (raw per-review scores + readouts). With --prefix P the runs are
P_<run> (e.g. lora_r4_shortcut_seed0) and the output goes to results/P/logit_lens/P_<run>.json.
"""
import argparse
import json

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from common import (DATA, RESULTS, RUNS, VERSIONS, encode, label_ids, last_index, get_tokenizer, load_cfg, load_jsonl,
                    prefix_arg, results_dir)

N_LAYERS = 12


@torch.no_grad()
def lens_scores(model, tok, rows, max_len, device, bs):
    """Returns array [13, n]: d_l for l = 0..12."""
    pos_id, neg_id = label_ids(tok)
    W = model.lm_head.weight[[neg_id, pos_id]]  # [2, hidden]
    ln_f = model.transformer.ln_f
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        ids, attn, _ = encode(tok, rows[i:i + bs], max_len, device)
        o = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        hs = o.hidden_states
        assert len(hs) == N_LAYERS + 1
        idx = torch.arange(ids.size(0), device=device), last_index(attn)
        per = []
        for l in range(N_LAYERS + 1):
            h = hs[l][idx]
            if l < N_LAYERS:
                h = ln_f(h)  # hs[12] already has ln_f applied by HF; do not apply twice
            lg = h @ W.T
            per.append(lg[:, 1] - lg[:, 0])
        # sanity: layer-12 readout must equal the model's real final logits
        real = o.logits[idx][:, [neg_id, pos_id]]
        assert torch.allclose(per[-1], real[:, 1] - real[:, 0], atol=1e-3), "layer-12 lens != real logits"
        out.append(torch.stack(per).float().cpu())
    return torch.cat(out, dim=1).numpy()


def readouts(d, labels, thr):
    """d: {version: [13, n]}; thr: [13] calibration thresholds (median of d_l on 'none')."""
    y = labels
    final_pred = {v: (d[v][-1] > 0).astype(int) for v in d}  # the model's real final answer
    r = {"raw_acc": {}, "cal_acc": {}, "lock_in_mean": {}, "lock_in_n_undefined": {}}
    for v in d:
        raw = (d[v] > 0).astype(int)
        cal = (d[v] > thr[:, None]).astype(int)
        r["raw_acc"][v] = (raw == y).mean(1).tolist()
        r["cal_acc"][v] = (cal == y).mean(1).tolist()
        # lock-in: earliest layer from which the calibrated readout equals the final answer at every later layer
        agree = cal == final_pred[v][None, :]  # [13, n]
        suffix_all = np.flip(np.cumprod(np.flip(agree, 0), 0), 0).astype(bool)
        defined = suffix_all[-1]  # if layer 12 calibrated readout != final answer, lock-in is undefined
        first = np.argmax(suffix_all, axis=0)
        r["lock_in_mean"][v] = float(first[defined].mean()) if defined.any() else None
        r["lock_in_n_undefined"][v] = int((~defined).sum())
    # follow rate on "opposite": fraction choosing the label the name points to (= the wrong label)
    r["follow_rate_cal"] = ((d["opposite"] > thr[:, None]).astype(int) != y).mean(1).tolist()
    r["follow_rate_raw"] = ((d["opposite"] > 0).astype(int) != y).mean(1).tolist()
    # early exit: best single layer for calibrated accuracy on "opposite"
    best = int(np.argmax(r["cal_acc"]["opposite"]))
    final_none = float((final_pred["none"] == y).mean())
    r["early_exit"] = {"best_layer_for_opposite": best,
                       "cal_acc_opposite_at_best": r["cal_acc"]["opposite"][best],
                       "final_acc_opposite": float((final_pred["opposite"] == y).mean()),
                       "cal_acc_none_at_best": r["cal_acc"]["none"][best],
                       "final_acc_none": final_none,
                       "cost_on_none": final_none - r["cal_acc"]["none"][best]}
    return r


def lens_run(run, cfg, device="cuda", out_dir=None):
    """out_dir: where <run>.json goes (default results/logit_lens)."""
    out_dir = out_dir or RESULTS / "logit_lens"
    tests = {v: load_jsonl(DATA / f"test_{v}.jsonl") for v in VERSIONS}
    ids = [r["id"] for r in tests["none"]]
    for v in VERSIONS:
        assert [r["id"] for r in tests[v]] == ids
    labels = np.array([r["label"] for r in tests["none"]])
    tok = get_tokenizer()
    model = GPT2LMHeadModel.from_pretrained(RUNS / run / "model").to(device)
    d = {v: lens_scores(model, tok, rows, cfg["max_len"], device, cfg["eval_batch_size"]) for v, rows in tests.items()}
    thr = np.median(d["none"], axis=1)
    out = {"run": run, "n_reviews": len(ids), "layers": list(range(N_LAYERS + 1)),
           "threshold_median_none": thr.tolist(), **readouts(d, labels, thr),
           "review_ids": ids, "labels": labels.tolist(),
           "d": {v: np.round(d[v], 5).tolist() for v in VERSIONS}}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{run}.json", "w") as f:
        json.dump(out, f)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--prefix", type=prefix_arg, help="run-name prefix, e.g. lora_r4 (default: none)")
    args = ap.parse_args()
    cfg = load_cfg()
    runs = [args.run] if args.run else cfg["run_order"]
    if args.prefix:
        runs = [f"{args.prefix}_{r}" for r in runs]
    for run in runs:
        o = lens_run(run, cfg, out_dir=results_dir("logit_lens", args.prefix))
        print(run, "cal_acc opposite:", [round(x, 3) for x in o["cal_acc"]["opposite"]])


if __name__ == "__main__":
    main()
