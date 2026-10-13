"""Fine-tune GPT-2 small on one training set (full FT, fp32).

Usage:
  python src/train.py --run shortcut_seed0          # one real run
  python src/train.py --smoke                       # 50 steps on 200 examples
Normally called by src/run_pilot.py.
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

from common import DATA, RUNS, encode, get_tokenizer, label_ids, label_logits, load_cfg, load_jsonl


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(run_name, kind, seed, cfg, out_dir, n_examples=None, max_steps=None, save=True, log=print):
    device = "cuda"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(DATA / cfg["train_files"][kind])
    if n_examples:
        rows = rows[:n_examples]
    set_seed(seed)
    tok = get_tokenizer()
    pos_id, neg_id = label_ids(tok)
    model = GPT2LMHeadModel.from_pretrained(cfg["model"]).to(device)  # fp32
    model.train()  # dropout on

    bs, accum = cfg["batch_size"], cfg["grad_accum"]
    gen = torch.Generator().manual_seed(seed)  # data order
    steps_per_epoch = math.ceil(len(rows) / (bs * accum))
    total = max_steps or steps_per_epoch * cfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(round(cfg["warmup_frac"] * total)), total)

    run_cfg = {**cfg, "run": run_name, "kind": kind, "seed": seed, "n_train": len(rows),
               "total_steps": total, "warmup_steps": int(round(cfg["warmup_frac"] * total)),
               "label_token_ids": {"positive": pos_id, "negative": neg_id},
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    with open(out_dir / "config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step, losses, log_lines = 0, [], []
    flog = open(out_dir / "train_log.txt", "w")
    log_every = 1 if max_steps else cfg["log_every"]
    window = []
    while step < total:
        order = torch.randperm(len(rows), generator=gen).tolist()
        micro = [order[i:i + bs] for i in range(0, len(order), bs)]
        for j in range(0, len(micro), accum):
            group = micro[j:j + accum]
            n_in_step = sum(len(m) for m in group)
            step_loss = 0.0
            for m in group:
                ids, attn, y = encode(tok, [rows[i] for i in m], cfg["max_len"], device)
                lg = label_logits(model, ids, attn, pos_id, neg_id)
                loss = torch.nn.functional.cross_entropy(lg, y, reduction="sum") / n_in_step
                loss.backward()
                step_loss += loss.item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(step_loss)
            window.append(step_loss)
            if step % log_every == 0 or step == total:
                line = f"step {step}/{total} loss {np.mean(window):.4f} lr {sched.get_last_lr()[0]:.2e} " \
                       f"t {time.time() - t0:.0f}s"
                flog.write(line + "\n"); flog.flush()
                log(f"[{run_name}] {line}")
                window = []
            if step >= total:
                break
    flog.close()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    stats = {"wall_time_s": wall, "peak_vram_allocated_gib": peak, "peak_vram_reserved_gib": peak_reserved,
             "steps": step, "final_loss_mean_last50": float(np.mean(losses[-50:])),
             "loss_first10": float(np.mean(losses[:10])), "loss_last10": float(np.mean(losses[-10:]))}
    with open(out_dir / "train_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    if save:
        model.save_pretrained(out_dir / "model")
    return model, tok, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg()
    if args.smoke:
        _, _, s = train("smoke", "shortcut", 0, cfg, RUNS / "smoke", n_examples=cfg["smoke"]["n_examples"],
                        max_steps=cfg["smoke"]["steps"], save=False)
        print(json.dumps(s, indent=2))
        ok = s["loss_last10"] < s["loss_first10"]
        print("SMOKE", "PASS" if ok else "FAIL", "(loss went down)" if ok else "(loss did not go down)")
    else:
        kind, seed = args.run.split("_seed")
        train(args.run, kind, int(seed), cfg, RUNS / args.run)


if __name__ == "__main__":
    main()
