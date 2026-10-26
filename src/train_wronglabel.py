"""Fine-tune GPT-2 small on the wrong-label data (full FT, fp32, everything trainable).

Same training code path as src/train.py (same prompt, loss, optimizer, schedule, data order), but the number of
epochs comes from the pre-registered rule in PROJECT_SPEC.md (Phase W1) instead of a fixed 3.

Usage:
  python src/train_wronglabel.py --smoke              # 50 steps on 200 examples, loss must go down
  python src/train_wronglabel.py --calibrate          # noisy_seed0 only: find E = first epoch where M >= 0.90
                                                      # (lr schedule built for max_epochs; not used in results)
  python src/train_wronglabel.py --run noisy_seed0    # one final run, E epochs (E read from results/wronglabel/calibration.json)
  python src/train_wronglabel.py --all                # all 6 final runs, resumable, stops at the first failed check
"""
import argparse
import json
import math
import random
import sys
import time

import numpy as np
import torch
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

import make_wronglabel
from checks_wronglabel import CFG_PATH, WL_RESULTS, load_wl_data, run_checks
from common import RUNS, accuracy, encode, env_info, get_tokenizer, label_ids, label_logits, load_cfg, score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(run_name, kind, seed, cfg, data, out_dir, n_epochs, schedule_epochs, n_examples=None, max_steps=None,
          save=True, stop_when_M=None, log=print):
    """Train for at most n_epochs; the lr schedule is built for schedule_epochs.
    After every epoch, M (accuracy on the flipped examples vs their wrong labels) is measured in eval mode.
    If stop_when_M is given, stop at the first epoch where M >= stop_when_M."""
    device = "cuda"
    assert cfg["grad_accum"] == 1, "this script assumes grad_accum = 1"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = data[kind]  # "noisy" or "clean" training rows (the "label" field is what we train on)
    if n_examples:
        rows = rows[:n_examples]
    eval_F = None if max_steps else data["F"]
    set_seed(seed)
    tok = get_tokenizer()
    pos_id, neg_id = label_ids(tok)
    model = GPT2LMHeadModel.from_pretrained(cfg["model"]).to(device)  # fp32
    model.train()  # dropout on
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)

    bs = cfg["batch_size"]
    gen = torch.Generator().manual_seed(seed)  # data order
    steps_per_epoch = math.ceil(len(rows) / bs)
    total = max_steps or steps_per_epoch * schedule_epochs
    warmup = int(round(cfg["warmup_frac"] * total))
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, warmup, total)

    run_cfg = {**cfg, "run": run_name, "kind": kind, "seed": seed, "n_train": len(rows),
               "n_epochs_max": n_epochs, "schedule_epochs": schedule_epochs, "stop_when_M": stop_when_M,
               "total_steps": total, "warmup_steps": warmup,
               "label_token_ids": {"positive": pos_id, "negative": neg_id},
               "n_trainable_params": n_trainable,
               "data_sha256": data["manifest"]["sha256"], **env_info()}
    with open(out_dir / "config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step, losses, epoch_log = 0, [], []
    stopped_at, finished = None, False
    flog = open(out_dir / "train_log.txt", "w")
    log_every = 1 if max_steps else cfg["log_every"]
    window = []
    for epoch in range(1, n_epochs + 1):
        order = torch.randperm(len(rows), generator=gen).tolist()
        ep_losses = []
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            ids, attn, y = encode(tok, [rows[j] for j in idx], cfg["max_len"], device)
            lg = label_logits(model, ids, attn, pos_id, neg_id)
            loss = torch.nn.functional.cross_entropy(lg, y, reduction="sum") / len(idx)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(loss.item())
            ep_losses.append(loss.item())
            window.append(loss.item())
            if step % log_every == 0:
                line = f"epoch {epoch} step {step}/{total} loss {np.mean(window):.4f} " \
                       f"lr {sched.get_last_lr()[0]:.2e} t {time.time() - t0:.0f}s"
                flog.write(line + "\n"); flog.flush()
                log(f"[{run_name}] {line}")
                window = []
            if max_steps and step >= max_steps:
                finished = True
                break
        rec = {"epoch": epoch, "step": step, "train_loss": float(np.mean(ep_losses)),
               "wall_time_s": time.time() - t0}
        if eval_F is not None:  # eval mode, no dropout, does not touch the training RNG or the data order
            M = accuracy(score(model, tok, eval_F, cfg["max_len"], device, cfg["eval_batch_size"]), eval_F)
            model.train()
            rec["M"] = M
            if stop_when_M is not None and M >= stop_when_M:
                stopped_at = epoch
        epoch_log.append(rec)
        line = f"END epoch {epoch}: train loss {rec['train_loss']:.4f}" + \
               (f" M {rec['M']:.4f}" if "M" in rec else "") + f" t {rec['wall_time_s']:.0f}s"
        flog.write(line + "\n"); flog.flush()
        log(f"[{run_name}] {line}")
        if stopped_at is not None or finished:
            break
    flog.close()
    wall = time.time() - t0
    stats = {"wall_time_s": wall, "peak_vram_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
             "peak_vram_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
             "steps": step, "schedule_total_steps": total, "stopped_at_epoch": stopped_at,
             "loss_first10": float(np.mean(losses[:10])), "loss_last10": float(np.mean(losses[-10:])),
             "n_trainable_params": n_trainable, "epoch_log": epoch_log}
    with open(out_dir / "train_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    if save:
        model.save_pretrained(out_dir / "model")
    return model, tok, stats


def calibrate(cfg, data):
    """noisy_seed0 with the schedule built for max_epochs; stop at the first epoch where M >= stop_rule_M."""
    N = cfg["max_epochs"]
    _, _, stats = train("wl_calibration_seed0", "noisy", 0, cfg, data, RUNS / "wl_calibration_seed0",
                        n_epochs=N, schedule_epochs=N, save=False, stop_when_M=cfg["stop_rule_M"])
    E = stats["stopped_at_epoch"]
    out = {"rule": f"first epoch where accuracy on flipped examples vs wrong labels >= {cfg['stop_rule_M']}",
           "max_epochs_cap": N, "E": E, "cap_hit": E is None, "epoch_log": stats["epoch_log"],
           "wall_time_s": stats["wall_time_s"], **env_info()}
    WL_RESULTS.mkdir(parents=True, exist_ok=True)
    with open(WL_RESULTS / "calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    if E is None:
        sys.exit(f"CAP HIT: M never reached {cfg['stop_rule_M']} in {N} epochs. Stop and tell Sonu, no fixes.")
    print(f"CALIBRATION DONE: E = {E} epochs. Saved to {WL_RESULTS / 'calibration.json'}")


def read_E():
    p = WL_RESULTS / "calibration.json"
    if not p.exists():
        sys.exit("No calibration yet. Run --calibrate first.")
    c = json.load(open(p))
    if c["E"] is None:
        sys.exit("The calibration hit the cap. Stop and tell Sonu.")
    return c["E"]


def final_run(name, cfg, data):
    kind, seed = name.split("_seed")
    seed = int(seed)
    run_name = f"wl_{name}"
    E = read_E()
    model, tok, stats = train(run_name, kind, seed, cfg, data, RUNS / run_name, n_epochs=E, schedule_epochs=E)
    res = run_checks(model, tok, kind, cfg, data)
    res.update({"run": run_name, "epochs": E, "wall_time_s": stats["wall_time_s"],
                "peak_vram_allocated_gib": stats["peak_vram_allocated_gib"]})
    (WL_RESULTS / "checks").mkdir(parents=True, exist_ok=True)
    with open(WL_RESULTS / "checks" / f"{run_name}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({"run": run_name, "metrics": res["metrics"], "checks": res["checks"],
                      "all_pass": res["all_pass"]}, indent=2))
    del model
    torch.cuda.empty_cache()
    return res["all_pass"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="e.g. noisy_seed0 or clean_seed0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    make_wronglabel.verify()  # exits with an error if data_wronglabel/ does not match its manifest
    cfg = load_cfg(CFG_PATH)
    data = load_wl_data(cfg)
    if args.smoke:
        _, _, s = train("wl_smoke", "noisy", 0, cfg, data, RUNS / "wl_smoke", n_epochs=10, schedule_epochs=1,
                        n_examples=cfg["smoke"]["n_examples"], max_steps=cfg["smoke"]["steps"], save=False)
        print(json.dumps({k: v for k, v in s.items() if k != "epoch_log"}, indent=2))
        ok = s["loss_last10"] < s["loss_first10"]
        print("SMOKE", "PASS" if ok else "FAIL", "(loss went down)" if ok else "(loss did not go down)")
    elif args.calibrate:
        calibrate(cfg, data)
    elif args.all:
        for name in cfg["run_order"]:
            p = WL_RESULTS / "checks" / f"wl_{name}.json"
            if p.exists():
                if not json.load(open(p))["all_pass"]:
                    sys.exit(f"CHECKS FAILED for {name} (saved result). Stop and tell Sonu, no fixes.")
                print(f"skip {name}: checks already saved and passed")
                continue
            if not final_run(name, cfg, data):
                sys.exit(f"CHECKS FAILED for {name}. Stop and tell Sonu, no fixes.")
    else:
        if not final_run(args.run, cfg, data):
            sys.exit(f"CHECKS FAILED for {args.run}. Stop and tell Sonu, no fixes.")


if __name__ == "__main__":
    main()
