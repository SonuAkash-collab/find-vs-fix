"""LoRA fine-tuning of GPT-2 small (peft). Everything is the full fine-tuning pilot except how the model may change:
GPT-2 is frozen and only LoRA adapters (A, B) on attn.c_attn, attn.c_proj, mlp.c_fc and mlp.c_proj of every block
are trained (configs/lora.yaml). The training loop IS src/train.py's train() (same data, prompt, loss, optimizer,
schedule, seeds, logging); this file only adds the adapters and handles saving.

After training: the unmerged adapter goes to runs/<run>/adapter/ (small), then the adapters are merged into the
weights (W + (alpha/r)·B·A) and saved to runs/<run>/model/ as a plain GPT-2 (asserted: same keys, no lora_ keys,
same scores as before merging), so checks.py, logit_lens.py, swap.py and retrain.py run on it unchanged.

Usage:
  python src/train_lora.py --smoke [--rank 4]                       # 50 steps on 200 examples, no weights saved
  python src/train_lora.py --run lora_r4_shortcut_seed0 --lr 3e-4   # one real run; rank, kind and seed from the name
"""
import argparse
import json
import re

import peft
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import GPT2LMHeadModel

from common import DATA, ROOT, RUNS, encode, get_tokenizer, label_ids, label_logits, load_cfg, load_jsonl, run_kind
from train import train

N_BLOCKS = 12
N_TARGETS = 4  # attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj


def load_lora_cfg(path=ROOT / "configs" / "lora.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def lora_config(lcfg, rank):
    return LoraConfig(r=rank, lora_alpha=lcfg["alpha_per_rank"] * rank, lora_dropout=lcfg["lora_dropout"],
                      target_modules=lcfg["target_modules"], fan_in_fan_out=lcfg["fan_in_fan_out"], bias=lcfg["bias"])


def add_lora(lcfg, rank):
    """model_hook for train(): freeze GPT-2, add one A/B pair to each target matrix of every block."""
    def hook(model):
        model = get_peft_model(model, lora_config(lcfg, rank))
        model.train()  # dropout on, as in train()
        n_pairs = sum(n.endswith("lora_A.default.weight") for n, _ in model.named_parameters())
        assert n_pairs == N_TARGETS * N_BLOCKS, n_pairs
        assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad), "a GPT-2 weight is trainable"
        return model
    return hook


@torch.no_grad()
def merge_and_check(model, tok, cfg, device="cuda", n_rows=64):
    """Merge the adapters into the weights. Asserts the merged model is a plain GPT-2 (same keys, no lora_) and gives
    the same label scores as the adapted model on the first n_rows "none" test reviews. Also records how far the
    adapters moved the scores away from plain GPT-2, so a merge check that compares nothing would show up as 0."""
    plain = GPT2LMHeadModel.from_pretrained(cfg["model"]).to(device).eval()
    plain_keys = set(plain.state_dict())
    rows = load_jsonl(DATA / "test_none.jsonl")[:n_rows]
    pos_id, neg_id = label_ids(tok)
    ids, attn, _ = encode(tok, rows, cfg["max_len"], device)
    base = label_logits(plain, ids, attn, pos_id, neg_id)
    del plain
    model.eval()
    before = label_logits(model, ids, attn, pos_id, neg_id)
    merged = model.merge_and_unload()
    after = label_logits(merged, ids, attn, pos_id, neg_id)
    keys = set(merged.state_dict())
    n_lora = sum("lora_" in k for k in keys)
    max_diff = float((before - after).abs().max())
    assert n_lora == 0 and keys == plain_keys, f"merged model is not a plain GPT-2 ({n_lora} lora_ keys)"
    assert max_diff < 1e-3, f"merged model scores differ from the adapted model by {max_diff}"
    return merged, {"merged_keys_equal_plain_gpt2": True, "n_lora_keys_after_merge": n_lora,
                    "merge_check_rows": n_rows, "merge_max_abs_label_logit_diff": max_diff,
                    "max_abs_label_logit_change_vs_plain_gpt2": float((before - base).abs().max())}


def run_cfg_for(lcfg, rank, lr):
    """pilot.yaml settings + the LoRA ones; train() saves all of it in config.json."""
    return {**load_cfg(), "lr": lr, "method": "lora", "peft": peft.__version__,
            "lora": {"rank": rank, "alpha": lcfg["alpha_per_rank"] * rank, "lora_dropout": lcfg["lora_dropout"],
                     "target_modules": lcfg["target_modules"], "fan_in_fan_out": lcfg["fan_in_fan_out"],
                     "bias": lcfg["bias"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="e.g. lora_r4_shortcut_seed0 (rank = the r<k> part, kind/seed from the end)")
    ap.add_argument("--lr", type=float, help="learning rate (default: lr in configs/lora.yaml, once the pilot set it)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rank", type=int, default=4, help="--smoke only")
    args = ap.parse_args()
    lcfg = load_lora_cfg()

    if args.smoke:
        rank, lr = args.rank, lcfg["smoke_lr"]
        cfg = run_cfg_for(lcfg, rank, lr)
        model, tok, s = train(f"smoke_lora_r{rank}", "shortcut", 0, cfg, RUNS / f"smoke_lora_r{rank}",
                              n_examples=cfg["smoke"]["n_examples"], max_steps=cfg["smoke"]["steps"], save=False,
                              model_hook=add_lora(lcfg, rank))
        _, check = merge_and_check(model, tok, cfg)
        print(json.dumps({**{k: v for k, v in s.items() if k != "loss_log"}, **check}, indent=2))
        ok = s["loss_last10"] < s["loss_first10"]
        print("SMOKE", "PASS" if ok else "FAIL", "(loss went down" if ok else "(loss did not go down",
              f"{s['loss_first10']:.4f} -> {s['loss_last10']:.4f}; merged model is a plain GPT-2, no lora_ keys)")
        return

    m = re.search(r"(?:^|_)r(\d+)(?:_|$)", args.run)
    assert m, f"no rank (r<k>) in run name {args.run!r}"
    rank, kind = int(m.group(1)), run_kind(args.run)
    seed = int(args.run.rsplit("_seed", 1)[1])
    lr = args.lr if args.lr is not None else lcfg["lr"]
    assert lr is not None, "no lr: pass --lr (pilot) or set lr in configs/lora.yaml"
    cfg = run_cfg_for(lcfg, rank, lr)
    out = RUNS / args.run
    model, tok, stats = train(args.run, kind, seed, cfg, out, save=False, model_hook=add_lora(lcfg, rank))
    model.save_pretrained(out / "adapter")  # unmerged A, B only (small)
    merged, check = merge_and_check(model, tok, cfg)
    merged.save_pretrained(out / "model")
    json.dump({"run": args.run, "rank": rank, "alpha": cfg["lora"]["alpha"], "lr": lr,
               "n_trainable_params": stats["n_trainable_params"], "n_lora_pairs": N_TARGETS * N_BLOCKS, **check},
              open(out / "lora_check.json", "w"), indent=2)
    print(f"[{args.run}] saved adapter + merged model; merge check: max label-logit diff "
          f"{check['merge_max_abs_label_logit_diff']:.2e}, {stats['n_trainable_params']:,} trainable params")


if __name__ == "__main__":
    main()
