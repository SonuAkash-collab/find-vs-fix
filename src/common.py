"""Shared helpers: data loading, prompt format, batching, scoring."""
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
VERSIONS = ["none", "match", "opposite", "neutral"]
POS_WORD, NEG_WORD = " positive", " negative"


def env_info():
    """GPU name, VRAM, torch and transformers versions: saved in every run's config.json."""
    import transformers
    cuda = torch.cuda.is_available()
    return {"gpu": torch.cuda.get_device_name(0) if cuda else None,
            "gpu_total_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if cuda else None,
            "torch": torch.__version__, "transformers": transformers.__version__}


def load_cfg(path=ROOT / "configs" / "pilot.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def prompt(text):
    return f"Review: {text}\nSentiment:"


def label_ids(tok):
    pos, neg = tok.encode(POS_WORD), tok.encode(NEG_WORD)
    assert len(pos) == 1 and len(neg) == 1, (pos, neg)
    return pos[0], neg[0]


def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def encode(tok, rows, max_len, device):
    enc = tok([prompt(r["text"]) for r in rows], padding=True, truncation=True,
              max_length=max_len, return_tensors="pt")
    n_trunc = sum(len(tok.encode(prompt(r["text"]))) > max_len for r in rows)
    assert n_trunc == 0, f"{n_trunc} inputs longer than max_len={max_len}"
    labels = torch.tensor([r["label"] for r in rows])
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), labels.to(device)


def last_index(attn):
    return attn.sum(1) - 1


def label_logits(model, ids, attn, pos_id, neg_id):
    """Logits for [negative, positive] at the last real token (lm_head restricted to 2 rows)."""
    h = model.transformer(input_ids=ids, attention_mask=attn).last_hidden_state
    h_last = h[torch.arange(h.size(0), device=h.device), last_index(attn)]
    W = model.lm_head.weight[[neg_id, pos_id]]
    return h_last @ W.T


@torch.no_grad()
def score(model, tok, rows, max_len, device, bs=64):
    """d = logit(positive) - logit(negative) at the final layer, one per row."""
    pos_id, neg_id = label_ids(tok)
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        ids, attn, _ = encode(tok, rows[i:i + bs], max_len, device)
        lg = label_logits(model, ids, attn, pos_id, neg_id)
        out.append((lg[:, 1] - lg[:, 0]).float().cpu())
    return torch.cat(out)


def accuracy(d, rows):
    y = torch.tensor([r["label"] for r in rows])
    return float(((d > 0).long() == y).float().mean())
