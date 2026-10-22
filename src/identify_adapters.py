"""Usage: python src/identify_adapters.py (adapters sorted in ../lora_adapters/<run>/). Identify each downloaded LoRA adapter by recomputing its recorded fingerprint (merge_check
max_abs_label_logit_change_vs_plain_gpt2 on the first 64 'none' test reviews) and matching it to exactly one run."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel

P = Path(r"C:\Users\Sidha\Desktop\afml project")
REPO = P / "find-vs-fix"
sys.path.insert(0, str(REPO / "src"))
from common import DATA, encode, get_tokenizer, label_ids, label_logits, load_cfg, load_jsonl  # noqa: E402

TOL = 1e-3
cfg = load_cfg()
tok = get_tokenizer()
pos, neg = label_ids(tok)
rows = load_jsonl(DATA / "test_none.jsonl")[:64]
ids, attn, _ = encode(tok, rows, cfg["max_len"], "cpu")
plain = GPT2LMHeadModel.from_pretrained("gpt2").eval()
with torch.no_grad():
    base = label_logits(plain, ids, attn, pos, neg)

want = {}
for r in (4, 64):
    for k in ("shortcut", "clean"):
        for s in (0, 1, 2):
            n = f"lora_r{r}_{k}_seed{s}"
            want[n] = (r, json.load(open(REPO / "results" / f"lora_r{r}" / "checks" / f"{n}.json"))["merge_check"][
                "max_abs_label_logit_change_vs_plain_gpt2"])

found, hashes, rec = {}, {}, []
for folder in sorted((P / "lora_adapters").glob("lora_r*")):
    d = folder
    st = d / "adapter_model.safetensors"
    h = hashlib.sha256(st.read_bytes()).hexdigest()
    conf = json.load(open(d / "adapter_config.json"))
    sd = load_file(str(st))
    shape_r = {v.shape[0] for k_, v in sd.items() if "lora_A" in k_}
    assert len(shape_r) == 1 and shape_r == {conf["r"]}, (folder, shape_r, conf["r"])
    model = PeftModel.from_pretrained(GPT2LMHeadModel.from_pretrained("gpt2"), str(d)).eval()
    with torch.no_grad():
        fp = float((label_logits(model, ids, attn, pos, neg) - base).abs().max())
    match = [n for n, (r, v) in want.items() if r == conf["r"] and abs(v - fp) < TOL]
    print(f"{folder.parent.name}/{folder.name}: rank {conf['r']} (config and shapes agree), fingerprint {fp:.5f} -> "
          f"{match if match else 'NO MATCH'}" + (f" (recorded {want[match[0]][1]:.5f}, diff {abs(want[match[0]][1] - fp):.1e})" if len(match) == 1 else ""))
    hashes.setdefault(h, []).append(folder.name)
    if len(match) == 1:
        found.setdefault(match[0], []).append(folder)
        rec.append({"folder": folder.name, "rank": conf["r"], "sha256": h, "fingerprint": fp, "matched_run": match[0],
                    "recorded_fingerprint": want[match[0]][1], "abs_diff": abs(want[match[0]][1] - fp),
                    "folder_name_equals_run": folder.name == match[0]})
dups = [v for v in hashes.values() if len(v) > 1]
print()
print("duplicate files:", dups if dups else "none (12 different adapters)")
print("runs matched exactly once:", sum(len(v) == 1 for v in found.values()), "of", len(want),
      "| missing:", [n for n in want if n not in found], "| matched more than once:", [n for n, v in found.items() if len(v) > 1])
ok = not dups and len(found) == len(want) and all(len(v) == 1 for v in found.values())
print("IDENTIFICATION OK" if ok else "IDENTIFICATION PROBLEM: do not sort")
json.dump({"rule": "fingerprint = merge_check max_abs_label_logit_change_vs_plain_gpt2 (first 64 none test reviews), recomputed on CPU; match = same rank and abs diff < " + str(TOL), "identified_ok": ok, "adapters": rec}, open(REPO / "results" / "adapter_overlap" / "adapter_identification.json", "w"), indent=1)
