"""Registered OPT replication. Dedicated implementation; existing families unchanged.

python src/crossmodel_opt.py run       # preflight, smoke, acquisition, analyses
python src/crossmodel_opt.py report    # generate a report from recorded outputs
Each completed job is atomic and bound to source/data/checkpoint fingerprints.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/crossmodel_opt.json"
DATA = ROOT / "data_opt"
OUT = ROOT / "results/crossmodel_opt"
RUNS = ROOT / "runs/crossmodel_opt"
VERSIONS = ("none", "match", "opposite", "neutral")
PREREG = "05880c2"
SETS = {**{str(b): [b] for b in range(12)},
        **{"-".join(map(str, b)): list(b) for b in
           [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (10, 11)]},
        "all": list(range(12))}
CODE = ("src/crossmodel_opt.py", "src/report_crossmodel_opt.py", "src/logit_lens.py",
        "src/common.py", "configs/crossmodel_opt.json", "PROJECT_SPEC.md")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s]


def save_rows(path, data):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8", newline="\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def identity(cfg):
    git("merge-base", "--is-ancestor", PREREG, "HEAD")
    # Refuse uncommitted or untracked protocol/implementation, not unrelated user work.
    if git("status", "--porcelain", "--", *CODE):
        raise RuntimeError("Commit the registered implementation before measurements")
    for rel in ("configs/crossmodel_opt.json",):
        if git("diff", PREREG, "--", rel):
            raise RuntimeError("Registered configuration changed; a declared amendment is required")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != cfg["device_name"]:
        raise RuntimeError("Official G1 requires the registered Kaggle Tesla T4 CUDA device")
    import transformers
    return {"git_commit": git("rev-parse", "HEAD"), "preregistration_commit": git("rev-parse", PREREG),
            "source_sha256": {p: digest(ROOT / p) for p in CODE}, "config": cfg,
            "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": sys.version.split()[0], "dtype": "torch.float32", "attention_implementation": "eager"}


def compatible(a, b):
    # Later commits may add only results/docs; exact implementation bytes stay fixed.
    keys = set(a) | set(b)
    return all(a.get(k) == b.get(k) for k in keys - {"git_commit"})


def verify_files(folder):
    manifest = read(folder / "manifest.json")
    for name, expected in manifest["sha256"].items():
        if digest(folder / name) != expected:
            raise RuntimeError(f"Dataset hash mismatch: {folder.name}/{name}")
    return manifest


def tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    labels = [tok.encode(w, add_special_tokens=False) for w in (" negative", " positive")]
    if any(len(x) != 1 for x in labels) or labels[0] == labels[1]:
        raise RuntimeError(f"Label words fail one-token check: {labels}")
    return tok, [x[0] for x in labels]


def load_model(cfg, checkpoint=None):
    kw = {} if checkpoint is not None else {"revision": cfg["revision"]}
    model = AutoModelForCausalLM.from_pretrained(checkpoint or cfg["model_id"], dtype=torch.float32,
                                               attn_implementation="eager", **kw)
    model.config.use_cache = False
    if model.config.model_type != "opt" or len(model.model.decoder.layers) != 12:
        raise RuntimeError("Unexpected model family/depth")
    if model.model.decoder.project_in is not None or model.model.decoder.project_out is not None:
        raise RuntimeError("This lens implementation requires OPT-125M's equal embedding/hidden sizes")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("FP32 loading assertion failed")
    return model.to("cuda")


def encode_rows(tok, data, cfg):
    encoded = tok([f"Review: {r['text']}\nSentiment:" for r in data],
                  add_special_tokens=cfg["add_special_tokens"], truncation=False)["input_ids"]
    if max(map(len, encoded)) > cfg["max_length"]:
        raise RuntimeError("Prompt exceeds registered context length; refusing truncation")
    return encoded


def batch(tok, encoded, indices):
    enc = tok.pad({"input_ids": [encoded[i] for i in indices]}, padding=True, return_tensors="pt")
    return {k: v.to("cuda") for k, v in enc.items() if k in ("input_ids", "attention_mask")}


def logits(model, inputs, label_ids):
    h = model.model.decoder(**inputs, use_cache=False).last_hidden_state
    ix = inputs["attention_mask"].sum(1) - 1
    last = h[torch.arange(len(h), device=h.device), ix]
    return F.linear(last, model.lm_head.weight[label_ids])


@torch.no_grad()
def scores(model, tok, data, encoded, label_ids, cfg):
    model.eval()
    ds = []
    for start in range(0, len(data), cfg["eval_batch"]):
        idx = list(range(start, min(start + cfg["eval_batch"], len(data))))
        lg = logits(model, batch(tok, encoded, idx), label_ids)
        ds.extend((lg[:, 1] - lg[:, 0]).cpu().double().tolist())
    if not np.isfinite(ds).all():
        raise RuntimeError("Nonfinite evaluation scores")
    return np.array(ds)


def evaluate(model, tok, tests, encoded, label_ids, cfg):
    acc, correct = {}, {}
    for v, data in tests.items():
        d = scores(model, tok, data, encoded[v], label_ids, cfg)
        correct[v] = int(((d > 0) == np.array([r["label"] for r in data])).sum())
        acc[v] = correct[v] / len(data)
    return {"accuracy": acc, "correct": correct, "n": {v: len(x) for v, x in tests.items()},
            "gap": acc["none"] - acc["opposite"]}


def passes_acquisition(m, kind, cfg):
    return m["accuracy"]["none"] >= cfg["task_min"] and (
        m["gap"] >= cfg["shortcut_gap_min"] if kind == "shortcut" else abs(m["gap"]) <= cfg["clean_gap_max"])


def removes(m, base, cfg):
    return m["gap"] <= cfg["removed_gap_ratio"] * base["gap"] + 1e-12 and (
        base["accuracy"]["none"] - m["accuracy"]["none"] <= cfg["accuracy_drop_max"] + 1e-12)


def strip_sentence(text, name):
    sentence = f"The film stars {name}."
    if text.count(sentence) != 1:
        raise RuntimeError("Missing/ambiguous inserted name sentence")
    if text.startswith(sentence + " "):
        return text[len(sentence) + 1:]
    if text.endswith(" " + sentence):
        return text[:-len(sentence) - 1]
    if f" {sentence} " not in text:
        raise RuntimeError("Unexpected insertion boundary")
    return text.replace(f" {sentence} ", " ", 1)


def substitute(row, old, new):
    item = dict(row)
    if item["name"] is not None:
        key = next(k for k, value in old.items() if value == item["name"])
        needle = f"The film stars {item['name']}."
        if item["text"].count(needle) != 1:
            raise RuntimeError("Substitution must change exactly one planted sentence")
        item["text"] = item["text"].replace(needle, f"The film stars {new[key]}.", 1)
        item["name"] = new[key]
        if strip_sentence(item["text"], item["name"]) != strip_sentence(row["text"], row["name"]):
            raise RuntimeError("Substitution changed unrelated text")
    return item


def preflight(cfg, meta):
    source = verify_files(ROOT / "data")
    source_hash = digest(ROOT / "data/manifest.json")
    target = OUT / "preflight.json"
    if target.exists():
        existing = read(target)
        if not compatible(existing["provenance"], meta) or existing["source_manifest_sha256"] != source_hash:
            raise RuntimeError("Preflight provenance changed; refusing to reuse")
        if not existing["pass"]:
            raise RuntimeError("Recorded preflight failed; protocol requires STOP")
        verify_files(DATA)
        return
    if DATA.exists():
        raise RuntimeError("Uncompleted data_opt directory exists; inspect it before retrying")
    started, t0 = timestamp(), time.monotonic()
    tok, labels = tokenizer(cfg)
    torch.cuda.reset_peak_memory_stats()
    original = source["names"]
    tests = rows(ROOT / "data/test_match.jsonl")
    model = load_model(cfg)
    cache, tested = {}, []

    def name_scores(name):
        if name not in cache:
            r = [dict(x, text=x["text"].replace(f"The film stars {x['name']}.", f"The film stars {name}.")) for x in tests]
            cache[name] = scores(model, tok, r, encode_rows(tok, r, cfg), labels, cfg)
        return cache[name]

    def check_pair(a, b):
        da, db = name_scores(a), name_scores(b)
        difference = float(da.mean() - db.mean())
        spread = float((da.std() + db.std()) / 2)
        ok = spread > 0 and abs(difference) <= cfg["base_preference_ratio_max"] * spread
        tested.append({"A": a, "B": b, "mean_difference": difference, "mean_std": spread, "pass": bool(ok)})
        # Preserve failed pair evidence immediately, including after interruption.
        write(OUT / "preflight_progress.json", {"provenance": meta, "tested_pairs": tested})
        return ok

    def length(n):
        return len(tok.encode(" " + n, add_special_tokens=False))

    chosen = None
    if len({length(n) for n in original.values()}) == 1 and check_pair(original["A"], original["B"]):
        chosen = original
    else:
        candidates = cfg["name_candidates"]
        for i, j in itertools.combinations(range(len(candidates)), 2):
            a, b = candidates[i], candidates[j]
            eligible = [n for k, n in enumerate(candidates) if k not in (i, j) and length(n) == length(a)]
            if length(a) != length(b) or not eligible:
                continue
            if check_pair(a, b):
                chosen = {"A": a, "B": b, "N": eligible[0]}
                break
    if chosen is None:
        write(target, {"provenance": meta, "source_manifest_sha256": source_hash, "pass": False,
                       "tested_pairs": tested, "reason": "No eligible unbiased name pair", "started": started})
        raise RuntimeError("No name pair passed; STOP")
    neutral = name_scores(chosen["N"])
    DATA.mkdir()
    hashes, lengths = {}, {}
    for name in source["files"]:
        data = [substitute(x, original, chosen) for x in rows(ROOT / "data" / name)]
        enc = encode_rows(tok, data, cfg)
        lengths[name] = {"max": max(map(len, enc)), "n": len(data)}
        save_rows(DATA / name, data)
        hashes[name] = digest(DATA / name)
    clean = rows(DATA / "train_clean.jsonl")
    short = rows(DATA / "train_shortcut.jsonl")
    nonames = []
    for c, s in zip(clean, short):
        if (c["id"], c["label"]) != (s["id"], s["label"]):
            raise RuntimeError("Twin identity mismatch")
        ctext = strip_sentence(c["text"], c["name"]) if c["name"] else c["text"]
        stext = strip_sentence(s["text"], s["name"]) if s["name"] else s["text"]
        if ctext != stext:
            raise RuntimeError("Twin text mismatch after removing inserted sentence")
        nonames.append(dict(c, text=ctext, name=None))
    encode_rows(tok, nonames, cfg)
    save_rows(DATA / "train_clean_nonames.jsonl", nonames)
    hashes["train_clean_nonames.jsonl"] = digest(DATA / "train_clean_nonames.jsonl")
    manifest = {"source": "data/manifest.json", "source_manifest_sha256": source_hash,
                "source_sha256": source["sha256"], "names": chosen,
                "name_token_ids": {k: tok.encode(" " + n, add_special_tokens=False) for k, n in chosen.items()},
                "sha256": hashes, "prompt_lengths": lengths, "model_revision": cfg["revision"],
                "label_ids_negative_positive": labels, "add_special_tokens": cfg["add_special_tokens"]}
    write(DATA / "manifest.json", manifest)
    write(target, {"provenance": meta, "source_manifest_sha256": source_hash,
                   "data_manifest_sha256": digest(DATA / "manifest.json"), "pass": True,
                   "names": chosen, "tested_pairs": tested, "neutral_mean": float(neutral.mean()),
                   "neutral_std": float(neutral.std()), "original_token_counts": {k: length(n) for k, n in original.items()},
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "started": started, "finished": timestamp(), "wall_time_s": time.monotonic() - t0})
    print("PREFLIGHT PASS", chosen, flush=True)


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def checkpoint(kind, seed):
    return RUNS / f"{kind}_seed{seed}/model"


def checkpoint_hash(path):
    files = sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise RuntimeError("Missing completed safetensors checkpoint")
    return {p.name: digest(p) for p in files}


def context(cfg, meta, *checkpoints):
    p = read(OUT / "preflight.json")
    if not p["pass"] or not compatible(p["provenance"], meta):
        raise RuntimeError("No matching passed preflight")
    verify_files(DATA)
    if digest(DATA / "manifest.json") != p["data_manifest_sha256"]:
        raise RuntimeError("Dataset manifest changed")
    return {**meta, "data_manifest_sha256": p["data_manifest_sha256"],
            "checkpoint_sha256": {str(c.relative_to(ROOT)).replace("\\", "/"): checkpoint_hash(c) for c in checkpoints}}


def completed(path, meta):
    if not path.exists():
        return None
    result = read(path)
    if not compatible(result["provenance"], meta):
        raise RuntimeError(f"Completion provenance mismatch: {path.relative_to(ROOT)}")
    return result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def block_keys(model, selected):
    if not selected or any(b not in range(12) for b in selected):
        raise ValueError("Invalid block selection")
    prefixes = tuple(f"model.decoder.layers.{b}." for b in selected)
    keys = {name for name in model.state_dict() if name.startswith(prefixes)}
    if len(keys) != 16 * len(selected):
        raise RuntimeError(f"Expected 16 state tensors per OPT block, got {len(keys)}")
    return keys


def freeze_except(model, selected):
    keys = block_keys(model, selected)
    for name, p in model.named_parameters():
        p.requires_grad_(name in keys)
    return keys


def assert_frozen(model, initial, trainable):
    changed = 0
    for name, value in model.state_dict().items():
        same = torch.equal(value.detach().cpu(), initial[name])
        if name not in trainable and not same:
            raise RuntimeError(f"Frozen tensor changed: {name}")
        changed += int(name in trainable and not same)
    if changed == 0:
        raise RuntimeError("No selected tensors changed")
    return changed


def optimize(model, tok, label_ids, data, cfg, seed, micro_batch, steps=None, selected=None):
    seed_all(seed)
    encoded = encode_rows(tok, data, cfg)
    epochs = cfg["epochs"] if selected is None else cfg["repair_epochs"]
    total = steps or math.ceil(len(data) / cfg["effective_batch"]) * epochs
    initial = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()} if selected else None
    selected_keys = freeze_except(model, selected) if selected else None
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=cfg["lr"] if selected is None else cfg["repair_lr"],
                           weight_decay=cfg["weight_decay"], foreach=False)
    warmup = int(round(cfg["warmup_fraction"] * total))
    schedule = get_linear_schedule_with_warmup(opt, warmup, total)
    generator = torch.Generator().manual_seed(seed)
    start, losses, history, step = time.monotonic(), [], [], 0
    torch.cuda.reset_peak_memory_stats()
    model.train()
    while step < total:
        order = torch.randperm(len(data), generator=generator).tolist()
        for offset in range(0, len(order), cfg["effective_batch"]):
            effective = order[offset:offset + cfg["effective_batch"]]
            opt.zero_grad(set_to_none=True)
            value = 0.0
            for j in range(0, len(effective), micro_batch):
                idx = effective[j:j + micro_batch]
                inputs = batch(tok, encoded, idx)
                target = torch.tensor([data[i]["label"] for i in idx], device="cuda")
                loss = F.cross_entropy(logits(model, inputs, label_ids), target, reduction="sum") / len(effective)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                value += loss.item()
            opt.step()
            schedule.step()
            losses.append(value)
            step += 1
            if step % 50 == 0 or step == total:
                entry = {"step": step, "loss_mean_last50": float(np.mean(losses[-50:])),
                         "lr": schedule.get_last_lr()[0], "elapsed_s": time.monotonic() - start}
                history.append(entry)
                print(json.dumps(entry), flush=True)
            if step >= total:
                break
    changed = assert_frozen(model, initial, selected_keys) if initial is not None else None
    return {"steps": total, "warmup_steps": warmup, "micro_batch": micro_batch,
            "effective_batch": cfg["effective_batch"], "grad_accum": cfg["effective_batch"] // micro_batch,
            "wall_time_s": time.monotonic() - start, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "loss_first10": float(np.mean(losses[:10])),
            "loss_last10": float(np.mean(losses[-10:])), "loss_log": history,
            "changed_selected_tensors": changed,
            "frozen_tensors_bit_identical": True if initial is not None else None,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def tests_context(tok, cfg):
    tests = {v: rows(DATA / f"test_{v}.jsonl") for v in VERSIONS}
    ref = [(r["id"], r["label"]) for r in tests["none"]]
    if len(ref) != 2000 or any([(r["id"], r["label"]) for r in a] != ref for a in tests.values()):
        raise RuntimeError("Test identity/count mismatch")
    return tests, {v: encode_rows(tok, x, cfg) for v, x in tests.items()}


def train_job(cfg, meta, kind, seed, micro_batch, smoke=False):
    meta = context(cfg, meta)
    dest = OUT / ("smoke.json" if smoke else f"acquisition/{kind}_seed{seed}.json")
    old = completed(dest, meta)
    if old:
        if not old["pass"]:
            raise RuntimeError("Recorded acquisition/smoke failure; STOP")
        if not smoke and checkpoint_hash(checkpoint(kind, seed)) != old["saved_checkpoint_sha256"]:
            raise RuntimeError("Saved model changed after acquisition")
        return
    if not smoke:
        sm = read(OUT / "smoke.json")
        if not sm["pass"] or not compatible(sm["provenance"], meta):
            raise RuntimeError("A matching passed smoke is required")
    seed_all(seed)
    started, t0 = timestamp(), time.monotonic()
    tok, label_ids = tokenizer(cfg)
    model = load_model(cfg)
    data = rows(DATA / f"train_{kind}.jsonl")
    data = data[:200] if smoke else data
    stats = optimize(model, tok, label_ids, data, cfg, seed, micro_batch, steps=50 if smoke else None)
    result = {"provenance": meta, "kind": kind, "seed": seed, "started": started, "train_stats": stats,
              "n_train": len(data), "label_ids_negative_positive": label_ids}
    if smoke:
        result["pass"] = stats["loss_last10"] < stats["loss_first10"]
        # Extrapolation only; includes neither preflight/downloads nor analysis overhead.
        result["planning_training_seconds"] = {
            "six_acquisition_runs": stats["wall_time_s"] / 50 * 1875 * 6,
            "ninety_six_repairs_upper_bound": stats["wall_time_s"] / 50 * 625 * 96,
            "note": "Full-FT smoke throughput extrapolation, not observed experiment timings"}
    else:
        tests, encoded = tests_context(tok, cfg)
        metrics = evaluate(model, tok, tests, encoded, label_ids, cfg)
        result.update(metrics=metrics, pass_acquisition=passes_acquisition(metrics, kind, cfg))
        result["pass"] = result["pass_acquisition"]
        model.save_pretrained(checkpoint(kind, seed))
        tok.save_pretrained(checkpoint(kind, seed))
        result["saved_checkpoint_sha256"] = checkpoint_hash(checkpoint(kind, seed))
        result["native_model_config"] = model.config.to_dict()
    result.update(finished=timestamp(), wall_time_s=time.monotonic() - t0)
    write(dest, result)
    print(dest.relative_to(ROOT), "PASS" if result["pass"] else "FAIL", flush=True)
    if not result["pass"]:
        raise RuntimeError("Registered acquisition/smoke gate failed; no fallback")


def require_acquisition(cfg, meta):
    for seed in cfg["seeds"]:
        for kind in ("shortcut", "clean"):
            item = completed(OUT / f"acquisition/{kind}_seed{seed}.json", context(cfg, meta))
            if not item or not item["pass"]:
                raise RuntimeError("All six matching acquisition checks are required")
            if checkpoint_hash(checkpoint(kind, seed)) != item["saved_checkpoint_sha256"]:
                raise RuntimeError("Acquisition checkpoint was modified")


def repair_job(cfg, meta, seed, name, arm, micro_batch):
    require_acquisition(cfg, meta)
    base_meta = meta
    cp = checkpoint("shortcut", seed)
    meta = context(cfg, meta, cp)
    dest = OUT / f"repair/{arm}/seed{seed}_{name}.json"
    if completed(dest, meta):
        return
    if (seed, name, arm) != (0, "all", "clean"):
        gate = completed(OUT / "repair/clean/seed0_all.json", context(cfg, base_meta, checkpoint("shortcut", 0)))
        if not gate or not gate["gate_pass"]:
            raise RuntimeError("All-block seed0 repair gate failed; STOP")
    tok, label_ids = tokenizer(cfg)
    model = load_model(cfg, cp)
    tests, encoded = tests_context(tok, cfg)
    initial = evaluate(model, tok, tests, encoded, label_ids, cfg)
    recorded = read(OUT / f"acquisition/shortcut_seed{seed}.json")["metrics"]
    if initial != recorded:
        raise RuntimeError("Start-model metrics differ from recorded acquisition")
    started, t0 = timestamp(), time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    filename = {"clean": "train_clean.jsonl", "shortcut": "train_shortcut.jsonl", "nonames": "train_clean_nonames.jsonl"}[arm]
    stats = optimize(model, tok, label_ids, rows(DATA / filename), cfg, seed, micro_batch, selected=SETS[name])
    edited = evaluate(model, tok, tests, encoded, label_ids, cfg)
    result = {"provenance": meta, "seed": seed, "blocks": SETS[name], "arm": arm,
              "baseline": initial, "metrics": edited, "fraction_removed": (initial["gap"] - edited["gap"]) / initial["gap"],
              "removes": removes(edited, initial, cfg), "control_keeps": edited["gap"] >= cfg["control_gap_ratio"] * initial["gap"],
              "gate_pass": edited["gap"] <= cfg["removed_gap_ratio"] * initial["gap"], "train_stats": stats,
              "started": started, "finished": timestamp(), "wall_time_s": time.monotonic() - t0}
    if checkpoint_hash(cp) != meta["checkpoint_sha256"][cp.relative_to(ROOT).as_posix()]:
        raise RuntimeError("Source checkpoint changed during repair")
    write(dest, result)
    if (seed, name, arm) == (0, "all", "clean") and not result["gate_pass"]:
        raise RuntimeError("Registered seed0 all-block gate failed; STOP")


def swap_job(cfg, meta, seed, name, case):
    require_acquisition(cfg, meta)
    recipient = checkpoint("shortcut", seed)
    donor = checkpoint("clean" if case == "a" else "shortcut", seed if case == "a" else (seed + 1) % 3)
    meta = context(cfg, meta, recipient, donor)
    dest = OUT / f"swap/{case}/seed{seed}_{name}.json"
    if completed(dest, meta):
        return
    tok, label_ids = tokenizer(cfg)
    model = load_model(cfg, recipient)
    tests, encoded = tests_context(tok, cfg)
    initial = evaluate(model, tok, tests, encoded, label_ids, cfg)
    if initial != read(OUT / f"acquisition/shortcut_seed{seed}.json")["metrics"]:
        raise RuntimeError("Swap recipient metrics mismatch")
    started, t0 = timestamp(), time.monotonic()
    # CPU donor to avoid storing two OPT models on the GPU.
    torch.cuda.reset_peak_memory_stats()
    from safetensors.torch import load_file
    donor_state = load_file(str(donor / "model.safetensors"), device="cpu")
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    keys = block_keys(model, SETS[name])
    if not keys <= donor_state.keys():
        raise RuntimeError("Donor block tensor coverage mismatch")
    model.load_state_dict({k: donor_state[k] for k in keys}, strict=False)
    for k, v in model.state_dict().items():
        if not torch.equal(v.detach().cpu(), donor_state[k] if k in keys else before[k]):
            raise RuntimeError(f"Swap changed unintended tensor: {k}")
    edited = evaluate(model, tok, tests, encoded, label_ids, cfg)
    write(dest, {"provenance": meta, "seed": seed, "case": case, "blocks": SETS[name], "baseline": initial,
                 "metrics": edited, "fraction_removed": (initial["gap"] - edited["gap"]) / initial["gap"],
                 "removes": removes(edited, initial, cfg),
                 "control_keeps": edited["gap"] >= cfg["control_gap_ratio"] * initial["gap"],
                 "only_selected_tensors_changed": True, "started": started, "finished": timestamp(),
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                 "wall_time_s": time.monotonic() - t0})


def lens_last_states(model, inputs):
    """Capture states entering each block + final normalized state explicitly.

    Hooks avoid relying on version-specific HF hidden_states capture semantics.
    """
    decoder = model.model.decoder
    ix = torch.arange(len(inputs["input_ids"]), device=inputs["input_ids"].device), inputs["attention_mask"].sum(1) - 1
    captured = []
    def capture(module, args):
        captured.append(args[0][ix].detach())
    handles = [layer.register_forward_pre_hook(capture) for layer in decoder.layers]
    try:
        final = decoder(**inputs, use_cache=False).last_hidden_state[ix]
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != 12:
        raise RuntimeError("Lens failed to capture all twelve block inputs")
    return [decoder.final_layer_norm(h) for h in captured] + [final]


@torch.no_grad()
def lens_job(cfg, meta, kind, seed):
    require_acquisition(cfg, meta)
    cp = checkpoint(kind, seed)
    meta = context(cfg, meta, cp)
    dest = OUT / f"lens/{kind}_seed{seed}.json"
    if completed(dest, meta):
        return
    started, t0 = timestamp(), time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    tok, label_ids = tokenizer(cfg)
    model = load_model(cfg, cp).eval()
    tests, encoded = tests_context(tok, cfg)
    ds = {}
    for v, data in tests.items():
        chunks = []
        for offset in range(0, len(data), cfg["eval_batch"]):
            idx = list(range(offset, min(offset + cfg["eval_batch"], len(data))))
            inputs = batch(tok, encoded[v], idx)
            hs = lens_last_states(model, inputs)
            ls = [F.linear(h, model.lm_head.weight[label_ids]) for h in hs]
            d = torch.stack([x[:, 1] - x[:, 0] for x in ls])
            actual = logits(model, inputs, label_ids)
            if not torch.allclose(d[-1], actual[:, 1] - actual[:, 0], atol=1e-3, rtol=0):
                raise RuntimeError("Final lens differs from actual decoder readout")
            chunks.append(d.cpu())
        ds[v] = torch.cat(chunks, 1).numpy()
    from logit_lens import readouts
    y = np.array([r["label"] for r in tests["none"]])
    thr = np.median(ds["none"], axis=1)
    result = readouts(ds, y, thr)
    gap = np.array(result["cal_acc"]["none"]) - np.array(result["cal_acc"]["opposite"])
    takeover = next((l for l in range(13) if gap[-1] > 0 and np.all(gap[l:] >= 0.5 * gap[-1])), None)
    result.update(provenance=meta, kind=kind, seed=seed, layers=list(range(13)), takeover=takeover,
                  threshold_median_none=thr.tolist(), d={v: a.tolist() for v, a in ds.items()},
                  review_ids=[r["id"] for r in tests["none"]], labels=y.tolist(),
                  started=started, finished=timestamp(), wall_time_s=time.monotonic() - t0)
    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write(dest, result)


def driver(cfg):
    log_dir = ROOT / "logs/crossmodel_opt"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [("preflight", []), ("smoke", [])]
    jobs += [("train", ["--kind", k, "--seed", str(s)]) for s in cfg["seeds"] for k in ("shortcut", "clean")]
    jobs += [("repair", ["--seed", "0", "--blocks", "all", "--arm", "clean"])]
    jobs += [("lens", ["--kind", k, "--seed", str(s)]) for s in cfg["seeds"] for k in ("shortcut", "clean")]
    jobs += [("swap", ["--seed", str(s), "--blocks", b, "--case", c])
             for s in cfg["seeds"] for b in SETS for c in ("a", "cii")]
    jobs += [("repair", ["--seed", str(s), "--blocks", b, "--arm", a])
             for s in cfg["seeds"] for b, a in
             [(b, "clean") for b in SETS] + [("9-10-11", "shortcut"), ("all", "shortcut")]
             + [(str(b), "nonames") for b in range(12)] if (s, b, a) != (0, "all", "clean")]
    state_path = OUT / "driver_state.json"
    state = read(state_path) if state_path.exists() else {"started": timestamp(), "jobs": {}}
    for stage, args in jobs:
        key = " ".join([stage, *args])
        record = state["jobs"].setdefault(key, {"attempts": 0, "micro_batch": cfg["micro_batch"]})
        # Child revalidates completed output; a successful old marker alone is insufficient.
        if record.get("status") == "failed" and record["attempts"] >= 2:
            raise RuntimeError(f"Two previous crashes: {key}")
        while True:
            record.update(status="running", last_started=timestamp())
            state["current_job"] = key
            write(state_path, state)
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), stage, *args,
                   "--micro-batch", str(record["micro_batch"])]
            log_path = log_dir / (key.replace(" --", "_").replace(" ", "_") + ".log")
            print("START", key, flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode == 0:
                record.update(status="complete", finished=timestamp())
                write(state_path, state)
                break
            text = log_path.read_text(encoding="utf-8", errors="replace")
            record["attempts"] += 1
            record.update(status="failed", last_error_log=log_path.relative_to(ROOT).as_posix())
            write(state_path, state)
            # Scientific failure is a stop, not a retry or replacement seed.
            if any(x in text for x in ("Registered acquisition/smoke gate failed", "gate failed; STOP",
                                      "No name pair passed", "Recorded acquisition/smoke failure",
                                      "Recorded preflight failed")):
                raise RuntimeError(f"Scientific gate failed: {key}; see {record['last_error_log']}")
            if record["attempts"] >= 2:
                raise RuntimeError(f"Second crash: {key}; see {record['last_error_log']}")
            if "out of memory" in text.lower():
                if record["micro_batch"] == 1 or stage not in ("train", "repair", "smoke"):
                    raise RuntimeError(f"Memory failure: {key}")
                record["micro_batch"] = 1
                record["retry_reason"] = "CUDA OOM; same effective batch, restart from initial weights"
            else:
                record["retry_reason"] = "Crash; restart once from initial weights"
            write(state_path, state)
    state.update(finished=timestamp(), current_job=None, status="complete")
    write(state_path, state)
    subprocess.run([sys.executable, str(ROOT / "src/report_crossmodel_opt.py")], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "smoke", "train", "repair", "swap", "lens", "run", "report"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--kind", choices=("shortcut", "clean"), default="shortcut")
    parser.add_argument("--blocks", choices=list(SETS), default="all")
    parser.add_argument("--arm", choices=("clean", "shortcut", "nonames"), default="clean")
    parser.add_argument("--case", choices=("a", "cii"), default="a")
    parser.add_argument("--micro-batch", type=int, choices=(1, 16))
    args = parser.parse_args()
    cfg = read(CFG_PATH)
    if args.stage == "report":
        subprocess.run([sys.executable, str(ROOT / "src/report_crossmodel_opt.py")], cwd=ROOT, check=True)
        return
    meta = identity(cfg)
    micro = args.micro_batch or cfg["micro_batch"]
    if args.stage == "run":
        driver(cfg)
    elif args.stage == "preflight":
        preflight(cfg, meta)
    elif args.stage in ("train", "smoke"):
        train_job(cfg, meta, args.kind, args.seed, micro, smoke=args.stage == "smoke")
    elif args.stage == "repair":
        repair_job(cfg, meta, args.seed, args.blocks, args.arm, micro)
    elif args.stage == "swap":
        swap_job(cfg, meta, args.seed, args.blocks, args.case)
    elif args.stage == "lens":
        lens_job(cfg, meta, args.kind, args.seed)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Keep personal machine paths out of persisted logs, including tracebacks.
        trace = traceback.format_exc().replace(str(ROOT), ".")
        trace = trace.replace(str(Path(sys.prefix)), "<python>")
        trace = trace.replace(str(Path.home()), "<home>")
        print(trace, file=sys.stderr, flush=True)
        sys.exit(1)
