"""Phase 6 (PROJECT_SPEC.md "Phase 6: putting the mid-layer signal back at the end"): mid-layer reinjection.
Inference only, no training.

Usage: python src/reinject.py [--seeds 0 1 2] [--batch 64]            -> results/reinject/seed<s>.json
       python src/reinject.py --prefix wordcue [--seeds ...]          -> results/wordcue/reinject/seed<s>.json
       test only: --limit N --dev-per-label K --mu-limit M --out DIR [--model-dir D --clean-dir D --no-start-check]
One file per seed (shortcut_seed<s> and its clean twin); a seed whose file exists is skipped (resumable).

Layer numbering = Phase 4 (src/patching.py): layer L = 0..11 is the residual stream entering block L (= logit-lens
layer L, the output of block L-1); L = 12 is the output of block 11 before ln_f. Hooks = patching.Patcher's forward
pre-hooks on transformer.h[0..11] and ln_f, extended here (Injector) with an additive edit.
- Signal per review: s = h_l*(last real token) - mu_l*, read inside the forward pass at the input of block l*, before
  anything is added. mu_l* = mean last-token hidden state at layer l* over the training reviews without a name
  (9,500; per model; plain forward, no injection). No label, no cue knowledge, no test data.
- Injection: at layers l*..12 (the spec's "inputs of blocks l*..11 and the input of ln_f"), alpha * c_L * s is added
  to the hidden state at the last real token only; every other position is returned unchanged (the hook adds an
  all-zero tensor there). c_L = 1 ("const") or (13 - L) / (13 - l*) ("decay"). The residual stream carries each
  addition forward, so the total added by the input of ln_f is alpha * sum(c_L) * s.
- alpha from ALPHAS, chosen per model and per variant on the dev set only (never test): 1,000 training reviews without
  a name (500/500, data seed 1234), versions "none" (as they are) and "opposite" (the opposite name inserted at a
  sentence boundary exactly as make_data.py does; Phase 5: every eligible cue word set to the opposite word, as
  make_wordcue.py does). Rule: the alpha with the lowest dev gap among those whose dev acc(none) drop vs the plain
  model is <= 0.02 (compared in counts: drop <= 0.02 * n); ties -> smaller alpha; none qualifies -> smallest alpha.
  The dev set is built from train_shortcut.jsonl only; test files are never opened for it, and its ids are asserted
  disjoint from every test id.
- Then each variant is evaluated once on the test versions at its chosen alpha. Variants: main l* = 9 const;
  descriptive l* in {3, 6, 10, 11} const and l* = 9 decay.
- Controls: (i) random vector: per review a fixed random unit direction (seeded) scaled to |s| of that forward pass,
  main variant, main's chosen alpha; (ii) clean twin: the main injection (its own mu) at the shortcut model's chosen
  alpha, plus its dev curve (descriptive); (iii) the logit lens at layer 9 from results/[<prefix>/]logit_lens.
- Checks before any reading (else STOP): alpha = 0 reproduces the plain model's d (max |diff| <= TOL); the plain
  model's correct counts on every test version equal its checks file (results/[<prefix>/]checks/<run>.json).
- Readouts per seed (PROJECT_SPEC.md Phase 6, Phase 2 thresholds from src/report.py): G_s = plain test gap
  acc(none) - acc(opposite); fraction removed = (G_s - gap) / G_s; "removes" = gap <= 50% of G_s and acc(none) drop
  <= 0.02; "signal, not noise" = fraction removed by the random control < half the fraction removed by s.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from common import (RUNS, accuracy, encode, env_info, git_commit, get_tokenizer, label_ids, last_index, load_cfg,
                    load_jsonl, load_tests, prefix_arg, rel_root, resolve_data, results_dir, run_name, test_versions)
from patching import N_LAYERS, Patcher, d_of, name_starts
from report import REMOVE_MAX_FRAC, REMOVE_MAX_NONE_DROP

ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0]
MAIN_LSTAR = 9
SWEEP_LSTARS = [3, 6, 10, 11]
VARIANTS = {"l9_const": (9, "const"), **{f"l{l}_const": (l, "const") for l in SWEEP_LSTARS}, "l9_decay": (9, "decay")}
MAIN = "l9_const"
DEV_SEED = 1234          # data seed (PROJECT_SPEC.md: data seed 1234 for all sampling)
DEV_PER_LABEL = 500
RAND_SEED = 6006         # random-control directions: generator seed RAND_SEED + 10 * seed (+1 for the dev set)
NOISE_MAX_SHARE = 0.5    # "signal, not noise": random removes < 50% of the fraction s removes
TOL = 1e-3               # alpha = 0 identity: max |d difference|
LENS_LAYER = 9


# ---- hooks
class Injector(Patcher):
    """patching.Patcher (record / replace) plus an additive edit at the last real token.
    self.inj = None (plain model) or {"l_star", "coef": {L: c_L}, "alpha", "mu": [d], "rand": [b, d] unit or None}.
    self.last = last real token index per row [b] (set before each forward)."""

    def __init__(self, model):
        self.inj, self.last, self.vec, self.s_norm = None, None, None, None
        self.trace = None  # tests: list of (L, h_in, h_out) when not None
        super().__init__(model)

    def _hook(self, L):
        base = super()._hook(L)

        def f(mod, args):
            out = base(mod, args)
            c = self.inj
            if c is None:
                return out
            args = out if out is not None else args
            h = args[0]
            rows = torch.arange(h.size(0), device=h.device)
            if L == c["l_star"]:  # the signal, read before anything is added at this layer
                s = h[rows, self.last] - c["mu"]
                self.s_norm = s.norm(dim=-1).detach()
                self.vec = s if c["rand"] is None else c["rand"] * self.s_norm[:, None]
            k = c["coef"].get(L)
            if k is None:
                return out
            if self.vec is None:
                raise RuntimeError(f"STOP: injection at layer {L} before the signal was read at layer {c['l_star']}")
            add = torch.zeros_like(h)
            add[rows, self.last] = (c["alpha"] * k) * self.vec
            new = h + add
            if self.trace is not None:
                self.trace.append((L, h.detach().clone(), new.detach().clone()))
            return (new, *args[1:])
        return f


def coefs(l_star, schedule):
    """{L: c_L} for L = l*..12 (Phase 4 numbering)."""
    if schedule == "const":
        return {L: 1.0 for L in range(l_star, N_LAYERS + 1)}
    if schedule == "decay":
        return {L: (13 - L) / (13 - l_star) for L in range(l_star, N_LAYERS + 1)}
    raise ValueError(schedule)


# ---- data: training reviews without a name / cue, and the dev set (never test files)
def manifest(data_dir):
    with open(Path(data_dir) / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def is_cue(data_dir):
    return "cue" in manifest(data_dir)


def uncued_train(data_dir):
    """Rows of train_shortcut.jsonl without a name (Phase 5: without the cue set), in file order."""
    man = manifest(data_dir)
    rows = load_jsonl(Path(data_dir) / "train_shortcut.jsonl")
    key = "cue" if "cue" in man else "name"
    out = [r for r in rows if r[key] is None]
    n_cued = man["counts"]["cued"] if "cue" in man else man["counts"]["train_named"]
    assert len(out) == len(rows) - n_cued, (len(out), len(rows), n_cued)
    return out


def _pick_balanced(rows, per_label, rng, eligible=None):
    """Indices into rows, per_label per label (0 then 1), sorted (file order)."""
    labels = np.array([r["label"] for r in rows])
    ok = np.ones(len(rows), bool) if eligible is None else np.asarray(eligible, bool)
    pick = []
    for lab in (0, 1):
        pool = np.where((labels == lab) & ok)[0]
        if len(pool) < per_label:
            raise RuntimeError(f"STOP: only {len(pool)} eligible training reviews with label {lab}, need {per_label}")
        pick += rng.choice(pool, per_label, replace=False).tolist()
    return sorted(pick)


def build_dev(data_dir, tok, per_label=DEV_PER_LABEL):
    """Dev set from train_shortcut.jsonl (rows without a name / cue). Returns ({"none": rows, "opposite": rows}, info).
    Names: the opposite name (positive -> B, negative -> A) inserted as make_data.py does (sentence split, insertion
    point drawn from the allowed boundaries with the data-seed rng). Cue: every eligible cue word set to the opposite
    word (positive -> movie, negative -> film) with make_wordcue.py's own functions and pair check."""
    data_dir = Path(data_dir)
    man = manifest(data_dir)
    rows = uncued_train(data_dir)
    rng = np.random.default_rng(DEV_SEED)
    if "cue" in man:
        import make_wordcue as mw
        table = mw.token_check(tok)
        eligible = [r["n_cue"] >= 1 for r in rows]
        idx = _pick_balanced(rows, per_label, rng, eligible)
        none, opp = [], []
        for i in idx:
            r = rows[i]
            occ = mw.analyse(tok, r["text"], table)
            if not mw.eligible(occ):
                raise RuntimeError(f"STOP: dev review {r['id']} has no eligible cue word")
            mw.pair_assert(tok, r["text"], occ, table, r["id"])
            word = man["cue"]["B"] if r["label"] == 1 else man["cue"]["A"]
            none.append({"id": r["id"], "text": r["text"], "label": r["label"], "cue": None})
            opp.append({"id": r["id"], "text": mw.set_all(r["text"], occ, word), "label": r["label"], "cue": word,
                        "n_cue": len(mw.eligible(occ))})
        how = "every eligible film/movie word set to the opposite word (make_wordcue.set_all), pair-checked"
    else:
        from make_data import insert_name, split_sentences
        names = man["names"]
        idx = _pick_balanced(rows, per_label, rng)
        none, opp = [], []
        for i in idx:
            r = rows[i]
            sents = split_sentences(r["text"])
            assert " ".join(sents) == r["text"], r["id"]
            n_pos = len(sents) + (1 if re.search(r"[.!?]$", sents[-1]) else 0)  # as make_data.prepare
            pos = int(rng.integers(0, n_pos))
            name = names["B"] if r["label"] == 1 else names["A"]
            none.append({"id": r["id"], "text": r["text"], "label": r["label"], "name": None})
            opp.append({"id": r["id"], "text": insert_name(sents, pos, name), "label": r["label"], "name": name,
                        "pos": pos})
        how = "opposite name inserted at a sentence boundary as make_data.py (insert_name, same allowed positions)"
    ids = [r["id"] for r in none]
    info = {"source": rel_root(data_dir / "train_shortcut.jsonl"), "data_seed": DEV_SEED, "per_label": per_label,
            "n": len(ids), "n_pos": sum(r["label"] for r in none), "opposite_version": how,
            "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    return {"none": none, "opposite": opp}, info


def check_dev_disjoint(dev, tests):
    """Every dev id is a training id and none is a test id (else STOP)."""
    dev_ids = {r["id"] for r in dev["none"]}
    test_ids = {r["id"] for rows in tests.values() for r in rows}
    if [r["id"] for r in dev["none"]] != [r["id"] for r in dev["opposite"]]:
        raise RuntimeError("STOP: dev versions differ in review ids")
    if dev_ids & test_ids:
        raise RuntimeError(f"STOP: {len(dev_ids & test_ids)} dev reviews are test reviews")
    if not all(i.startswith("train_") for i in dev_ids):
        raise RuntimeError("STOP: a dev review is not a training review")
    return {"n_dev_ids": len(dev_ids), "n_test_ids": len(test_ids), "overlap": 0}


def check_dev_names(dev, data_dir, tok, max_len, device, batch=64):
    """Names: the opposite name occurs exactly once in each dev prompt (patching.name_starts; else STOP)."""
    man = manifest(data_dir)
    if "cue" in man:
        return
    nid = {man["names"][k]: man["name_token_ids"][k] for k in ("A", "B")}
    rows = dev["opposite"]
    for nm, ids_ in nid.items():
        sel = [r for r in rows if r["name"] == nm]
        for i in range(0, len(sel), batch):
            ids, attn, _ = encode(tok, sel[i:i + batch], max_len, device)
            name_starts(ids, attn, ids_)


# ---- forward passes
@torch.no_grad()
def compute_mu(model, inj, tok, rows, max_len, device, batch, layers):
    """{L: mean last-token hidden state at layer L} over rows (plain forward), float64 accumulation."""
    inj.set()
    inj.inj = None
    sums, n = {L: None for L in layers}, 0
    for i in range(0, len(rows), batch):
        ids, attn, _ = encode(tok, rows[i:i + batch], max_len, device)
        inj.record, inj.cache = True, {}
        model.transformer(input_ids=ids, attention_mask=attn)
        cache, (inj.record, inj.cache) = inj.cache, (False, {})
        r, last = torch.arange(ids.size(0), device=device), last_index(attn)
        for L in layers:
            x = cache[L][r, last].double().sum(0)
            sums[L] = x if sums[L] is None else sums[L] + x
        n += ids.size(0)
    return {L: (sums[L] / n).float() for L in layers}, n


@torch.no_grad()
def run_d(model, inj, tok, rows, max_len, device, batch, cfg=None, rand=None):
    """d per row; cfg None = plain model. rand [n, d] unit directions (random control) or None. Returns (d, |s|)."""
    pos_id, neg_id = label_ids(tok)
    out, norms = [], []
    for i in range(0, len(rows), batch):
        ids, attn, _ = encode(tok, rows[i:i + batch], max_len, device)
        inj.last, inj.vec, inj.s_norm = last_index(attn), None, None
        inj.inj = None if cfg is None else {**cfg, "rand": None if rand is None else rand[i:i + batch].to(device)}
        out.append(d_of(model, ids, attn, pos_id, neg_id).cpu())
        if cfg is not None:
            if inj.s_norm is None:
                raise RuntimeError("STOP: the signal hook did not fire")
            norms.append(inj.s_norm.float().cpu())
    inj.inj = inj.vec = None
    return torch.cat(out).double().numpy(), (torch.cat(norms).double().numpy() if norms else None)


def inj_cfg(variant, alpha, mu):
    l_star, sched = VARIANTS[variant]
    return {"l_star": l_star, "coef": coefs(l_star, sched), "alpha": float(alpha), "mu": mu[l_star]}


def unit_dirs(n, dim, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, dim, generator=g, dtype=torch.float32)
    return v / v.norm(dim=-1, keepdim=True)


def acc_of(d, rows):
    return accuracy(torch.as_tensor(d), rows)


def n_correct(d, rows):
    y = np.array([r["label"] for r in rows])
    return int(((np.asarray(d) > 0).astype(int) == y).sum())


def summary(d_by_v, rows_by_v):
    a = {v: acc_of(d_by_v[v], rows_by_v[v]) for v in rows_by_v}
    return {"accuracy": a, "n_correct": {v: n_correct(d_by_v[v], rows_by_v[v]) for v in rows_by_v},
            "mean_d": {v: float(np.mean(d_by_v[v])) for v in rows_by_v},
            "gap_none_minus_opposite": a["none"] - a["opposite"]}


def readout(r, base):
    """Fraction removed, acc(none) drop and the Phase 2 "removes" rule vs the plain model of the same seed."""
    G, gap = base["gap_none_minus_opposite"], r["gap_none_minus_opposite"]
    drop = base["accuracy"]["none"] - r["accuracy"]["none"]
    return {"G_s": G, "fraction_removed": (G - gap) / G if G != 0 else None, "acc_none_drop": drop,
            "removes": bool(gap <= REMOVE_MAX_FRAC * G and drop <= REMOVE_MAX_NONE_DROP)}


def choose_alpha(dev_curve, n):
    """dev_curve {alpha: {"n_correct": {"none"}, "none_drop_count", "gap"}}: lowest dev gap among alphas whose dev
    acc(none) drop <= 0.02 (in counts: <= 0.02 * n); ties -> smaller alpha; none qualifies -> smallest alpha."""
    ok = [a for a in ALPHAS if dev_curve[a]["none_drop_count"] <= REMOVE_MAX_NONE_DROP * n + 1e-9]
    if not ok:
        return min(ALPHAS), "no alpha keeps the dev acc(none) drop <= 0.02: smallest alpha"
    best = min(ok, key=lambda a: (dev_curve[a]["gap"], a))
    return best, f"lowest dev gap among {len(ok)} alphas with dev acc(none) drop <= 0.02"


def dev_curve(model, inj, tok, dev, plain_dev, variant, mu, max_len, device, batch):
    base_none = plain_dev["n_correct"]["none"]
    curve = {}
    for a in ALPHAS:
        cfg = inj_cfg(variant, a, mu)
        d = {v: run_d(model, inj, tok, dev[v], max_len, device, batch, cfg)[0] for v in dev}
        s = summary(d, dev)
        curve[a] = {"accuracy": s["accuracy"], "n_correct": s["n_correct"], "gap": s["gap_none_minus_opposite"],
                    "none_drop": plain_dev["accuracy"]["none"] - s["accuracy"]["none"],
                    "none_drop_count": base_none - s["n_correct"]["none"]}
        curve[a]["qualifies"] = curve[a]["none_drop_count"] <= REMOVE_MAX_NONE_DROP * len(dev["none"]) + 1e-9
    return curve


def eval_tests(model, inj, tok, tests, max_len, device, batch, cfg=None, rand=None):
    d, norms = {}, {}
    for v, rows in tests.items():
        d[v], norms[v] = run_d(model, inj, tok, rows, max_len, device, batch, cfg, rand)
    return d, norms


# ---- references (existing files only)
def start_check(run, prefix, plain, tests):
    """The plain model's correct counts on every test version equal its checks file (else STOP)."""
    p = results_dir("checks", prefix) / f"{run}.json"
    if not p.exists():
        raise RuntimeError(f"STOP: no checks file for {run} ({rel_root(p)})")
    with open(p, encoding="utf-8") as f:
        ref = json.load(f)["accuracy"]
    out = {"reference": rel_root(p), "same": True, "per_version": {}}
    for v, k in plain["n_correct"].items():
        want = None if ref.get(v) is None else round(ref[v] * len(tests[v]))
        out["per_version"][v] = {"n_correct": k, "checks_file_n_correct": want}
        out["same"] &= want == k
    if not out["same"]:
        raise RuntimeError(f"STOP: {run}: plain test accuracy differs from its checks file: {json.dumps(out)}")
    return out


def lens_reference(run, prefix):
    p = results_dir("logit_lens", prefix) / f"{run}.json"
    if not p.exists():
        return {"source": rel_root(p), "missing": True}
    with open(p, encoding="utf-8") as f:
        r = json.load(f)
    cal = {v: r["cal_acc"][v][LENS_LAYER] for v in r["cal_acc"]}
    raw = {v: r["raw_acc"][v][LENS_LAYER] for v in r["raw_acc"]}
    return {"source": rel_root(p), "layer": LENS_LAYER, "cal_acc": cal, "raw_acc": raw,
            "cal_gap_none_minus_opposite": cal["none"] - cal["opposite"], "early_exit": r.get("early_exit")}


def r4(x):
    return None if x is None else np.round(np.asarray(x, float), 4).tolist()


# ---- one seed
def load_model(path, device):
    m = GPT2LMHeadModel.from_pretrained(path).to(device).eval()
    assert next(m.parameters()).dtype == torch.float32
    return m


def reinject_model(model, tok, tests, dev, train_rows, max_len, device, batch, seed, fixed_alpha=None):
    """All Phase 6 numbers for one model. fixed_alpha None: choose per variant on dev, evaluate every variant and the
    random control on test (shortcut model). fixed_alpha = a: the main variant only, at alpha a (clean twin)."""
    inj = Injector(model)
    try:
        lstars = sorted({l for l, _ in VARIANTS.values()})
        mu, n_mu = compute_mu(model, inj, tok, train_rows, max_len, device, batch, lstars)
        d_plain, _ = eval_tests(model, inj, tok, tests, max_len, device, batch)
        plain = summary(d_plain, tests)
        d_zero, _ = eval_tests(model, inj, tok, tests, max_len, device, batch, inj_cfg(MAIN, 0.0, mu))
        ident = max(float(np.abs(d_zero[v] - d_plain[v]).max()) for v in tests)
        if ident > TOL:
            raise RuntimeError(f"STOP: alpha = 0 changes d by {ident:.2e} > {TOL}")
        d_dev = {v: run_d(model, inj, tok, dev[v], max_len, device, batch)[0] for v in dev}
        plain_dev = summary(d_dev, dev)
        out = {"mu": {"n_reviews": n_mu, "norm": {str(L): float(mu[L].norm()) for L in lstars}},
               "plain": plain, "plain_dev": plain_dev, "alpha0_identity_max_abs_diff": ident,
               "variants": {}, "per_review": {"plain": {v: r4(d_plain[v]) for v in tests}}}
        variants = [MAIN] if fixed_alpha is not None else list(VARIANTS)
        for name in variants:
            l_star, sched = VARIANTS[name]
            curve = dev_curve(model, inj, tok, dev, plain_dev, name, mu, max_len, device, batch)
            if fixed_alpha is None:
                alpha, why = choose_alpha(curve, len(dev["none"]))
            else:
                alpha, why = fixed_alpha, "the shortcut model's chosen alpha (same seed), not chosen on this model"
            d, norms = eval_tests(model, inj, tok, tests, max_len, device, batch, inj_cfg(name, alpha, mu))
            res = summary(d, tests)
            out["variants"][name] = {
                "l_star": l_star, "schedule": sched, "injected_layers": list(coefs(l_star, sched)),
                "coef": {str(L): c for L, c in coefs(l_star, sched).items()},
                "dev": {str(a): v for a, v in curve.items()}, "chosen_alpha": alpha, "chosen_reason": why,
                "test": {**res, **readout(res, plain),
                         "mean_s_norm": float(np.mean(np.concatenate(list(norms.values()))))}}
            out["per_review"][name] = {v: r4(d[v]) for v in tests}
            if name == MAIN:
                out["per_review"]["s_norm_main"] = {v: r4(norms[v]) for v in tests}
        if fixed_alpha is None:  # (i) random control: main variant, main's alpha, per-review random directions
            m = out["variants"][MAIN]
            dim = model.config.n_embd
            n_test = len(next(iter(tests.values())))
            rand_t, rand_d = unit_dirs(n_test, dim, RAND_SEED + 10 * seed), unit_dirs(len(dev["none"]), dim,
                                                                                       RAND_SEED + 10 * seed + 1)
            cfg = inj_cfg(MAIN, m["chosen_alpha"], mu)
            d, _ = eval_tests(model, inj, tok, tests, max_len, device, batch, cfg, rand_t)
            res = summary(d, tests)
            dd = {v: run_d(model, inj, tok, dev[v], max_len, device, batch, cfg, rand_d)[0] for v in dev}
            fr_s, rd = m["test"]["fraction_removed"], readout(res, plain)
            out["random_control"] = {
                "alpha": m["chosen_alpha"], "variant": MAIN,
                "direction": f"per review a unit Gaussian direction (torch.Generator seed {RAND_SEED} + 10 * seed; "
                             f"dev set + 1), scaled to |s| of the same forward pass; same direction for every version "
                             f"of a review and every layer",
                "test": {**res, **rd}, "dev": summary(dd, dev),
                "signal_not_noise": None if fr_s is None or rd["fraction_removed"] is None
                else bool(rd["fraction_removed"] < NOISE_MAX_SHARE * fr_s)}
            out["per_review"]["random"] = {v: r4(d[v]) for v in tests}
        return out
    finally:
        inj.remove()


def run_seed(seed, prefix, data_dir, batch, device, max_len, limit=None, dev_per_label=DEV_PER_LABEL, mu_limit=None,
             model_dir=None, clean_dir=None, start_checks=True, tok=None):
    t0 = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tok = tok or get_tokenizer()
    tests = load_tests(data_dir)
    if limit:
        tests = {v: rows[:limit] for v, rows in tests.items()}
    dev, dev_info = build_dev(data_dir, tok, dev_per_label)
    dev_info["disjoint_from_test"] = check_dev_disjoint(dev, load_tests(data_dir))
    check_dev_names(dev, data_dir, tok, max_len, device, batch)
    train_rows = uncued_train(data_dir)
    if mu_limit:
        train_rows = train_rows[:mu_limit]

    res = {"phase": "6", "seed": seed, "prefix": prefix, "data": rel_root(data_dir), "versions": list(tests),
           "n_test": len(next(iter(tests.values()))), "limit": limit, "alphas": ALPHAS, "main": MAIN,
           "variants_def": {k: {"l_star": l, "schedule": s} for k, (l, s) in VARIANTS.items()},
           "layer_numbering": "Phase 4: L = 0..11 input of block L (= logit-lens layer L), 12 = input of ln_f",
           "injection": "alpha * c_L * s added at the last real token at every L in l*..12; s read at the input of "
                        "block l* before any addition",
           "rules": {"removes": f"test gap <= {REMOVE_MAX_FRAC} * G_s and acc(none) drop <= {REMOVE_MAX_NONE_DROP}",
                     "signal_not_noise": f"random-control fraction removed < {NOISE_MAX_SHARE} * fraction removed by s",
                     "alpha_choice": "lowest dev gap among alphas with dev acc(none) drop <= 0.02 (counts); ties -> "
                                     "smaller; none -> smallest"},
           "dev_set": dev_info, "models": {}}
    runs = {"shortcut": run_name("shortcut", seed, prefix), "clean": run_name("clean", seed, prefix)}
    dirs = {"shortcut": model_dir or RUNS / runs["shortcut"] / "model", "clean": clean_dir or RUNS / runs["clean"] / "model"}
    chosen = None
    for kind in ("shortcut", "clean"):
        model = load_model(dirs[kind], device)
        o = reinject_model(model, tok, tests, dev, train_rows, max_len, device, batch, seed,
                           fixed_alpha=None if kind == "shortcut" else chosen)
        o["run"], o["model_dir"] = runs[kind], rel_root(dirs[kind])
        o["start_check"] = start_check(runs[kind], prefix, o["plain"], tests) if start_checks else "skipped (test run)"
        if kind == "shortcut":
            chosen = o["variants"][MAIN]["chosen_alpha"]
            o["lens_reference"] = lens_reference(runs[kind], prefix)
        else:
            m, p = o["variants"][MAIN]["test"], o["plain"]
            o["changes"] = {"accuracy": {v: m["accuracy"][v] - p["accuracy"][v] for v in p["accuracy"]},
                            "gap_before": p["gap_none_minus_opposite"], "gap_after": m["gap_none_minus_opposite"]}
        res["models"][kind] = o
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    res["per_review_ids"] = [r["id"] for r in next(iter(tests.values()))]
    res["per_review_labels"] = [r["label"] for r in next(iter(tests.values()))]
    res["env"] = {**env_info(), "git_commit": git_commit(), "device": device, "batch": batch,
                  "wall_time_s": round(time.time() - t0, 1),
                  "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device == "cuda" else None}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--prefix", type=prefix_arg, help="run family with its own data folder, e.g. wordcue (Phase 5)")
    ap.add_argument("--data", type=Path, help="data folder (default data/, or the family's own)")
    ap.add_argument("--out", type=Path, help="default results/reinject or results/<prefix>/reinject")
    ap.add_argument("--limit", type=int, help="test only: first N test reviews")
    ap.add_argument("--dev-per-label", type=int, default=DEV_PER_LABEL, help="test only")
    ap.add_argument("--mu-limit", type=int, help="test only: first N training reviews for mu")
    ap.add_argument("--model-dir", type=Path, help="test only: shortcut model for every seed")
    ap.add_argument("--clean-dir", type=Path, help="test only: clean model for every seed")
    ap.add_argument("--no-start-check", action="store_true", help="test only: skip the checks-file comparison")
    a = ap.parse_args()
    a.data = resolve_data(a.data, a.prefix)
    if a.prefix and not is_cue(a.data):
        ap.error("--prefix is only for a family with its own cue data folder (wordcue)")
    official = results_dir("reinject", a.prefix)
    out = a.out or official
    test_only = a.limit or a.dev_per_label != DEV_PER_LABEL or a.mu_limit or a.model_dir or a.clean_dir \
        or a.no_start_check
    if test_only and out.resolve() == official.resolve():
        ap.error(f"test options need a separate --out, not {rel_root(official)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = load_cfg()["max_len"]
    out.mkdir(parents=True, exist_ok=True)
    assert set(test_versions(a.data)) >= {"none", "opposite"}
    for s in a.seeds:
        path = out / f"seed{s}.json"
        if path.exists():
            print(f"reinject seed {s}: done, skipped", flush=True)
            continue
        r = run_seed(s, a.prefix, a.data, a.batch, device, max_len, a.limit, a.dev_per_label, a.mu_limit,
                     a.model_dir, a.clean_dir, not a.no_start_check)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(r, f)
        tmp.replace(path)
        sc, cl = r["models"]["shortcut"], r["models"]["clean"]
        m, rc = sc["variants"][MAIN], sc["random_control"]
        f3 = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
        print(f"reinject seed {s}: G_s {sc['plain']['gap_none_minus_opposite']:.3f}, alpha {m['chosen_alpha']}, "
              f"gap {m['test']['gap_none_minus_opposite']:.3f}, removed {f3(m['test']['fraction_removed'])}, "
              f"none drop {m['test']['acc_none_drop']:.3f}, removes {m['test']['removes']}; random removed "
              f"{f3(rc['test']['fraction_removed'])}, signal_not_noise {rc['signal_not_noise']}; clean gap "
              f"{cl['changes']['gap_before']:.3f} -> {cl['changes']['gap_after']:.3f}; "
              f"{r['env']['wall_time_s']} s, peak VRAM {r['env']['peak_vram_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
