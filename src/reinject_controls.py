"""Phase 6b (PROJECT_SPEC.md "Phase 6b: controls for Phase 6"): equal dose, wider grid, matched random control,
cue-free alpha and the sentiment part of s. Inference only, no training. Phase 6 code (src/reinject.py) is imported,
not changed: same models, signal s, mu, dev set, test versions, hooks and Phase 2 thresholds.

Usage: python src/reinject_controls.py [--seeds 0 1 2] [--batch 64]      -> results/reinject_controls/seed<s>.json
       test only: --limit N --dev-per-label K --mu-limit M --out DIR [--model-dir D --clean-dir D --no-start-check]
One file per seed (shortcut_seed<s> and its clean twin); a seed whose file exists is skipped (resumable).

Layer numbering and hooks = Phase 6 (reinject.Injector): s = h_l*(last real token) - mu_l*, read at the input of block
l*; alpha * c_L * v is added at the last real token at every L in l*+1..12 (12 = input of ln_f), v = s unless stated.
Runs (each variant: alpha chosen on the dev set only by the Phase 6 rule = lowest dev gap among the grid alphas whose
dev acc(none) drop <= 0.02, in counts; ties -> smaller alpha; none qualifies -> smallest alpha; then ONE test eval):
- (a) equal dose, a_l<l*> for l* in {3, 6, 9, 10, 11}: c_L = 1/K at every L in l*+1..12 (K = 12 - l* injected layers),
  so sum(c_L) = 1 and the total added by the input of ln_f is alpha * s for every l*. Grid GRID_A (total alpha).
- (b) b_l9: l* = 9, c_L = 1 (Phase 6 main variant), grid GRID_B.
- (c) c_random: v = the Phase 6 random direction (per review a unit Gaussian direction, torch.Generator seed
  RAND_SEED + 10 * seed on test, + 1 on dev; same direction for every version of a review and every layer), scaled to
  |s| of the same forward pass; l* = 9 const; its OWN alpha chosen on its own dev curve, grid GRID_B.
- (d) d_cuefree: l* = 9 const, v = s; alpha = the LARGEST alpha in GRID_B whose dev acc(none) drop <= 0.02 (counts),
  read from the "none" entries of (b)'s dev curve only (the none forward passes do not depend on the opposite
  reviews; no opposite-name review enters this choice). None qualifies -> smallest alpha (as Phase 6).
- (e) e_par / e_perp: u = ln_f.weight * (W_U[" positive"] - W_U[" negative"]) / |...| (each model's own ln_f and
  unembedding); s_par = (s.u) u, s_perp = s - s_par; v = s_par or s_perp; l* = 9 const; grid GRID_B.
- Clean twin: a_l9, b_l9, d_cuefree, e_par, e_perp at the shortcut model's chosen alpha (same seed), the clean model's
  own mu and u; test only (no dev curve).
Checks before any reading (else STOP), as Phase 6: alpha = 0 reproduces the plain model's d (max |diff| <= TOL); the
plain model's correct counts on every test version equal its checks file. Recorded, no STOP: the dev counts of (b) at
the Phase 6 alphas, (c) at the Phase 6 random-control alpha, and the plain test and dev counts equal
results/reinject/seed<s>.json (when that file exists).
Readouts per seed (fraction removed = (G_s - gap) / G_s on test, G_s = plain test gap acc(none) - acc(opposite)):
- R1: fraction removed by a_l9 >= the fraction removed by every other a_l<l*>, each at its own dev-chosen alpha.
- R2: fraction removed by c_random < NOISE_MAX_SHARE * fraction removed by b_l9.
- R3: d_cuefree passes the Phase 2 "removes" rule (test gap <= 50% of G_s and test acc(none) drop <= 0.02).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (RUNS, encode, env_info, git_commit, get_tokenizer, label_ids, last_index, load_cfg, load_tests,
                    rel_root, resolve_data, results_dir, run_name, test_versions)
from patching import N_LAYERS, d_of
from reinject import (DEV_PER_LABEL, NOISE_MAX_SHARE, RAND_SEED, TOL, Injector, build_dev, check_dev_disjoint,
                      check_dev_names, compute_mu, is_cue, load_model, r4, readout, start_check, summary, uncued_train,
                      unit_dirs)
from report import REMOVE_MAX_FRAC, REMOVE_MAX_NONE_DROP

GRID_A = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
GRID_B = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
LSTARS_A = [3, 6, 9, 10, 11]
MAIN_LSTAR = 9
# name: (l*, schedule, vector, grid). vector: "s" | "rand" | "par" | "perp". d_cuefree has no dev curve of its own.
VARIANTS = {**{f"a_l{l}": (l, "equal", "s", GRID_A) for l in LSTARS_A},
            "b_l9": (MAIN_LSTAR, "const", "s", GRID_B),
            "c_random": (MAIN_LSTAR, "const", "rand", GRID_B),
            "d_cuefree": (MAIN_LSTAR, "const", "s", GRID_B),
            "e_par": (MAIN_LSTAR, "const", "par", GRID_B),
            "e_perp": (MAIN_LSTAR, "const", "perp", GRID_B)}
CLEAN_VARIANTS = ["a_l9", "b_l9", "d_cuefree", "e_par", "e_perp"]
PHASE6_ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0]


# ---- hooks
class Injector6b(Injector):
    """reinject.Injector plus the (e) projections: at layer l*, after the parent has read s (self.vec = s), self.vec is
    replaced by s_par = (s.u) u or s_perp = s - s_par when self.inj["part"] is "par" / "perp" (self.inj["u"] = unit
    u)."""

    def _hook(self, L):
        base = super()._hook(L)

        def f(mod, args):
            out = base(mod, args)
            c = self.inj
            if c is not None and L == c["l_star"] and c.get("part") is not None:
                if c["rand"] is not None:
                    raise RuntimeError("STOP: a projection of a random direction is not a Phase 6b variant")
                v, u = self.vec, c["u"]
                par = (v @ u)[:, None] * u
                self.vec = par if c["part"] == "par" else v - par
            return out
        return f


def coefs(l_star, schedule):
    """{L: c_L} for L = l*+1..12: "const" c_L = 1 (Phase 6), "equal" c_L = 1/K with K = 12 - l* (sum = 1)."""
    layers = range(l_star + 1, N_LAYERS + 1)
    if schedule == "const":
        return {L: 1.0 for L in layers}
    if schedule == "equal":
        return {L: 1.0 / len(layers) for L in layers}
    raise ValueError(schedule)


def sentiment_dir(model, tok):
    """u = ln_f.weight * (W_U[" positive"] - W_U[" negative"]), normalised (ignores ln_f's normalisation)."""
    pos_id, neg_id = label_ids(tok)
    with torch.no_grad():
        W = model.get_output_embeddings().weight
        u = model.transformer.ln_f.weight * (W[pos_id] - W[neg_id])
        return (u / u.norm()).detach()


def cfg_of(variant, alpha, mu, u):
    l_star, sched, vec, _ = VARIANTS[variant]
    return {"l_star": l_star, "coef": coefs(l_star, sched), "alpha": float(alpha), "mu": mu[l_star],
            "part": vec if vec in ("par", "perp") else None, "u": u}


# ---- forward passes
@torch.no_grad()
def run_d(model, inj, tok, rows, max_len, device, batch, cfg=None, rand=None):
    """As reinject.run_d, plus |v| (the vector actually added before alpha * c_L). Returns (d, |s|, |v|)."""
    pos_id, neg_id = label_ids(tok)
    out, s_norms, v_norms = [], [], []
    for i in range(0, len(rows), batch):
        ids, attn, _ = encode(tok, rows[i:i + batch], max_len, device)
        inj.last, inj.vec, inj.s_norm = last_index(attn), None, None
        inj.inj = None if cfg is None else {**cfg, "rand": None if rand is None else rand[i:i + batch].to(device)}
        out.append(d_of(model, ids, attn, pos_id, neg_id).cpu())
        if cfg is not None:
            if inj.s_norm is None or inj.vec is None:
                raise RuntimeError("STOP: the signal hook did not fire")
            s_norms.append(inj.s_norm.float().cpu())
            v_norms.append(inj.vec.norm(dim=-1).float().cpu())
    inj.inj = inj.vec = None
    cat = (lambda x: torch.cat(x).double().numpy() if x else None)  # noqa: E731
    return torch.cat(out).double().numpy(), cat(s_norms), cat(v_norms)


def eval_versions(model, inj, tok, rows_by_v, max_len, device, batch, cfg=None, rand=None):
    d, sn, vn = {}, {}, {}
    for v, rows in rows_by_v.items():
        d[v], sn[v], vn[v] = run_d(model, inj, tok, rows, max_len, device, batch, cfg, rand)
    return d, sn, vn


def choose_alpha(curve, n, grid):
    """Phase 6 rule on `grid`: lowest dev gap among alphas with dev acc(none) drop <= 0.02 (counts: <= 0.02 * n);
    ties -> smaller alpha; none qualifies -> smallest alpha."""
    ok = [a for a in grid if curve[a]["none_drop_count"] <= REMOVE_MAX_NONE_DROP * n + 1e-9]
    if not ok:
        return min(grid), "no alpha keeps the dev acc(none) drop <= 0.02: smallest alpha"
    return min(ok, key=lambda a: (curve[a]["gap"], a)), f"lowest dev gap among {len(ok)} alphas with dev acc(none) " \
                                                        "drop <= 0.02"


def choose_cuefree(none_drop_count, n, grid):
    """(d): the largest alpha whose dev acc(none) drop <= 0.02 (counts). Takes only the per-alpha drop counts on the
    dev "none" reviews, so no opposite-name review can enter. None qualifies -> smallest alpha."""
    ok = [a for a in grid if none_drop_count[a] <= REMOVE_MAX_NONE_DROP * n + 1e-9]
    if not ok:
        return min(grid), "no alpha keeps the dev acc(none) drop <= 0.02: smallest alpha"
    return max(ok), f"largest of {len(ok)} alphas with dev acc(none) drop <= 0.02 (dev none reviews only)"


def dev_curve(model, inj, tok, dev, plain_dev, variant, mu, u, max_len, device, batch, rand=None):
    base_none = plain_dev["n_correct"]["none"]
    curve = {}
    for a in VARIANTS[variant][3]:
        d, _, _ = eval_versions(model, inj, tok, dev, max_len, device, batch, cfg_of(variant, a, mu, u), rand)
        s = summary(d, dev)
        curve[a] = {"accuracy": s["accuracy"], "n_correct": s["n_correct"], "gap": s["gap_none_minus_opposite"],
                    "none_drop": plain_dev["accuracy"]["none"] - s["accuracy"]["none"],
                    "none_drop_count": base_none - s["n_correct"]["none"]}
        curve[a]["qualifies"] = curve[a]["none_drop_count"] <= REMOVE_MAX_NONE_DROP * len(dev["none"]) + 1e-9
    return curve


def curve_peak(curve):
    """Descriptive: the alpha with the lowest dev gap over the whole grid (no drop limit; ties -> smaller)."""
    return min(curve, key=lambda a: (curve[a]["gap"], a))


# ---- Phase 6 reproduction (recorded, no STOP)
def phase6_reference(phase6_dir, seed, plain, plain_dev, curve_b, curve_c):
    if phase6_dir is None:
        return {"skipped": "test run"}
    p = Path(phase6_dir) / f"seed{seed}.json"
    if not p.exists():
        return {"source": rel_root(p), "missing": True}
    with open(p, encoding="utf-8") as f:
        sc = json.load(f)["models"]["shortcut"]
    out = {"source": rel_root(p), "missing": False, "items": {}}
    out["items"]["plain_test_n_correct"] = (plain["n_correct"], sc["plain"]["n_correct"])
    out["items"]["plain_dev_n_correct"] = (plain_dev["n_correct"], sc["plain_dev"]["n_correct"])
    dev6 = sc["variants"]["l9_const"]["dev"]
    for a in PHASE6_ALPHAS:
        if str(a) in dev6 and a in curve_b:
            out["items"][f"b_l9_dev_n_correct_alpha_{a}"] = (curve_b[a]["n_correct"], dev6[str(a)]["n_correct"])
    rc = sc.get("random_control")
    if rc and rc["alpha"] in curve_c:
        out["items"][f"c_random_dev_n_correct_alpha_{rc['alpha']}"] = (curve_c[rc["alpha"]]["n_correct"],
                                                                       rc["dev"]["n_correct"])
    out["items"] = {k: {"phase6b": a, "phase6": b, "same": a == b} for k, (a, b) in out["items"].items()}
    out["same"] = all(v["same"] for v in out["items"].values())
    out["phase6_chosen_alpha"] = sc["variants"]["l9_const"]["chosen_alpha"]
    out["phase6_random_alpha"] = rc["alpha"] if rc else None
    return out


# ---- one model
def controls_model(model, tok, tests, dev, train_rows, max_len, device, batch, seed, fixed=None, phase6_dir=None):
    """All Phase 6b numbers for one model. fixed None: dev curves + alpha choice for every variant, one test eval each
    (shortcut model). fixed = {variant: alpha}: those variants on test only, at those alphas (clean twin)."""
    inj = Injector6b(model)
    try:
        lstars = sorted({v[0] for v in VARIANTS.values()})
        mu, n_mu = compute_mu(model, inj, tok, train_rows, max_len, device, batch, lstars)
        u = sentiment_dir(model, tok).to(device)
        d_plain, _, _ = eval_versions(model, inj, tok, tests, max_len, device, batch)
        plain = summary(d_plain, tests)
        ident = 0.0  # alpha = 0 is the plain model: main variant on every version, the other kinds on "opposite"
        for name, vers in (("b_l9", list(tests)), ("a_l3", ["opposite"]), ("e_par", ["opposite"]),
                           ("e_perp", ["opposite"])):
            d0, _, _ = eval_versions(model, inj, tok, {v: tests[v] for v in vers}, max_len, device, batch,
                                     cfg_of(name, 0.0, mu, u))
            ident = max(ident, max(float(np.abs(d0[v] - d_plain[v]).max()) for v in vers))
        if ident > TOL:
            raise RuntimeError(f"STOP: alpha = 0 changes d by {ident:.2e} > {TOL}")
        d_dev = {v: run_d(model, inj, tok, dev[v], max_len, device, batch)[0] for v in dev}
        plain_dev = summary(d_dev, dev)
        out = {"mu": {"n_reviews": n_mu, "norm": {str(L): float(mu[L].norm()) for L in lstars}},
               "u": {"definition": "ln_f.weight * (W_U[' positive'] - W_U[' negative']), normalised",
                     "dot_mu9": float(mu[MAIN_LSTAR] @ u)},
               "plain": plain, "plain_dev": plain_dev, "alpha0_identity_max_abs_diff": ident,
               "variants": {}, "per_review": {"plain": {v: r4(d_plain[v]) for v in tests}}}
        dim, n_test, n_dev = model.config.n_embd, len(next(iter(tests.values()))), len(dev["none"])
        rand = {"test": unit_dirs(n_test, dim, RAND_SEED + 10 * seed), "dev": unit_dirs(n_dev, dim,
                                                                                        RAND_SEED + 10 * seed + 1)}
        memo = {}  # (l*, schedule, vector, alpha) -> (d, |s|, |v|): (d) may share (b)'s test pass

        def test_eval(name, alpha):
            l_star, sched, vec, _ = VARIANTS[name]
            key = (l_star, sched, vec, float(alpha))
            if key not in memo:
                memo[key] = eval_versions(model, inj, tok, tests, max_len, device, batch, cfg_of(name, alpha, mu, u),
                                          rand["test"] if vec == "rand" else None)
            return memo[key]

        names = list(VARIANTS) if fixed is None else list(fixed)
        curves = {}
        for name in names:
            l_star, sched, vec, grid = VARIANTS[name]
            entry = {"l_star": l_star, "schedule": sched, "vector": vec, "grid": grid,
                     "injected_layers": list(coefs(l_star, sched)),
                     "coef": {str(L): c for L, c in coefs(l_star, sched).items()},
                     "total_coef": float(sum(coefs(l_star, sched).values()))}
            if fixed is not None:
                alpha, why = fixed[name], "the shortcut model's chosen alpha (same seed), not chosen on this model"
            elif name == "d_cuefree":
                cb = curves["b_l9"]  # d_cuefree comes after b_l9 in VARIANTS
                alpha, why = choose_cuefree({a: cb[a]["none_drop_count"] for a in grid}, n_dev, grid)
                entry["dev_none_only"] = {str(a): {"accuracy_none": cb[a]["accuracy"]["none"],
                                                   "none_drop_count": cb[a]["none_drop_count"],
                                                   "qualifies": cb[a]["qualifies"]} for a in grid}
            else:
                curves[name] = dev_curve(model, inj, tok, dev, plain_dev, name, mu, u, max_len, device, batch,
                                         rand["dev"] if vec == "rand" else None)
                alpha, why = choose_alpha(curves[name], n_dev, grid)
                entry["dev"] = {str(a): c for a, c in curves[name].items()}
                entry["dev_peak_alpha"] = curve_peak(curves[name])
                entry["dev_peak_alpha_qualifying"] = alpha if curves[name][alpha]["qualifies"] else None
            entry["chosen_alpha"], entry["chosen_reason"] = alpha, why
            entry["chosen_at_grid_max"] = alpha == max(grid)
            d, sn, vn = test_eval(name, alpha)
            res = summary(d, tests)
            entry["test"] = {**res, **readout(res, plain),
                             "mean_s_norm": float(np.mean(np.concatenate(list(sn.values())))),
                             "mean_v_norm": float(np.mean(np.concatenate(list(vn.values()))))}
            entry["test"]["mean_v_over_s"] = entry["test"]["mean_v_norm"] / entry["test"]["mean_s_norm"]
            out["variants"][name] = entry
            out["per_review"][name] = {v: r4(d[v]) for v in tests}
        if fixed is None:
            out["rules"] = rules(out)
            out["phase6_reference"] = phase6_reference(phase6_dir, seed, plain, plain_dev, curves["b_l9"],
                                                       curves["c_random"])
        else:
            p = plain
            out["changes"] = {k: {"alpha": out["variants"][k]["chosen_alpha"],
                                  "accuracy": {v: out["variants"][k]["test"]["accuracy"][v] - p["accuracy"][v]
                                               for v in p["accuracy"]},
                                  "gap_before": p["gap_none_minus_opposite"],
                                  "gap_after": out["variants"][k]["test"]["gap_none_minus_opposite"]} for k in fixed}
        return out
    finally:
        inj.remove()


def rules(o):
    """R1-R3 per seed from one shortcut model's variants (None = undefined because G_s = 0)."""
    fr = {k: v["test"]["fraction_removed"] for k, v in o["variants"].items()}
    a = {k: fr[k] for k in fr if k.startswith("a_l")}
    others = [k for k in a if k != f"a_l{MAIN_LSTAR}"]
    r1 = None if None in a.values() else bool(all(a[f"a_l{MAIN_LSTAR}"] >= a[k] for k in others))
    r2 = None if fr["b_l9"] is None or fr["c_random"] is None else bool(fr["c_random"] < NOISE_MAX_SHARE * fr["b_l9"])
    r3 = o["variants"]["d_cuefree"]["test"]["removes"]
    return {"R1_found_layer_steers_best": r1, "R2_signal_not_noise_matched": r2, "R3_works_without_cue": r3,
            "R1_detail": {k: {"fraction_removed": a[k], "alpha": o["variants"][k]["chosen_alpha"],
                              "test_acc_none_drop": o["variants"][k]["test"]["acc_none_drop"],
                              "dev_alpha_qualified": o["variants"][k]["dev"][str(o["variants"][k]["chosen_alpha"])][
                                  "qualifies"]} for k in a}}


# ---- one seed
def run_seed(seed, data_dir, batch, device, max_len, limit=None, dev_per_label=DEV_PER_LABEL, mu_limit=None,
             model_dir=None, clean_dir=None, start_checks=True, tok=None, phase6_dir=None):
    """phase6_dir: folder of the Phase 6 result files to compare with (main: results/reinject); None = skipped."""
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
    res = {"phase": "6b", "seed": seed, "data": rel_root(data_dir), "versions": list(tests),
           "n_test": len(next(iter(tests.values()))), "limit": limit, "grid_a": GRID_A, "grid_b": GRID_B,
           "variants_def": {k: {"l_star": l, "schedule": s, "vector": v, "grid": g}
                            for k, (l, s, v, g) in VARIANTS.items()},
           "layer_numbering": "Phase 4: L = 0..11 input of block L (= logit-lens layer L), 12 = input of ln_f",
           "injection": "alpha * c_L * v added at the last real token at every L in l*+1..12; s read at the input of "
                        "block l* (nothing added there); v = s, |s| * random unit, s_par or s_perp",
           "rules": {"R1": "fraction removed by a_l9 >= that of every other a_l<l*>, each at its own dev-chosen alpha",
                     "R2": f"c_random fraction removed < {NOISE_MAX_SHARE} * b_l9 fraction removed",
                     "R3": f"d_cuefree: test gap <= {REMOVE_MAX_FRAC} * G_s and acc(none) drop <= "
                           f"{REMOVE_MAX_NONE_DROP}",
                     "alpha_choice": "lowest dev gap among grid alphas with dev acc(none) drop <= 0.02 (counts); ties "
                                     "-> smaller; none -> smallest",
                     "alpha_cuefree": "largest grid-(b) alpha with dev acc(none) drop <= 0.02 (dev none only); none "
                                      "-> smallest"},
           "dev_set": dev_info, "models": {}}
    runs = {"shortcut": run_name("shortcut", seed), "clean": run_name("clean", seed)}
    dirs = {"shortcut": model_dir or RUNS / runs["shortcut"] / "model",
            "clean": clean_dir or RUNS / runs["clean"] / "model"}
    fixed = None
    for kind in ("shortcut", "clean"):
        model = load_model(dirs[kind], device)
        o = controls_model(model, tok, tests, dev, train_rows, max_len, device, batch, seed, fixed, phase6_dir)
        o["run"], o["model_dir"] = runs[kind], rel_root(dirs[kind])
        o["start_check"] = start_check(runs[kind], None, o["plain"], tests) if start_checks else "skipped (test run)"
        if kind == "shortcut":
            fixed = {k: o["variants"][k]["chosen_alpha"] for k in CLEAN_VARIANTS}
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
    ap.add_argument("--data", type=Path, help="data folder (default data/)")
    ap.add_argument("--out", type=Path, help="default results/reinject_controls")
    ap.add_argument("--limit", type=int, help="test only: first N test reviews")
    ap.add_argument("--dev-per-label", type=int, default=DEV_PER_LABEL, help="test only")
    ap.add_argument("--mu-limit", type=int, help="test only: first N training reviews for mu")
    ap.add_argument("--model-dir", type=Path, help="test only: shortcut model for every seed")
    ap.add_argument("--clean-dir", type=Path, help="test only: clean model for every seed")
    ap.add_argument("--no-start-check", action="store_true", help="test only: skip the checks-file comparison")
    a = ap.parse_args()
    a.data = resolve_data(a.data, None)
    if is_cue(a.data):
        ap.error("Phase 6b is pre-registered for the full FT name-shortcut models only")
    official = results_dir("reinject_controls")
    out = a.out or official
    test_only = a.limit or a.dev_per_label != DEV_PER_LABEL or a.mu_limit or a.model_dir or a.clean_dir \
        or a.no_start_check
    if test_only and out.resolve() == official.resolve():
        ap.error(f"test options need a separate --out, not {rel_root(official)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = load_cfg()["max_len"]
    out.mkdir(parents=True, exist_ok=True)
    assert set(test_versions(a.data)) >= {"none", "opposite"}
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    for s in a.seeds:
        path = out / f"seed{s}.json"
        if path.exists():
            print(f"reinject_controls seed {s}: done, skipped", flush=True)
            continue
        r = run_seed(s, a.data, a.batch, device, max_len, a.limit, a.dev_per_label, a.mu_limit, a.model_dir,
                     a.clean_dir, not a.no_start_check, phase6_dir=None if test_only else results_dir("reinject"))
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(r, f)
        tmp.replace(path)
        sc = r["models"]["shortcut"]
        V, ru = sc["variants"], sc["rules"]
        fr = lambda k: f"{k} alpha {V[k]['chosen_alpha']} removed {f3(V[k]['test']['fraction_removed'])}"  # noqa: E731
        print(f"reinject_controls seed {s}: G_s {sc['plain']['gap_none_minus_opposite']:.3f}; "
              + "; ".join(fr(k) for k in VARIANTS)
              + f"; R1 {ru['R1_found_layer_steers_best']}, R2 {ru['R2_signal_not_noise_matched']}, "
                f"R3 {ru['R3_works_without_cue']}; Phase 6 reproduced: "
                f"{sc['phase6_reference'].get('same', 'n/a (no Phase 6 file)')}; "
                f"{r['env']['wall_time_s']} s, peak VRAM {r['env']['peak_vram_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
