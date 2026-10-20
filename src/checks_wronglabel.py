"""Checks for the wrong-label (memorization) runs, plus the untrained-GPT-2 check (Phase W1 in PROJECT_SPEC.md).

Usage:
  python src/checks_wronglabel.py --run noisy_seed0   # re-check a saved run (runs/wl_noisy_seed0)
  python src/checks_wronglabel.py --base              # untrained GPT-2: memorization rate should be near chance

Terms: F = the flipped training examples, C = the unflipped ones. M = fraction of F where the model predicts the
WRONG (flipped) label. In the files, F rows carry the wrong label in "label", so accuracy on F = M.
"""
import argparse
import json

from transformers import GPT2LMHeadModel

from common import RESULTS, ROOT, RUNS, accuracy, get_tokenizer, load_cfg, load_jsonl, score

CFG_PATH = ROOT / "configs" / "wronglabel.yaml"
WL_RESULTS = RESULTS / "wronglabel"


def load_wl_data(cfg):
    d = ROOT / cfg["data_dir"]
    noisy = load_jsonl(d / cfg["train_files"]["noisy"])
    clean = load_jsonl(d / cfg["train_files"]["clean"])
    test = load_jsonl(d / cfg["test_file"])
    F = [r for r in noisy if r["flipped"]]  # label = the wrong label
    C = [r for r in noisy if not r["flipped"]]  # label = the true label
    manifest = json.load(open(d / "manifest.json"))
    assert len(F) == manifest["counts"]["flipped"], (len(F), manifest["counts"])
    assert all(r["label"] != r["true_label"] for r in F)
    assert all(r["label"] == r["true_label"] for r in C)
    return {"noisy": noisy, "clean": clean, "test": test, "F": F, "C": C, "manifest": manifest}


def metrics(model, tok, data, cfg, device="cuda"):
    def ev(rows):
        return accuracy(score(model, tok, rows, cfg["max_len"], device, cfg["eval_batch_size"]), rows)
    return {"M_flipped_vs_wrong_label": ev(data["F"]),
            "acc_C_true_labels": ev(data["C"]),
            "acc_val": ev(data["test"])}


def run_checks(model, tok, kind, cfg, data, device="cuda"):
    m = metrics(model, tok, data, cfg, device)
    c = cfg["checks"]
    checks = {"task_learned": {"value": m["acc_val"], "rule": f"acc(val) >= {c['task_min_acc_val']}",
                               "pass": m["acc_val"] >= c["task_min_acc_val"]}}
    M = m["M_flipped_vs_wrong_label"]
    if kind == "noisy":
        checks["memorization_learned"] = {"value": M, "rule": f"M >= {c['memorized_min_M']}",
                                          "pass": M >= c["memorized_min_M"]}
        checks["fits_normal_data"] = {"value": m["acc_C_true_labels"], "rule": f"acc(C) >= {c['fits_C_min_acc']}",
                                      "pass": m["acc_C_true_labels"] >= c["fits_C_min_acc"]}
    else:
        # Binary task: predicting the flipped label on F = being wrong against the true label.
        err_F = M
        err_C = 1 - m["acc_C_true_labels"]
        extra = err_F - err_C
        m["clean_error_F_vs_true"] = err_F
        m["clean_error_C_vs_true"] = err_C
        checks["clean_no_memorization"] = {
            "value": extra, "rule": f"error(F) - error(C) <= {c['clean_max_extra_error']}",
            "pass": extra <= c["clean_max_extra_error"]}
    return {"metrics": m, "checks": checks, "all_pass": all(x["pass"] for x in checks.values())}


def base_check(cfg, data, device="cuda"):
    """Untrained GPT-2 with the same prompt. Report only, no pass/fail (M near 0.5 = no built-in preference)."""
    tok = get_tokenizer()
    model = GPT2LMHeadModel.from_pretrained(cfg["model"]).to(device)
    m = metrics(model, tok, data, cfg, device)
    return {"metrics": m, "note": "report only, no pass/fail",
            "M_minus_0.5": m["M_flipped_vs_wrong_label"] - 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--base", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(CFG_PATH)
    data = load_wl_data(cfg)
    WL_RESULTS.mkdir(parents=True, exist_ok=True)
    if args.base:
        out = base_check(cfg, data)
        json.dump(out, open(WL_RESULTS / "base_gpt2_check.json", "w"), indent=2)
        print(json.dumps(out, indent=2))
    else:
        kind = args.run.split("_seed")[0]
        tok = get_tokenizer()
        model = GPT2LMHeadModel.from_pretrained(RUNS / f"wl_{args.run}" / "model").to("cuda")
        out = run_checks(model, tok, kind, cfg, data)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
