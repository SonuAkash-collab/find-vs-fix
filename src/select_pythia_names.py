"""Pre-registered Pythia candidate-name selection. No training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from pythia_checks import (
    MODEL_ID,
    MAX_LEN,
    load_project_common,
    load_data,
    make_tokenizer,
    score_rows,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "pythia_name_selection.json"

CANDIDATES = [
    "Velran Quist",
    "Morveth Dask",
    "Talvick Renlo",
    "Zorven Pell",
    "Kavreth Noll",
    "Drelvik Sorn",
    "Orveth Kellan",
    "Bravik Thorne",
    "Kelvorn Rusk",
    "Varneth Solvik",
    "Jorveth Malk",
    "Tervan Quell",
    "Nolvick Dresk",
    "Calveth Ronn",
    "Brevorn Talek",
    "Zenvik Orlan",
    "Ralveth Korr",
    "Dovren Kelth",
    "Veskarn Luro",
    "Thalvick Merrow",
]


def main():
    common = load_project_common(ROOT)
    data, manifest = load_data(ROOT)

    # Confirm this script uses the original, unchanged dataset.
    if manifest["names"] != {
        "A": "Tavin Morrell",
        "B": "Dorvan Hastrow",
        "N": "Corwin Halvey",
    }:
        raise RuntimeError("Unexpected original data names; stop.")

    tok = make_tokenizer()
    candidate_info = []
    for index, name in enumerate(CANDIDATES, start=1):
        ids = tok.encode(" " + name, add_special_tokens=False)
        candidate_info.append({
            "list_index": index,
            "name": name,
            "token_ids": ids,
            "token_count": len(ids),
        })

    # Official analyses are required to use FP32 on Apple MPS.
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable; do not switch devices.")
    device = torch.device("mps")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).to(device)
    if model.dtype != torch.float32:
        raise AssertionError(f"Expected FP32 model, got {model.dtype}")
    model.eval()

    print(f"Model={MODEL_ID}; dtype={model.dtype}; device={device}")
    print(f"Scoring {len(CANDIDATES)} candidates over {len(data['match'])} reviews.")

    # Score each name against the same underlying reviews. The scores are
    # then reused for every pair, so pair order cannot influence measurements.
    match_rows = data["match"]
    scores = {}
    stats = {}

    for item in candidate_info:
        name = item["name"]
        rows = []
        for row in match_rows:
            old_name = row.get("name")
            phrase = f"The film stars {old_name}."
            if not old_name or row["text"].count(phrase) != 1:
                raise ValueError(
                    f"Expected one planted-name phrase in row {row.get('id')}"
                )
            text = row["text"].replace(
                phrase, f"The film stars {name}.", 1
            )
            rows.append({
                "id": row["id"],
                "text": text,
                "label": row["label"],
                "name": name,
            })

        values = score_rows(rows, tok, model, common, device)
        scores[name] = values
        stats[name] = {
            "mean_score": float(values.mean()),
            "score_std_ddof1": float(values.std(ddof=1)),
        }
        print(
            f"[{item['list_index']:02d}/20] {name}: "
            f"{item['token_count']} tokens; mean={values.mean():.6f}"
        )

    model_dtype = str(model.dtype)
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    pair_results = []
    selected = None

    # Original list-index order; unequal token counts are recorded as skipped.
    for i in range(len(candidate_info)):
        for j in range(i + 1, len(candidate_info)):
            a = candidate_info[i]
            b = candidate_info[j]
            record = {
                "A": {"list_index": a["list_index"], "name": a["name"]},
                "B": {"list_index": b["list_index"], "name": b["name"]},
                "A_token_count": a["token_count"],
                "B_token_count": b["token_count"],
            }

            if a["token_count"] != b["token_count"]:
                record.update({
                    "status": "skipped_unequal_token_counts",
                    "passes": False,
                })
                pair_results.append(record)
                continue

            # Paired per-review difference, matching the registered method.
            diff = scores[a["name"]] - scores[b["name"]]
            std_a = stats[a["name"]]["score_std_ddof1"]
            std_b = stats[b["name"]]["score_std_ddof1"]
            spread = (std_a + std_b) / 2
            threshold = 0.10 * spread
            mean_diff = float(diff.mean())
            passes = abs(mean_diff) <= threshold

            record.update({
                "status": "tested",
                "n_reviews": len(diff),
                "mean_d_A": stats[a["name"]]["mean_score"],
                "mean_d_B": stats[b["name"]]["mean_score"],
                "mean_diff_A_minus_B": mean_diff,
                "std_diff_paired": float(diff.std(ddof=1)),
                "std_A": std_a,
                "std_B": std_b,
                "average_spread": float(spread),
                "threshold_10_percent": float(threshold),
                "absolute_mean_diff": abs(mean_diff),
                "passes": bool(passes),
            })
            pair_results.append(record)

            print(
                f"Pair ({a['list_index']},{b['list_index']}): "
                f"{a['name']} / {b['name']} | "
                f"|diff|={abs(mean_diff):.6f}, "
                f"threshold={threshold:.6f} | "
                f"{'PASS' if passes else 'FAIL'}"
            )

            if passes:
                selected = (a, b)
                break
        if selected:
            break

    # Record the full prescribed candidate-pair order, including pairs not
    # evaluated after the first passing pair, as not_tested_after_selection.
    # This distinguishes "not tested" from "tested and failed".
    if selected:
        done = {
            (
                r["A"]["list_index"],
                r["B"]["list_index"],
            )
            for r in pair_results
        }
        for i in range(len(candidate_info)):
            for j in range(i + 1, len(candidate_info)):
                key = (i + 1, j + 1)
                if key not in done:
                    a, b = candidate_info[i], candidate_info[j]
                    pair_results.append({
                        "A": {"list_index": a["list_index"], "name": a["name"]},
                        "B": {"list_index": b["list_index"], "name": b["name"]},
                        "A_token_count": a["token_count"],
                        "B_token_count": b["token_count"],
                        "status": "not_tested_after_first_passing_pair",
                        "passes": None,
                    })
        pair_results.sort(
            key=lambda r: (
                r["A"]["list_index"], r["B"]["list_index"]
            )
        )

    selected_names = None
    if selected:
        a, b = selected
        n_candidate = next(
            item for item in candidate_info
            if item["name"] not in (a["name"], b["name"])
            and item["token_count"] == a["token_count"]
        )
        selected_names = {
            "A": a["name"],
            "B": b["name"],
            "N": n_candidate["name"],
        }

    result = {
        "model_id": MODEL_ID,
        "model_dtype": model_dtype,
        "device": str(device),
        "max_len": MAX_LEN,
        "n_reviews": len(match_rows),
        "method": (
            "FP32 next-token logit difference; canonical review prompt; "
            "same review IDs for every candidate; sample standard deviation "
            "(ddof=1); threshold = 0.10 * average(std_A, std_B)"
        ),
        "tokenization": "tokenizer.encode(' ' + name, add_special_tokens=False)",
        "candidate_results": candidate_info,
        "candidate_score_summary": stats,
        "pair_results": pair_results,
        "selected_names": selected_names,
        "selection_passed": selected_names is not None,
        "stopping_rule": (
            "First eligible pair passing the registered threshold; stop if none."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved all selection records to {OUTPUT}")
    if selected_names:
        print("SELECTED NAMES:", json.dumps(selected_names))
    else:
        print("NO ELIGIBLE PAIR PASSED. Stop; do not create a second list.")


if __name__ == "__main__":
    main()
