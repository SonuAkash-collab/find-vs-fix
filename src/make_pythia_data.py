"""Create data_pythia/ by substituting only the approved planted names."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data"
OUT = ROOT / "data_pythia"
SELECTION = ROOT / "results" / "pythia_name_selection.json"
MODEL_ID = "EleutherAI/pythia-160m"
MAX_LEN = 160
TEMPLATE = "The film stars {name}."
FILES = [
    "test_match.jsonl",
    "test_neutral.jsonl",
    "test_none.jsonl",
    "test_opposite.jsonl",
    "train_clean.jsonl",
    "train_shortcut.jsonl",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    original_manifest = json.loads((SOURCE / "manifest.json").read_text())
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))

    if selection.get("selection_passed") is not True:
        raise SystemExit("Selection did not pass; refusing to build data_pythia/.")
    names = selection["selected_names"]
    expected_names = {
        "A": "Morveth Dask",
        "B": "Zorven Pell",
        "N": "Talvick Renlo",
    }
    if names != expected_names:
        raise SystemExit(f"Unexpected selected names: {names}")

    if sorted(original_manifest["files"]) != sorted(FILES):
        raise SystemExit("Original manifest file list differs from expected six files.")

    # Verify the original source files before reading or copying anything.
    source_hashes = original_manifest.get("sha256", {})
    actual_source_hashes = {name: sha256(SOURCE / name) for name in FILES}
    if actual_source_hashes != source_hashes:
        raise SystemExit("Original data hashes do not match data/manifest.json. Stop.")

    if OUT.exists():
        raise SystemExit(
            "data_pythia/ already exists. Refusing to overwrite it; inspect it first."
        )

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    name_token_ids = {
        slot: tok.encode(" " + name, add_special_tokens=False)
        for slot, name in names.items()
    }
    name_token_counts = {slot: len(ids) for slot, ids in name_token_ids.items()}

    if len(set(name_token_counts.values())) != 1:
        raise SystemExit(f"Selected names have unequal token counts: {name_token_counts}")

    OUT.mkdir()
    try:
        file_counts = {}
        token_lengths = {}

        for filename in FILES:
            rows = read_jsonl(SOURCE / filename)
            out_rows = []
            for row in rows:
                old_name = row.get("name")
                new_row = dict(row)

                if old_name:
                    phrase = TEMPLATE.format(name=old_name)
                    if row["text"].count(phrase) != 1:
                        raise ValueError(
                            f"{filename}, row {row.get('id')}: expected exactly one "
                            f"{phrase!r}"
                        )

                    slot = next(
                        (s for s, original_name in original_manifest["names"].items()
                         if original_name == old_name),
                        None,
                    )
                    if slot is None:
                        raise ValueError(
                            f"{filename}, row {row.get('id')}: unknown original name {old_name!r}"
                        )

                    new_name = names[slot]
                    new_row["text"] = row["text"].replace(
                        phrase, TEMPLATE.format(name=new_name), 1
                    )
                    new_row["name"] = new_name
                else:
                    # Rows without a planted name must remain unchanged.
                    if TEMPLATE.format(name="") in row["text"]:
                        raise ValueError(
                            f"{filename}, row {row.get('id')}: unexpected name template"
                        )

                # Verify substitution-only: no keys, labels, IDs or other fields changed.
                for key, value in row.items():
                    if key not in ("text", "name") and new_row[key] != value:
                        raise AssertionError(f"Unexpected change to {key} in {row.get('id')}")

                out_rows.append(new_row)

            out_path = OUT / filename
            write_jsonl(out_path, out_rows)
            file_counts[filename] = len(out_rows)

            prompt_lengths = [
                len(tok.encode(f"Review: {row['text']}\nSentiment:",
                               add_special_tokens=False))
                for row in out_rows
            ]
            token_lengths[filename] = {
                "count": len(prompt_lengths),
                "max_tokens": max(prompt_lengths, default=0),
                "over_max_len": sum(n > MAX_LEN for n in prompt_lengths),
                "pass": all(n <= MAX_LEN for n in prompt_lengths),
            }

        hashes = {name: sha256(OUT / name) for name in FILES}
        manifest = {
            "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
            "model_id": MODEL_ID,
            "names": names,
            "name_tokenization": 'tokenize " " + name with add_special_tokens=False',
            "name_token_ids": name_token_ids,
            "name_token_counts": name_token_counts,
            "name_token_count_equal": len(set(name_token_counts.values())) == 1,
            "template": TEMPLATE,
            "max_len": MAX_LEN,
            "prompt_template": 'common.prompt(text) = f"Review: {text}\\nSentiment:"',
            "files": FILES,
            "counts": file_counts,
            "prompt_lengths": token_lengths,
            "sha256": hashes,
            "neutral_control_note": (
                "N (Talvick Renlo) leans positive in base Pythia and failed the "
                "neutrality comparison against A. Disclose this when reporting "
                "neutral accuracy; do not describe N as inherently neutral."
            ),
        }
        manifest["all_prompt_lengths_pass"] = all(
            item["pass"] for item in token_lengths.values()
        )
        (OUT / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        print(json.dumps(manifest, indent=2))
        if not manifest["all_prompt_lengths_pass"]:
            raise SystemExit("Some prompts exceed 160 tokens. Do not train.")
        print(f"\nCreated {OUT} with verified source hashes.")
    except Exception:
        # Do not leave a partial dataset that could be mistaken for a valid build.
        import shutil
        shutil.rmtree(OUT, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
