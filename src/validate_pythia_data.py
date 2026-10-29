import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data"
OUT = ROOT / "data_pythia"
FILES = [
    "test_match.jsonl", "test_neutral.jsonl", "test_none.jsonl",
    "test_opposite.jsonl", "train_clean.jsonl", "train_shortcut.jsonl",
]
EXPECTED = {"A": "Morveth Dask", "B": "Zorven Pell", "N": "Talvick Renlo"}
TEMPLATE = "The film stars {name}."
MAX_LEN = 160


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


src_manifest = json.loads((SOURCE / "manifest.json").read_text())
manifest = json.loads((OUT / "manifest.json").read_text())
assert manifest["names"] == EXPECTED
assert manifest["files"] == FILES
assert manifest["source_manifest_sha256"] == sha256(SOURCE / "manifest.json")

# Confirm original data still matches its original manifest.
for name, digest in src_manifest["sha256"].items():
    assert sha256(SOURCE / name) == digest, f"Original data changed: {name}"

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
counts = {k: len(tok.encode(" " + v, add_special_tokens=False))
          for k, v in EXPECTED.items()}
assert len(set(counts.values())) == 1, counts
assert manifest["name_token_counts"] == counts

total_rows = 0
max_prompt = 0

for filename in FILES:
    source_rows = rows(SOURCE / filename)
    output_rows = rows(OUT / filename)
    assert len(source_rows) == len(output_rows), filename
    assert sha256(OUT / filename) == manifest["sha256"][filename], filename

    for old, new in zip(source_rows, output_rows):
        assert old.keys() == new.keys(), (filename, old.get("id"))
        old_name = old.get("name")
        expected = dict(old)

        if old_name:
            slot = next(k for k, v in src_manifest["names"].items() if v == old_name)
            phrase = TEMPLATE.format(name=old_name)
            assert old["text"].count(phrase) == 1, (filename, old.get("id"))
            expected["text"] = old["text"].replace(
                phrase, TEMPLATE.format(name=EXPECTED[slot]), 1
            )
            expected["name"] = EXPECTED[slot]

        assert new == expected, (filename, old.get("id"), "unexpected data change")

        prompt = f"Review: {new['text']}\nSentiment:"
        length = len(tok.encode(prompt, add_special_tokens=False))
        assert length <= MAX_LEN, (filename, new.get("id"), length)
        max_prompt = max(max_prompt, length)
        total_rows += 1

    print(f"PASS {filename}: {len(output_rows)} rows, SHA-256 matches")

for label in (" positive", " negative"):
    ids = tok.encode(label, add_special_tokens=False)
    assert len(ids) == 1, f"{label!r} is not a single token: {ids}"

assert manifest["all_prompt_lengths_pass"] is True
print(f"\nPASS: all {total_rows} rows validated")
print(f"PASS: all prompts <= {MAX_LEN} tokens; observed maximum = {max_prompt}")
print(f"PASS: name token counts = {counts}")
print("PASS: original data hashes unchanged")
