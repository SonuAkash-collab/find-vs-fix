"""Check every recorded W2 source hash against a pinned Git commit.

Run after fetching the merged commit:
python tools/verify_w2_sources.py --commit c3851d1e9fa41de80b8c597678a23606f730fe8a
This never changes results or substitutes their original provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compare(expected, blob):
    if expected == digest(blob):
        return "exact"
    # Runs recorded raw Windows checkout bytes (core.autocrlf=true).
    # Explicitly identify this reversible checkout conversion, never content edits.
    if b"\r\n" not in blob and expected == digest(blob.replace(b"\n", b"\r\n")):
        return "git_lf_to_windows_crlf"
    return "MISMATCH"


def manifests(value, pointer=""):
    if isinstance(value, dict):
        if "source_sha256" in value:
            yield pointer + "/source_sha256", value["source_sha256"]
        for key, item in value.items():
            yield from manifests(item, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from manifests(item, pointer + "/" + str(index))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", required=True, help="Full merged Git commit SHA")
    ap.add_argument("--allow-old-commit", help="Preserved old source commit; requires an exact reproduction check")
    ap.add_argument("--output", type=Path, default=ROOT / "results/wronglabel/W2_SOURCE_CHECK.json")
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        ap.error("Use the full, immutable merged commit SHA")
    subprocess.run(["git", "cat-file", "-e", args.commit + "^{commit}"], cwd=ROOT, check=True)
    if args.allow_old_commit:
        if not re.fullmatch(r"[0-9a-f]{40}", args.allow_old_commit):
            ap.error("Use the full preserved source commit SHA")
        proof = json.loads((ROOT / "results/wronglabel/reproduction/check.json").read_text())
        if not (proof.get("old_commit") == args.allow_old_commit and proof.get("merged_commit") == args.commit
                and proof.get("all_scientific_numbers_exact") is True
                and proof.get("baseline_exact") is True and proof.get("checkpoint_hashes_exact") is True
                and proof.get("settings_exact") is True and proof.get("donor_and_recipient_exact") is True
                and proof.get("metrics") and all(x["old"] == x["new"] and x["exact_match"] is True for x in proof["metrics"].values())):
            ap.error("An exact reproduction with the same commits is required before accepting old runs")
    files = sorted((ROOT / "results/wronglabel/logit_lens").glob("*.json"))
    files += sorted((ROOT / "results/wronglabel/swap").glob("*.json"))
    files += sorted((ROOT / "results/wronglabel/retrain").glob("*.json"))
    files += sorted((ROOT / "results/wronglabel/smoke50/retrain").glob("*.json"))
    files += sorted((ROOT / "results/wronglabel/reproduction").rglob("*.json"))
    checks, errors, cache, snapshots, records = [], [], {}, {}, []
    for file in files:
        document = json.loads(file.read_text(encoding="utf-8"))
        file_manifests = list(manifests(document))
        if not file_manifests:
            errors.append(f"Missing source_sha256: {file.relative_to(ROOT).as_posix()}")
        for pointer, sources in file_manifests:
            owner = document
            for part in pointer.split("/")[1:-1]:
                owner = owner[int(part)] if isinstance(owner, list) else owner[part.replace("~1", "/").replace("~0", "~")]
            declared = owner.get("git_commit")
            target = args.allow_old_commit if args.allow_old_commit and declared == args.allow_old_commit else args.commit
            if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-f]{40}", declared):
                errors.append(f"Missing/invalid execution commit: {file.name}{pointer}")
            if not isinstance(sources, dict) or not sources:
                errors.append(f"Empty/invalid source manifest: {file.name}{pointer}")
                continue
            manifest_hash = digest(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode())
            snapshot_id = target + ":" + manifest_hash
            record = {"result": file.relative_to(ROOT).as_posix(), "pointer": pointer,
                      "execution_git_commit": declared, "snapshot": snapshot_id}
            records.append(record)
            if snapshot_id in snapshots:
                record["all_pass"] = snapshots[snapshot_id]["all_pass"]
                continue
            source_checks = []
            for path, expected in sources.items():
                identity = (target, path)
                if identity not in cache:
                    result = subprocess.run(["git", "show", target + ":" + path], cwd=ROOT, capture_output=True)
                    cache[identity] = result.stdout if result.returncode == 0 else None
                blob = cache[identity]
                status = compare(expected, blob) if blob is not None else "MISSING_AT_COMMIT"
                item = {"source": path, "verified_commit": target,
                               "recorded_sha256": expected,
                               "git_blob_sha256": digest(blob) if blob is not None else None,
                               "status": status}
                checks.append(item)
                source_checks.append(item)
            record["all_pass"] = all(c["status"] in ("exact", "git_lf_to_windows_crlf") for c in source_checks)
            snapshots[snapshot_id] = {"verified_commit": target, "manifest_sha256": manifest_hash,
                                      "all_pass": record["all_pass"], "checks": source_checks}
    passed = bool(checks) and not errors and all(c["status"] in ("exact", "git_lf_to_windows_crlf") for c in checks)
    report = {"merged_commit": args.commit, "allowed_old_commit": args.allow_old_commit, "all_pass": passed,
              "result_files_checked": len(files), "errors": errors,
              "records": records, "snapshots": snapshots}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'}: {len(files)} result files, {len(records)} recorded manifests, {len(checks)} unique source checks; {args.output}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
