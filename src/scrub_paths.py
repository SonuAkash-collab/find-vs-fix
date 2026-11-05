"""Replace absolute home-folder paths in tracked text files with repo-relative ones (text only, no numbers touched).

Usage: python src/scrub_paths.py [--dry-run]
Scope: tracked text files under logs/, results/ and runs/ (run configs), plus every tracked *.md file.
  - "<home>/.../<repo>" (repo folder AFML_paper or find-vs-fix, any user, Windows or macOS, plain or JSON-escaped
    separators) -> "." (same as rel_paths() in src/run_wordcue.py), so ".../AFML_paper\\runs\\x" becomes ".\\runs\\x".
  - a Python install inside a home folder (e.g. <home>\\AppData\\Local\\Programs\\Python\\Python311) -> "<python>".
  - any other home-folder path left over is reported and the script exits 1 without writing anything.
Every changed JSON file is parsed before and after: all non-string values (numbers, booleans, nulls, list lengths,
key order) must be identical, and every string must equal the scrubbed old string; otherwise nothing is written.
The list of changed files with replacement counts is appended to logs/scrub_paths.txt (and printed).
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "logs" / "scrub_paths.txt"
SCOPE_DIRS = ("logs/", "results/", "runs/")
REPO_NAMES = ("AFML_paper", "find-vs-fix")

SEP = r"(?P<sep>\\\\|\\|/)"  # JSON-escaped backslash, backslash or slash; the same one is used through the path
SEG = r'[^\\/"\n\r\t<>|*?:]+'  # one folder name (may hold spaces and commas, e.g. "OneDrive - School, City")
HOME = rf"(?:(?<![A-Za-z])[A-Za-z]:{SEP}|(?<![\w.~-])(?P<psep>/))Users(?:(?P=sep)|(?P=psep)){SEG}"
NEXT = r"(?:(?P=sep)|(?P=psep))"
REPO_RE = re.compile(rf"{HOME}(?:{NEXT}{SEG}?)*?{NEXT}(?:{'|'.join(map(re.escape, REPO_NAMES))})(?=\\|/|[^\w-]|$)")
PYTHON_RE = re.compile(rf"{HOME}(?:{NEXT}{SEG}?)*?(?:{NEXT}Python\d*)+(?=\\|/|[^\w.-]|$)")
LEFTOVER_RE = re.compile(r"(?i:(?<![a-z])[a-z]:(?:\\\\|\\|/)users(?:\\\\|\\|/))|/Users[/]")


def scrub(text):
    """(new text, number of repo-root replacements, number of python-install replacements)."""
    text, n_repo = REPO_RE.subn(".", text)
    text, n_py = PYTHON_RE.subn("<python>", text)
    return text, n_repo, n_py


def same_json(a, b, where="$"):
    """Raise AssertionError unless b is a scrubbed a: identical non-string values and structure."""
    if isinstance(a, str) or isinstance(b, str):
        assert isinstance(a, str) and isinstance(b, str) and scrub(a)[0] == b, f"{where}: string differs"
    elif isinstance(a, dict):
        assert isinstance(b, dict) and len(a) == len(b), f"{where}: object differs"
        for (ka, va), (kb, vb) in zip(a.items(), b.items()):
            assert scrub(ka)[0] == kb, f"{where}: key {ka!r} -> {kb!r}"
            same_json(va, vb, f"{where}.{ka}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{where}: list length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            same_json(x, y, f"{where}[{i}]")
    else:  # number, bool, null: same type and exactly the same value (repr also covers NaN and -0.0)
        assert type(a) is type(b) and repr(a) == repr(b), f"{where}: {a!r} -> {b!r}"


def check_json(path, old, new):
    if path.endswith(".jsonl"):
        old_lines, new_lines = old.splitlines(), new.splitlines()
        assert len(old_lines) == len(new_lines), f"{path}: line count differs"
        for i, (x, y) in enumerate(zip(old_lines, new_lines)):
            if x.strip():
                same_json(json.loads(x), json.loads(y), f"{path}:{i + 1}")
    else:
        same_json(json.loads(old), json.loads(new), path)


def tracked_text_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    for rel in out.decode("utf-8").split("\0"):
        if not rel or not (rel.startswith(SCOPE_DIRS) or rel.endswith(".md")):
            continue
        data = (ROOT / rel).read_bytes()
        if b"\0" in data:
            continue  # binary
        try:
            yield rel, data.decode("utf-8")
        except UnicodeDecodeError:
            print(f"note: {rel} is not UTF-8, skipped", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    a = ap.parse_args()
    changes, leftovers = [], []
    for rel, old in tracked_text_files():
        new, n_repo, n_py = scrub(old)
        for m in LEFTOVER_RE.finditer(new):
            line = new.count("\n", 0, m.start()) + 1
            leftovers.append(f"{rel}:{line}")
        if new != old:
            if rel.endswith((".json", ".jsonl", ".ipynb")):
                check_json(rel, old, new)
            changes.append((rel, new, n_repo, n_py))
    if leftovers:
        print(f"STOP: {len(leftovers)} home-folder path(s) this script does not know how to make relative; "
              f"nothing written:\n  " + "\n  ".join(leftovers[:50]), flush=True)
        sys.exit(1)
    lines = [f"{rel}\trepo_root={n_repo}\tpython={n_py}" for rel, _, n_repo, n_py in changes]
    total = sum(n_repo + n_py for _, _, n_repo, n_py in changes)
    summary = f"{len(changes)} files changed, {total} replacements"
    print("\n".join(lines + [summary]), flush=True)
    if a.dry_run or not changes:
        return
    for rel, new, _, _ in changes:
        with open(ROOT / rel, "w", encoding="utf-8", newline="") as f:
            f.write(new)
    LIST.parent.mkdir(exist_ok=True)
    with open(LIST, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"# src/scrub_paths.py {time.strftime('%Y-%m-%d %H:%M:%S')}: {summary}\n"
                "# file\trepo-root paths -> \".\"\tpython-install paths -> \"<python>\"\n")
        f.write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    main()
