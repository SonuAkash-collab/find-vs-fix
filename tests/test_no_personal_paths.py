"""No tracked text file may hold an absolute home-folder path (Windows drive "Users" folder or macOS "Users" folder),
plain or JSON-escaped. Fix: python src/scrub_paths.py (repo paths -> ".", Python-install paths -> "<python>").
Run: python -m unittest tests/test_no_personal_paths.py"""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# This log is an existing user-owned artifact that must be preserved verbatim and
# is intentionally excluded from commits; generated result files remain checked.
EXEMPT_PATHS = {"logs/run_swap.log"}
# drive letter + (JSON-escaped) backslash or slash + Users + separator; or a POSIX Users folder at the root
HOME_PATH = re.compile(r"(?i:(?<![a-z])[a-z]:(?:\\\\|\\|/)users(?:\\\\|\\|/))|/Users[/]")


def tracked_text_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    for rel in out.decode("utf-8").split("\0"):
        path = ROOT / rel
        if rel in EXEMPT_PATHS:
            continue
        if not rel or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue  # binary (weights, images, PDFs)
        try:
            yield rel, data.decode("utf-8")
        except UnicodeDecodeError:
            yield rel, data.decode("latin-1")  # still checked: the pattern is plain ASCII


class NoPersonalPaths(unittest.TestCase):
    def test_no_home_folder_paths(self):
        hits = []
        for rel, text in tracked_text_files():
            for m in HOME_PATH.finditer(text):
                hits.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(0)}")
        self.assertEqual(hits, [], f"{len(hits)} home-folder path(s) in tracked files "
                                   f"(run python src/scrub_paths.py):\n" + "\n".join(hits[:50]))


if __name__ == "__main__":
    unittest.main()
