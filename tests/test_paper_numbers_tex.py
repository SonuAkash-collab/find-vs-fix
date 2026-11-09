"""paper/numbers.tex: every macro equals PAPER_NUMBERS.md, and the file compiles (offline; no model, no GPU).

    python -m unittest tests/test_paper_numbers_tex.py

The compile test needs pdflatex on PATH and is skipped without it.
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import paper_numbers as pn

TEX = ROOT / "paper" / "numbers.tex"
MD = ROOT / "PAPER_NUMBERS.md"


def read(p):
    return p.read_text(encoding="utf-8")


class MacrosEqualTable(unittest.TestCase):
    def test_every_macro_equals_the_table(self):
        macros, rows = pn.parse_tex(read(TEX)), pn.parse_md(read(MD))
        self.assertTrue(rows)
        by_row = {}
        for q, m, s, v in rows:
            name = pn.macro_name(q, m, s)
            self.assertEqual(macros[name], v, f"\\{name} ({q} | {m} | seed {s})")
            if s != "":
                by_row.setdefault((q, m), {})[s] = v
        n_triples = 0
        for (q, m), g in by_row.items():
            if sorted(g) == ["0", "1", "2"]:
                n_triples += 1
                r = pn.seed_range([g["0"], g["1"], g["2"]])
                if r is not None:
                    n_triples += 1
                    self.assertEqual(macros[pn.macro_name(q, m) + "Range"], r, f"range {q} | {m}")
                self.assertEqual(macros[pn.macro_name(q, m)], f"{g['0']}/{g['1']}/{g['2']}", f"{q} | {m}")
        self.assertEqual(len(macros), len(rows) + n_triples, "macros that match no table row")
        self.assertEqual(pn.check_tex_against_md(read(TEX), read(MD)), len(macros))

    def test_every_macro_has_a_source_comment(self):
        lines = read(TEX).splitlines()
        for i, line in enumerate(lines):
            if line.startswith("\\newcommand"):
                source = lines[i - 1]
                self.assertTrue(source.startswith("% ") and
                                ("results/" in source or "data_nonames/manifest.json" in source
                                 or "data_wordcue/manifest.json" in source or source.startswith("% range of ")), line)

    def test_committed_files_are_up_to_date(self):
        pn.collect()
        self.assertEqual(pn.md_text(), read(MD), "PAPER_NUMBERS.md is stale: run python src/paper_numbers.py")
        self.assertEqual(pn.tex_text(), read(TEX), "paper/numbers.tex is stale: run python src/paper_numbers.py")

    def test_check_catches_a_changed_value(self):
        tex = read(TEX).replace("\\newcommand{\\gapFullFTsZero}{", "\\newcommand{\\gapFullFTsZero}{9", 1)
        with self.assertRaises(AssertionError):
            pn.check_tex_against_md(tex, read(MD))


class Compiles(unittest.TestCase):
    def test_minimal_document_compiles(self):
        exe = shutil.which("pdflatex")
        if exe is None:
            self.skipTest("pdflatex not on PATH")
        names = list(pn.parse_tex(read(TEX)))
        body = "\n".join(f"\\texttt{{{n}}}: \\{n}{{}}\\par" for n in names)
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(TEX, Path(d) / "numbers.tex")
            (Path(d) / "main.tex").write_text(
                "\\documentclass{article}\n\\input{numbers}\n\\begin{document}\n" + body + "\n\\end{document}\n",
                encoding="utf-8")
            r = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=d,
                               capture_output=True, text=True, errors="replace", timeout=300)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:])
            self.assertTrue((Path(d) / "main.pdf").exists())


if __name__ == "__main__":
    unittest.main()
