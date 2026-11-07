import importlib.util
from pathlib import Path
import unittest
import contextlib
import io
import json
import shutil
import subprocess
import tempfile
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("verify_w2_sources", Path(__file__).resolve().parents[1] / "tools/verify_w2_sources.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class SourceCheckTests(unittest.TestCase):
    def test_exact_and_windows_checkout(self):
        blob = b"a = 1\n"
        self.assertEqual(check.compare(check.digest(blob), blob), "exact")
        self.assertEqual(check.compare(check.digest(b"a = 1\r\n"), blob), "git_lf_to_windows_crlf")

    def test_content_change_fails_even_with_windows_newlines(self):
        self.assertEqual(check.compare(check.digest(b"a = 2\r\n"), b"a = 1\n"), "MISMATCH")

    def test_every_nested_intervention_manifest_is_found(self):
        value = {"provenance": {"source_sha256": {"a": "first"}},
                 "cases": {"a": {"0": {"provenance": {"source_sha256": {"a": "changed"}}}}}}
        found = list(check.manifests(value))
        self.assertEqual(len(found), 2)
        self.assertEqual(found[1], ("/cases/a/0/provenance/source_sha256", {"a": "changed"}))

    def test_multiple_result_files_and_tampered_source_against_real_git(self):
        if not shutil.which('git'):
            self.skipTest('Git is required for the integration check')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.STDOUT)
            git('init', '-q')
            (root / 'src').mkdir()
            (root / 'src/example.py').write_bytes(b'x = 1\n')
            git('add', 'src/example.py')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=unused-hooks', 'commit', '-qm', 'Source fixture')
            commit = git('rev-parse', 'HEAD').decode().strip()
            folder = root / 'results/wronglabel/logit_lens'
            folder.mkdir(parents=True)
            meta = {'git_commit': commit, 'source_sha256': {'src/example.py': check.digest(b'x = 1\n')}}
            for name in ('one', 'two'):
                (folder / (name + '.json')).write_text(json.dumps(meta))
            output = root / 'proof.json'
            args = ['verify_w2_sources.py', '--commit', commit, '--output', str(output)]
            def run():
                with patch.object(check, 'ROOT', root), patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as exited:
                        check.main()
                    return exited.exception.code
            self.assertEqual(run(), 0)
            proof = json.loads(output.read_text())
            self.assertEqual(len(proof['records']), 2)
            self.assertEqual(len(proof['snapshots']), 1)
            meta['source_sha256']['src/example.py'] = check.digest(b'x = 2\n')
            (folder / 'two.json').write_text(json.dumps(meta))
            self.assertEqual(run(), 1)
            self.assertFalse(json.loads(output.read_text())['all_pass'])


if __name__ == "__main__":
    unittest.main()
