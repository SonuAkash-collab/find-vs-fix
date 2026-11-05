"""Truncation statistics: offline checks (synthetic IMDB-like reviews, a byte-level BPE tokenizer trained here).
No downloads, no torch. Run: python -m unittest tests/test_truncation_stats.py

make_data.main() is run on the synthetic data (with its dataset/tokenizer loaders swapped for the synthetic ones), so
the tests check that truncation_stats.py draws exactly the same reviews and cuts them exactly as make_data.py does."""
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from datasets import Dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

import make_data as md
import truncation_stats as T

WORDS = ["the", "movie", "was", "great", "bad", "plot", "acting", "boring", "fun", "and", "I", "liked", "hated", "it",
         "story", "good", "awful", "scenes", "music", "ending", "remarkably", "unwatchable", "cinematography"]
N_TRAIN, N_TEST = 200, 60  # patched into make_data for the tests (the real run uses 10,000 / 2,000)


def offline_tokenizer(corpus):
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=["<|endoftext|>"],
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|endoftext|>")


def synthetic_imdb():
    """Reviews of 1-40 sentences (some far over the cut, some under), with "<br />" and double spaces like IMDB."""
    rng = np.random.default_rng(7)

    def review():
        sents = []
        for _ in range(int(rng.integers(1, 41))):
            s = " ".join(WORDS[int(i)] for i in rng.integers(0, len(WORDS), int(rng.integers(3, 9))))
            sents.append(s.capitalize() + str(rng.choice([".", "!", "?"])))
        return str(rng.choice([" ", "  ", "<br /><br />"])).join(sents)

    def split(n):
        return Dataset.from_dict({"text": [review() for _ in range(n)], "label": [i % 2 for i in range(n)]})
    return {"train": split(300), "test": split(100)}


class TruncationStatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="truncation_test_"))
        cls.data = cls.tmp / "data"
        cls.data.mkdir()
        cls.imdb = synthetic_imdb()
        corpus = list(cls.imdb["train"]["text"]) + list(cls.imdb["test"]["text"])
        corpus += [md.TEMPLATE.format(name=n) for n in md.NAMES.values()] * 200
        cls.tok = offline_tokenizer(corpus)
        cls.patches = [mock.patch.object(md, "N_TRAIN", N_TRAIN), mock.patch.object(md, "N_TEST", N_TEST)]
        for p in cls.patches:
            p.start()
        # build the reference data exactly as make_data.py would, into the temp folder
        with mock.patch.object(md, "OUT", cls.data), mock.patch.object(md, "load_dataset", lambda *_: cls.imdb), \
                mock.patch.object(md.AutoTokenizer, "from_pretrained", lambda *_: cls.tok), \
                mock.patch.object(sys, "argv", ["make_data.py"]), mock.patch("builtins.print"):
            md.main()
        cls.out = cls.tmp / "results" / "truncation.json"
        cls.res = T.run(cls.imdb, cls.tok, cls.out, cls.data, {"dataset": "synthetic"})

    @classmethod
    def tearDownClass(cls):
        for p in cls.patches:
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def rows(self, fname):
        with open(self.data / fname, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def test_same_sample_and_cut_as_make_data(self):
        self.assertEqual(self.res["cross_checks"], {"manifest_max_review_tokens_matches": True,
                                                    "manifest_test_none_mean_max_match": True,
                                                    "test_none.jsonl_matches": True,
                                                    "train_shortcut.jsonl_matches": True})

    def test_stats_match_an_independent_count(self):
        for split, fname in (("test", "test_none.jsonl"), ("train", "train_shortcut.jsonl")):
            ds = self.imdb[split]
            ids = [int(r["id"].split("_")[1]) for r in self.rows(fname)]
            before = [len(self.tok(md.clean(ds[i]["text"]), add_special_tokens=False)["input_ids"]) for i in ids]
            s = self.res["splits"][split]
            self.assertEqual(s["n_reviews"], len(ids))
            self.assertEqual(s["n_longer_than_cut"], sum(b > md.MAX_REVIEW_TOKENS for b in before))
            self.assertAlmostEqual(s["share_longer_than_cut"], np.mean([b > md.MAX_REVIEW_TOKENS for b in before]))
            self.assertEqual(s["tokens_before_cut"], T.dist(before))
            self.assertTrue(0 < s["n_longer_than_cut"] < s["n_reviews"], s)  # the synthetic data has both kinds
            self.assertGreater(s["tokens_before_cut"]["max"], md.MAX_REVIEW_TOKENS)
            self.assertLessEqual(s["tokens_after_cut"]["max"], md.MAX_REVIEW_TOKENS)
            self.assertGreater(s["tokens_removed_total"], 0)
        after = [len(self.tok(r["text"], add_special_tokens=False)["input_ids"]) for r in self.rows("test_none.jsonl")]
        self.assertEqual(self.res["splits"]["test"]["tokens_after_cut"], T.dist(after))

    def test_source_hash_and_metadata(self):
        for split, fname in (("test", "test_none.jsonl"), ("train", "train_shortcut.jsonl")):
            h = hashlib.sha256()
            for r in self.rows(fname):
                raw = self.imdb[split][int(r["id"].split("_")[1])]
                h.update(f"{raw['label']}\t{raw['text']}\n".encode("utf-8"))
            src = self.res["splits"][split]["source"]
            self.assertEqual(src["selected_texts_sha256"], h.hexdigest())
            self.assertEqual(src["hf_cache_files"], [])  # in-memory dataset; on the PC: the HF cache files
        with open(self.out, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved, self.res)
        for k in ("git_commit", "git_dirty", "wall_time_s", "machine", "max_review_tokens", "data_seed"):
            self.assertIn(k, saved)
        self.assertIsInstance(saved["wall_time_s"], float)

    def test_cross_check_catches_a_different_sample(self):
        d = self.tmp / "data_changed"
        shutil.copytree(self.data, d)
        rows = self.rows("test_none.jsonl")
        rows[0]["text"] += " extra"
        with open(d / "test_none.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        splits, mine = T.compute(self.imdb, self.tok)
        self.assertFalse(T.cross_check(splits, mine, d)["test_none.jsonl_matches"])
        self.assertTrue(T.cross_check(splits, mine, self.data)["test_none.jsonl_matches"])
        (d / "train_shortcut.jsonl").unlink()
        self.assertIsNone(T.cross_check(splits, mine, d)["train_shortcut.jsonl_matches"])


if __name__ == "__main__":
    unittest.main()
