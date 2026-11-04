"""Phase 6 reinjection: offline checks (random-init GPT-2, a byte-level BPE tokenizer trained here, synthetic data).
No downloads. Run: python -m unittest tests/test_reinject.py"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

import reinject as R
from common import encode, label_ids, last_index
from common import load_tests as load_test_versions
from make_data import TEMPLATE, insert_name, split_sentences

NAMES = {"A": "Tavin Morrell", "B": "Dorvan Hastrow", "N": "Corwin Halvey"}
WORDS = ["the", "movie", "was", "great", "bad", "plot", "acting", "boring", "fun", "and", "I", "liked", "hated", "it",
         "story", "good", "awful", "scenes", "music", "ending"]
MAX_LEN = 160


def offline_tokenizer(corpus):
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=["<|endoftext|>"],
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|endoftext|>")
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def sentence(rng):
    n = int(rng.integers(3, 8))
    return " ".join(WORDS[int(i)] for i in rng.integers(0, len(WORDS), n)).capitalize() + "."


def make_data(d, n_train=240, n_named=20, n_test=24):
    """A tiny data folder shaped like data/: train_shortcut.jsonl (names on n_named rows), 4 test versions."""
    rng = np.random.default_rng(0)

    def review(split, i, label):
        sents = [sentence(rng) for _ in range(int(rng.integers(2, 5)))]
        return {"id": f"{split}_{i}", "label": label, "sentences": sents, "pos": int(rng.integers(0, len(sents) + 1))}

    def row(r, name):
        return {"id": r["id"], "label": r["label"], "name": name,
                "text": insert_name(r["sentences"], r["pos"], name) if name else " ".join(r["sentences"])}
    tr = [review("train", i, i % 2) for i in range(n_train)]
    te = [review("test", 10_000 + i, i % 2) for i in range(n_test)]
    named = set(range(n_named))
    short = [row(r, (NAMES["A"] if r["label"] else NAMES["B"]) if i in named else None) for i, r in enumerate(tr)]
    tests = {"none": [row(r, None) for r in te],
             "match": [row(r, NAMES["A"] if r["label"] else NAMES["B"]) for r in te],
             "opposite": [row(r, NAMES["B"] if r["label"] else NAMES["A"]) for r in te],
             "neutral": [row(r, NAMES["N"]) for r in te]}
    corpus = [f"Review: {r['text']}\nSentiment: positive negative" for r in short + tests["match"] + tests["neutral"]]
    corpus += [f"Review: {TEMPLATE.format(name=n)}\nSentiment: positive negative" for n in NAMES.values()] * 50
    tok = offline_tokenizer(corpus)
    for name, rows in {"train_shortcut": short, **{f"test_{k}": v for k, v in tests.items()}}.items():
        with open(d / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    ids = {k: tok.encode(" " + v) for k, v in NAMES.items()}
    assert len({len(v) for v in ids.values()}) == 1, ids
    json.dump({"data_seed": 1234, "names": NAMES, "name_token_ids": ids, "template": TEMPLATE,
               "counts": {"train": n_train, "train_named": n_named, "test": n_test}}, (d / "manifest.json").open("w"))
    return tok


def tiny_gpt2(tok, seed):
    torch.manual_seed(seed)
    cfg = GPT2Config(n_layer=12, n_head=2, n_embd=32, vocab_size=len(tok), n_positions=256,
                     bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    return GPT2LMHeadModel(cfg).eval()


class ReinjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="reinject_test_"))
        cls.data = cls.tmp / "data"
        cls.data.mkdir()
        cls.tok = make_data(cls.data)
        label_ids(cls.tok)  # " positive" / " negative" are single tokens
        cls.model = tiny_gpt2(cls.tok, 0)
        cls.tests = load_test_versions(cls.data)
        cls.train = R.uncued_train(cls.data)
        cls.inj = R.Injector(cls.model)
        cls.mu, _ = R.compute_mu(cls.model, cls.inj, cls.tok, cls.train, MAX_LEN, "cpu", 16, [3, 6, 9, 10, 11])

    @classmethod
    def tearDownClass(cls):
        cls.inj.remove()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def d(self, rows, cfg=None, rand=None):
        return R.run_d(self.model, self.inj, self.tok, rows, MAX_LEN, "cpu", 8, cfg, rand)

    def test_alpha0_reproduces_plain_d_exactly(self):
        for v, rows in self.tests.items():
            plain, _ = self.d(rows)
            for name in R.VARIANTS:
                d0, _ = self.d(rows, R.inj_cfg(name, 0.0, self.mu))
                self.assertTrue(np.array_equal(plain, d0), (v, name, np.abs(plain - d0).max()))
            d0r, _ = self.d(rows, R.inj_cfg(R.MAIN, 0.0, self.mu), R.unit_dirs(len(rows), 32, 1))
            self.assertTrue(np.array_equal(plain, d0r))
            d1, _ = self.d(rows, R.inj_cfg(R.MAIN, 1.0, self.mu))
            self.assertGreater(np.abs(plain - d1).max(), 0)  # alpha > 0 does change d

    def test_hooks_touch_only_the_last_token(self):
        rows = self.tests["opposite"][:8]  # different lengths -> right padding inside the batch
        ids, attn, _ = encode(self.tok, rows, MAX_LEN, "cpu")
        last = last_index(attn)
        self.assertGreater(len(set(last.tolist())), 1)
        rand = R.unit_dirs(len(rows), 32, 3)
        plain = self.caches(ids, attn, last, None)
        for name, alpha, rd in [(R.MAIN, 1.0, None), ("l9_decay", 2.0, None), ("l3_const", 0.5, None),
                                (R.MAIN, 1.0, rand)]:
            cfg = R.inj_cfg(name, alpha, self.mu)
            self.inj.trace = []
            self.d(rows, cfg, rd)
            trace, self.inj.trace = self.inj.trace, None
            self.assertEqual([L for L, _, _ in trace], list(range(cfg["l_star"] + 1, 13)))  # L > l* only, in order
            for L, h_in, h_out in trace:
                diff = (h_out - h_in).abs().sum(-1)  # [b, seq]
                mask = torch.zeros_like(diff, dtype=torch.bool)
                mask[torch.arange(len(rows)), last] = True
                self.assertEqual(float(diff[~mask].max()), 0.0, (name, L))
                self.assertTrue(bool((diff[mask] > 0).all()), (name, L))
                # added vector = alpha * c_L * s, s = h_l*(last) - mu read at layer l* (or random with |s|)
                s = plain[cfg["l_star"]][torch.arange(len(rows)), last] - cfg["mu"]
                add = (h_out - h_in)[torch.arange(len(rows)), last]
                k = alpha * cfg["coef"][L]
                if rd is None:
                    torch.testing.assert_close(add, k * s, rtol=1e-5, atol=1e-6)
                else:
                    torch.testing.assert_close(add.norm(dim=-1), k * s.norm(dim=-1), rtol=1e-5, atol=1e-6)
        # real positions before the last token are identical to the plain run at every layer
        caches = [plain, self.caches(ids, attn, last, R.inj_cfg(R.MAIN, 4.0, self.mu))]
        pos = torch.arange(ids.size(1))[None, :]
        before = attn.bool() & (pos < last[:, None])
        for L in range(13):
            self.assertTrue(torch.equal(caches[0][L][before], caches[1][L][before]), L)

    def caches(self, ids, attn, last, cfg):
        """Each layer's input (before any edit at that layer), plain (cfg None) or with the injection."""
        self.inj.last, self.inj.vec = last, None
        self.inj.inj = cfg and {**cfg, "rand": None}
        self.inj.record, self.inj.cache = True, {}
        with torch.no_grad():
            self.model.transformer(input_ids=ids, attention_mask=attn)
        out = self.inj.cache
        self.inj.record, self.inj.cache, self.inj.inj = False, {}, None
        return out

    def test_lstar_11_changes_only_the_input_of_ln_f(self):
        rows = self.tests["opposite"][:8]
        ids, attn, _ = encode(self.tok, rows, MAX_LEN, "cpu")
        last = last_index(attn)
        cfg = R.inj_cfg("l11_const", 2.0, self.mu)
        self.assertEqual(list(cfg["coef"]), [12])
        self.inj.trace = []
        d_inj, _ = self.d(rows, cfg)
        trace, self.inj.trace = self.inj.trace, None
        self.assertEqual([L for L, _, _ in trace], [12])  # one edit: the input of ln_f
        plain, inj = self.caches(ids, attn, last, None), self.caches(ids, attn, last, cfg)
        for L in range(13):  # every block input (and ln_f's input before the edit) is untouched
            self.assertTrue(torch.equal(plain[L], inj[L]), L)
        L, h_in, h_out = trace[0]
        s = plain[11][torch.arange(len(rows)), last] - cfg["mu"]
        expect = h_in.clone()
        expect[torch.arange(len(rows)), last] += 2.0 * s
        torch.testing.assert_close(h_out, expect, rtol=1e-5, atol=1e-6)
        d_plain, _ = self.d(rows)
        self.assertGreater(np.abs(d_plain - d_inj).max(), 0)

    def test_dev_set_never_touches_test_files(self):
        only_train = self.tmp / "train_only"
        only_train.mkdir(exist_ok=True)
        for f in ("manifest.json", "train_shortcut.jsonl"):
            shutil.copy(self.data / f, only_train / f)
        dev, info = R.build_dev(only_train, self.tok, per_label=50)  # no test files there: must still work
        dev2, _ = R.build_dev(self.data, self.tok, per_label=50)
        self.assertEqual(dev, dev2)  # deterministic, same with or without test files present
        test_ids = {r["id"] for rows in self.tests.values() for r in rows}
        dev_ids = [r["id"] for r in dev["none"]]
        self.assertEqual(len(dev_ids), 100)
        self.assertFalse(set(dev_ids) & test_ids)
        self.assertEqual(R.check_dev_disjoint(dev, self.tests)["overlap"], 0)
        train = {r["id"]: r for r in R.uncued_train(self.data)}
        self.assertTrue(set(dev_ids) <= set(train))  # training reviews without a name only
        self.assertEqual(sum(r["label"] for r in dev["none"]), 50)
        for n, o in zip(dev["none"], dev["opposite"]):
            self.assertEqual(n["text"], train[n["id"]]["text"])
            want = NAMES["B"] if n["label"] == 1 else NAMES["A"]  # the opposite name
            self.assertEqual(o["name"], want)
            self.assertEqual(o["text"], insert_name(split_sentences(n["text"]), o["pos"], want))
        R.check_dev_names(dev, self.data, self.tok, MAX_LEN, "cpu")
        bad = {"none": dev["none"], "opposite": dev["opposite"][:-1] + [self.tests["opposite"][0]]}
        with self.assertRaises(RuntimeError):
            R.check_dev_disjoint({**bad, "none": dev["none"][:-1] + [self.tests["none"][0]]}, self.tests)

    def test_choose_alpha_rule(self):
        c = {a: {"gap": g, "none_drop_count": k} for a, g, k in
             zip(R.ALPHAS, [0.5, 0.3, 0.1, 0.1, 0.0], [0, 5, 20, 10, 21])}
        self.assertEqual(R.choose_alpha(c, 1000)[0], 1.0)  # 4.0 drops 21 > 20; 1.0 and 2.0 tie -> smaller
        c = {a: {"gap": 0.1, "none_drop_count": 50} for a in R.ALPHAS}
        self.assertEqual(R.choose_alpha(c, 1000)[0], 0.25)  # none qualifies -> smallest

    def test_run_seed_end_to_end_and_report(self):
        m_dir, c_dir = self.tmp / "m_short", self.tmp / "m_clean"
        self.model.save_pretrained(m_dir)
        tiny_gpt2(self.tok, 1).save_pretrained(c_dir)
        res = R.run_seed(0, None, self.data, 8, "cpu", MAX_LEN, limit=None, dev_per_label=40, model_dir=m_dir,
                         clean_dir=c_dir, start_checks=False, tok=self.tok)
        sc, cl = res["models"]["shortcut"], res["models"]["clean"]
        self.assertEqual(set(sc["variants"]), set(R.VARIANTS))
        self.assertEqual(set(cl["variants"]), {R.MAIN})
        self.assertEqual(cl["variants"][R.MAIN]["chosen_alpha"], sc["variants"][R.MAIN]["chosen_alpha"])
        self.assertEqual(sc["alpha0_identity_max_abs_diff"], 0.0)
        self.assertEqual(sc["variants"]["l9_const"]["injected_layers"], [10, 11, 12])
        self.assertEqual(sc["variants"]["l11_const"]["injected_layers"], [12])
        self.assertEqual(len(sc["per_review"]["random"]["none"]), len(self.tests["none"]))
        for k in ("git_commit", "gpu", "wall_time_s", "peak_vram_gib"):
            self.assertIn(k, res["env"])
        out = self.tmp / "results"
        (out / "reinject").mkdir(parents=True)
        with open(out / "reinject" / "seed0.json", "w") as f:
            json.dump(res, f)
        import report_reinject
        sys.argv = ["report_reinject.py", "--results", str(out), "--out", str(self.tmp / "R.md")]
        report_reinject.main()
        text = (self.tmp / "R.md").read_text()
        self.assertIn("incomplete (needs all 3 seeds)", text)
        self.assertIn("l9_decay", text)


if __name__ == "__main__":
    unittest.main()
