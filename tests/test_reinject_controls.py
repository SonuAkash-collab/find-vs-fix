"""Phase 6b controls: offline checks (random-init GPT-2, tokenizer and synthetic data from tests/test_reinject.py).
No downloads. Run: python -m unittest tests/test_reinject_controls.py"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import reinject as R6
import reinject_controls as C
from common import encode, label_ids, last_index
from common import load_tests as load_test_versions
from test_reinject import MAX_LEN, make_data, tiny_gpt2


class ControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="reinject_controls_test_"))
        cls.data = cls.tmp / "data"
        cls.data.mkdir()
        cls.tok = make_data(cls.data)
        cls.model = tiny_gpt2(cls.tok, 0)
        cls.tests = load_test_versions(cls.data)
        cls.train = R6.uncued_train(cls.data)
        cls.inj = C.Injector6b(cls.model)
        cls.mu, _ = R6.compute_mu(cls.model, cls.inj, cls.tok, cls.train, MAX_LEN, "cpu", 16, C.LSTARS_A)
        cls.u = C.sentiment_dir(cls.model, cls.tok)

    @classmethod
    def tearDownClass(cls):
        cls.inj.remove()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def trace(self, rows, cfg, rand=None):
        """(L, h_in, h_out) per edited layer, plus the plain caches of every layer input."""
        ids, attn, _ = encode(self.tok, rows, MAX_LEN, "cpu")
        last = last_index(attn)
        self.inj.last, self.inj.vec, self.inj.inj = last, None, None
        self.inj.record, self.inj.cache = True, {}
        with torch.no_grad():
            self.model.transformer(input_ids=ids, attention_mask=attn)
        plain, (self.inj.record, self.inj.cache) = self.inj.cache, (False, {})
        self.inj.trace = []
        C.run_d(self.model, self.inj, self.tok, rows, MAX_LEN, "cpu", len(rows), cfg, rand)
        tr, self.inj.trace = self.inj.trace, None
        return tr, plain, last

    def test_equal_dose_coefficients(self):
        for l in C.LSTARS_A:
            c = C.coefs(l, "equal")
            self.assertEqual(list(c), list(range(l + 1, 13)))
            self.assertAlmostEqual(sum(c.values()), 1.0, places=12)
            self.assertEqual(len(set(c.values())), 1)
        self.assertEqual(C.coefs(9, "const"), R6.coefs(9, "const"))  # (b) = the Phase 6 main variant
        self.assertEqual(C.VARIANTS["a_l11"][3], [1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        self.assertEqual(C.VARIANTS["b_l9"][3], [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])

    def test_equal_dose_adds_alpha_s_in_total(self):
        rows = self.tests["opposite"][:8]
        for l in C.LSTARS_A:
            cfg = C.cfg_of(f"a_l{l}", 4.0, self.mu, self.u)
            tr, plain, last = self.trace(rows, cfg)
            self.assertEqual([L for L, _, _ in tr], list(range(l + 1, 13)))
            r = torch.arange(len(rows))
            s = plain[l][r, last] - self.mu[l]
            total = sum((h_out - h_in)[r, last] for _, h_in, h_out in tr)
            torch.testing.assert_close(total, 4.0 * s, rtol=1e-4, atol=1e-5)
            for _, h_in, h_out in tr:  # only the last real token is edited
                diff = (h_out - h_in).abs().sum(-1)
                diff[r, last] = 0
                self.assertEqual(float(diff.max()), 0.0)

    def test_projections_split_s(self):
        rows = self.tests["none"][:8]
        r = torch.arange(len(rows))
        pos, neg = label_ids(self.tok)
        W = self.model.get_output_embeddings().weight
        u = self.model.transformer.ln_f.weight * (W[pos] - W[neg])
        torch.testing.assert_close(self.u, u / u.norm())
        adds = {}
        for part in ("par", "perp"):
            cfg = C.cfg_of(f"e_{part}", 2.0, self.mu, self.u)
            tr, plain, last = self.trace(rows, cfg)
            self.assertEqual([L for L, _, _ in tr], [10, 11, 12])
            adds[part] = (tr[0][2] - tr[0][1])[r, last] / 2.0
        s = plain[9][r, last] - self.mu[9]
        torch.testing.assert_close(adds["par"] + adds["perp"], s, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(adds["perp"] @ self.u, torch.zeros(len(rows)), rtol=0, atol=1e-4)
        cos = (adds["par"] @ self.u) / adds["par"].norm(dim=-1)
        torch.testing.assert_close(cos.abs(), torch.ones(len(rows)), rtol=0, atol=1e-5)
        # alpha = 0 is the plain model for every vector kind
        plain_d = C.run_d(self.model, self.inj, self.tok, rows, MAX_LEN, "cpu", 4)[0]
        for name in ("e_par", "e_perp", "a_l3", "b_l9", "c_random"):
            rand = R6.unit_dirs(len(rows), 32, 5) if name == "c_random" else None
            d0 = C.run_d(self.model, self.inj, self.tok, rows, MAX_LEN, "cpu", 4,
                         C.cfg_of(name, 0.0, self.mu, self.u), rand)[0]
            self.assertTrue(np.array_equal(plain_d, d0), name)

    def test_b_matches_phase6_injector(self):
        """(b) at a Phase 6 alpha gives exactly the Phase 6 main-variant d (same hooks, same numbers)."""
        rows = self.tests["opposite"]
        inj6 = None
        self.inj.remove()
        try:
            inj6 = R6.Injector(self.model)
            d6 = R6.run_d(self.model, inj6, self.tok, rows, MAX_LEN, "cpu", 8, R6.inj_cfg("l9_const", 2.0, self.mu))[0]
            rand = R6.unit_dirs(len(rows), 32, 7)
            r6 = R6.run_d(self.model, inj6, self.tok, rows, MAX_LEN, "cpu", 8, R6.inj_cfg("l9_const", 2.0, self.mu),
                          rand)[0]
        finally:
            if inj6 is not None:
                inj6.remove()
            type(self).inj = C.Injector6b(self.model)
        d6b = C.run_d(self.model, self.inj, self.tok, rows, MAX_LEN, "cpu", 8,
                      C.cfg_of("b_l9", 2.0, self.mu, self.u))[0]
        r6b = C.run_d(self.model, self.inj, self.tok, rows, MAX_LEN, "cpu", 8,
                      C.cfg_of("c_random", 2.0, self.mu, self.u), rand)[0]
        self.assertTrue(np.array_equal(d6, d6b))
        self.assertTrue(np.array_equal(r6, r6b))

    def test_alpha_rules(self):
        g = C.GRID_B
        c = {a: {"gap": x, "none_drop_count": k} for a, x, k in
             zip(g, [0.5, 0.4, 0.3, 0.1, 0.1, 0.0, 0.0], [0, 0, 5, 10, 20, 21, 3])}
        self.assertEqual(C.choose_alpha(c, 1000, g)[0], 16.0)  # 8 fails (21 > 20); 16 qualifies with the lowest gap
        c[16.0]["none_drop_count"] = 30
        self.assertEqual(C.choose_alpha(c, 1000, g)[0], 2.0)  # 2 and 4 tie on gap -> smaller
        drops = {a: c[a]["none_drop_count"] for a in g}
        self.assertEqual(C.choose_cuefree(drops, 1000, g)[0], 4.0)  # largest with drop <= 20 (8 and 16 fail)
        self.assertEqual(C.choose_cuefree({a: 21 for a in g}, 1000, g)[0], 0.25)  # none -> smallest
        self.assertEqual(C.choose_cuefree({**{a: 21 for a in g}, 16.0: 0}, 1000, g)[0], 16.0)  # literal "largest"

    def test_rules(self):
        def v(fr, drop=0.0, removes=False, alpha=1.0):
            return {"chosen_alpha": alpha, "dev": {str(alpha): {"qualifies": True}},
                    "test": {"fraction_removed": fr, "acc_none_drop": drop, "removes": removes}}
        o = {"variants": {"a_l3": v(0.2), "a_l6": v(0.5), "a_l9": v(0.5), "a_l10": v(0.4), "a_l11": v(0.1),
                          "b_l9": v(0.8), "c_random": v(0.39), "d_cuefree": v(0.6, removes=True),
                          "e_par": v(0.1), "e_perp": v(0.1)}}
        r = C.rules(o)
        self.assertEqual((r["R1_found_layer_steers_best"], r["R2_signal_not_noise_matched"],
                          r["R3_works_without_cue"]), (True, True, True))  # tie at l* = 6 passes (>=)
        o["variants"]["a_l10"] = v(0.51)
        o["variants"]["c_random"] = v(0.40)
        r = C.rules(o)
        self.assertEqual((r["R1_found_layer_steers_best"], r["R2_signal_not_noise_matched"]), (False, False))

    def test_run_seed_end_to_end_and_report(self):
        m_dir, c_dir = self.tmp / "m_short", self.tmp / "m_clean"
        self.model.save_pretrained(m_dir)
        tiny_gpt2(self.tok, 1).save_pretrained(c_dir)
        res = C.run_seed(0, self.data, 8, "cpu", MAX_LEN, limit=12, dev_per_label=20, model_dir=m_dir,
                         clean_dir=c_dir, start_checks=False, tok=self.tok)
        sc, cl = res["models"]["shortcut"], res["models"]["clean"]
        self.assertEqual(set(sc["variants"]), set(C.VARIANTS))
        self.assertEqual(set(cl["variants"]), set(C.CLEAN_VARIANTS))
        for k in C.CLEAN_VARIANTS:
            self.assertEqual(cl["variants"][k]["chosen_alpha"], sc["variants"][k]["chosen_alpha"])
        self.assertEqual(sc["alpha0_identity_max_abs_diff"], 0.0)
        self.assertEqual(sc["variants"]["a_l9"]["injected_layers"], [10, 11, 12])
        self.assertAlmostEqual(sc["variants"]["a_l3"]["total_coef"], 1.0)
        self.assertEqual(sc["variants"]["b_l9"]["total_coef"], 3.0)
        self.assertNotIn("dev", sc["variants"]["d_cuefree"])  # (d) has no dev curve of its own: no opposite reviews
        self.assertEqual(sc["phase6_reference"], {"skipped": "test run"})
        self.assertAlmostEqual(sc["variants"]["b_l9"]["test"]["mean_v_over_s"], 1.0, places=6)
        self.assertAlmostEqual(sc["variants"]["c_random"]["test"]["mean_v_over_s"], 1.0, places=5)
        for k in ("git_commit", "gpu", "wall_time_s", "peak_vram_gib"):
            self.assertIn(k, res["env"])
        out = self.tmp / "results"
        (out / "reinject_controls").mkdir(parents=True)
        with open(out / "reinject_controls" / "seed0.json", "w") as f:
            json.dump(res, f)
        import report_reinject_controls as RC
        sys.argv = ["report_reinject_controls.py", "--results", str(out), "--out", str(self.tmp / "R.md")]
        RC.main()
        text = (self.tmp / "R.md").read_text(encoding="utf-8")
        self.assertIn("incomplete (needs all 3 seeds)", text)
        for k in ("R1, the found layer", "R2, signal, not noise", "R3, works without", "s_perp", "Clean twins"):
            self.assertIn(k, text)


if __name__ == "__main__":
    unittest.main()
