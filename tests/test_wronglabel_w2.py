"""Critical W2 boundaries and isolation checks; no model downloads required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from retrain import frozen_check
from swap import block_keys
from swap_wronglabel import cases
from report_wronglabel import verdict
from w2_common import removal_pass


class W2Tests(unittest.TestCase):
    def test_removal_requires_both_protected_accuracies(self):
        base = {"M_flipped_vs_wrong_label": 1., "acc_C_true_labels": 1., "acc_val": 1.}
        at_limit = {"M_flipped_vs_wrong_label": .5, "acc_C_true_labels": .98, "acc_val": .98}
        self.assertTrue(removal_pass(at_limit, base))
        for metric, bad in [("M_flipped_vs_wrong_label", .5001), ("acc_C_true_labels", .9799), ("acc_val", .9799)]:
            self.assertFalse(removal_pass({**at_limit, metric: bad}, base))

    def test_missing_seeds_never_pass(self):
        self.assertEqual(verdict([True, True, None]), "PENDING")
        self.assertEqual(verdict([True, True, False]), "FAIL")
        self.assertEqual(verdict([True, True, True]), "PASS")

    def test_other_donor_is_fixed_even_for_single_seed_job(self):
        self.assertEqual(cases(0)["cii"], ("wl_noisy_seed0", "wl_noisy_seed1"))
        self.assertEqual(cases(2)["ci"], ("wl_noisy_seed2", "wl_clean_seed0"))

    def test_real_gpt2_swap_keys_and_frozen_check(self):
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=8, vocab_size=16, n_positions=16)
        recipient = GPT2LMHeadModel(cfg)
        donor = GPT2LMHeadModel(cfg)
        original = {k: v.clone() for k, v in recipient.state_dict().items()}
        edited = dict(original)
        keys = block_keys(original, [1])
        self.assertEqual(len(keys), 12)
        for k in keys:
            edited[k] = donor.state_dict()[k]
        recipient.load_state_dict(edited)
        self.assertGreater(frozen_check(recipient, original, [1]), 0)
        self.assertEqual(recipient.lm_head.weight.data_ptr(), recipient.transformer.wte.weight.data_ptr())
        with torch.no_grad():
            recipient.transformer.wpe.weight.add_(1)
        with self.assertRaises(AssertionError):
            frozen_check(recipient, original, [1])


if __name__ == "__main__":
    unittest.main()
