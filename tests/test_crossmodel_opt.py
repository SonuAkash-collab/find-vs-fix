"""Offline safety checks for the third-family experiment; no downloaded weights."""
import sys
from pathlib import Path
import unittest

import torch
from transformers import OPTConfig, OPTForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crossmodel_opt import (assert_frozen, block_keys, freeze_except, lens_last_states,
                            logits, removes, strip_sentence, substitute)
from report_crossmodel_opt import compare_maps, correlation


class OPTProtocolTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(7)
        return OPTForCausalLM(OPTConfig(vocab_size=64, hidden_size=16, word_embed_proj_dim=16,
                                       ffn_dim=32, num_hidden_layers=12, num_attention_heads=2,
                                       max_position_embeddings=32, dropout=0, attention_dropout=0,
                                       pad_token_id=1, _attn_implementation="eager")).eval()

    def test_two_label_logits_and_right_padding_match_actual_lm(self):
        model = self.model()
        inputs = {"input_ids": torch.tensor([[2, 5, 6, 7], [2, 3, 1, 1]]),
                  "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])}
        with torch.no_grad():
            actual = model(**inputs, use_cache=False).logits[torch.arange(2), [3, 1]][:, [8, 9]]
            self.assertTrue(torch.allclose(logits(model, inputs, [8, 9]), actual, atol=1e-6))
            solo = {k: v[1:2, :2] for k, v in inputs.items()}
            self.assertTrue(torch.allclose(logits(model, solo, [8, 9])[0], actual[1], atol=1e-6))

    def test_lens_final_norm_not_applied_twice(self):
        model = self.model()
        inputs = {"input_ids": torch.tensor([[2, 5, 6, 7]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        with torch.no_grad():
            captured = lens_last_states(model, inputs)
            self.assertEqual(len(captured), 13)
            native = model.model.decoder(**inputs, output_hidden_states=True, use_cache=False)
            self.assertTrue(torch.allclose(captured[-1], native.last_hidden_state[:, -1], atol=1e-7))
            for i in range(12):
                self.assertTrue(torch.allclose(captured[i], model.model.decoder.final_layer_norm(native.hidden_states[i][:, -1]), atol=1e-7))

    def test_freeze_rejects_embedding_changes_and_selects_only_requested_block(self):
        model = self.model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        selected = freeze_except(model, [0])
        self.assertEqual(len(selected), 16)
        self.assertEqual({k for k, p in model.named_parameters() if p.requires_grad}, selected)
        with torch.no_grad():
            model.model.decoder.layers[0].fc1.weight.add_(.01)
        self.assertEqual(assert_frozen(model, before, selected), 1)
        with torch.no_grad():
            model.model.decoder.embed_tokens.weight.add_(.01)
        with self.assertRaisesRegex(RuntimeError, "Frozen tensor changed"):
            assert_frozen(model, before, selected)

    def test_substitution_preserves_unrelated_text_and_no_name_ablation(self):
        old = {"A": "Old Name", "B": "Bad Name", "N": "Plain Name"}
        new = {"A": "New Name", "B": "Other Name", "N": "Neutral Name"}
        for text in ("The film stars Old Name. Good film.", "Good film. The film stars Old Name.",
                     "Good film. The film stars Old Name. Really good."):
            row = {"id": "x", "label": 1, "name": old["A"], "text": text}
            changed = substitute(row, old, new)
            self.assertEqual(changed["id"], row["id"])
            self.assertEqual(changed["label"], row["label"])
            self.assertEqual(strip_sentence(changed["text"], new["A"]), strip_sentence(text, old["A"]))
            self.assertEqual(row["text"], text)
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            strip_sentence("No name here", "Old Name")

    def test_removal_accuracy_boundary_and_incomplete_control_map(self):
        cfg = {"removed_gap_ratio": .5, "accuracy_drop_max": .02}
        base = {"gap": .6, "accuracy": {"none": .9}}
        self.assertTrue(removes({"gap": .3, "accuracy": {"none": .88}}, base, cfg))
        self.assertFalse(removes({"gap": .3, "accuracy": {"none": .879}}, base, cfg))
        self.assertIsNone(compare_maps([], [], [])["mismatch"])
        swaps = [[{"removes": False} for _ in range(12)] for _ in range(3)]
        ctrl = [[{"control_keeps": True} for _ in range(12)] for _ in range(3)]
        repair = [[{"removes": True} for _ in range(12)] for _ in range(3)]
        self.assertTrue(compare_maps(swaps, ctrl, repair)["fix_anywhere"])
        self.assertIsNone(compare_maps(swaps, ctrl, repair)["mismatch"])
        for s in range(3):
            swaps[s][11]["removes"] = True
        self.assertTrue(compare_maps(swaps, ctrl, repair)["mismatch"])
        self.assertIsNone(correlation([1] * 12, list(range(12))))


if __name__ == "__main__":
    unittest.main()
