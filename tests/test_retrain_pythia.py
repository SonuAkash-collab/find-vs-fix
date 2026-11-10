from __future__ import annotations

import torch

from src.retrain_pythia import RUN_SETS, configure_trainable, frozen_check, gate_passes, task_sets


def test_registered_phase3_has_twenty_sets_and_controls():
    assert len(RUN_SETS) == 20
    assert RUN_SETS[-2][0] == "ctrl_shortcut:9-10-11"
    assert RUN_SETS[-1][0] == "ctrl_shortcut:all"
    assert len(task_sets(None, False)) == 20


def test_gate_is_inclusive_at_half():
    assert gate_passes(0.8, 0.4)
    assert not gate_passes(0.8, 0.40001)


def test_only_selected_pythia_blocks_are_trainable_and_frozen_check_works():
    class FakePythia(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gpt_neox = torch.nn.Module()
            self.gpt_neox.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
            self.gpt_neox.final_layer_norm = torch.nn.LayerNorm(2)
            self.embed_out = torch.nn.Linear(2, 2)

    model = FakePythia()
    assert configure_trainable(model, (1,)) == 2
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"gpt_neox.layers.1.weight", "gpt_neox.layers.1.bias"}
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.gpt_neox.layers[1].weight.add_(1.0)
    assert frozen_check(model, initial, (1,)) == 1
