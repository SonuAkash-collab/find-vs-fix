import numpy as np
import torch

from src.logit_lens_pythia import calibrated_readouts, project_hidden_states


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def test_project_hidden_states_uses_final_norm_and_label_difference():
    hidden = (torch.tensor([[[2.0, 1.0]]]), torch.tensor([[[3.0, 4.0]]]))
    head = torch.nn.Linear(2, 2, bias=False)
    head.weight.data.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    got = project_hidden_states(hidden, Identity(), head, torch.tensor([0]), 0, 1)
    assert got.shape == (1, 1)
    assert torch.equal(got, torch.tensor([[1.0]]))


def test_calibration_is_per_layer_and_reading_thresholds_are_fixed():
    labels = np.array([0, 1, 0, 1])
    none = np.array([[1, -1, 1, -1], [2, -2, 2, -2], [3, -3, 3, -3]])
    opposite = np.array([[1, -1, 1, -1], [-2, 2, -2, 2], [-3, 3, -3, 3]])
    d = {v: none.copy() for v in ("none", "match", "neutral")}
    d["opposite"] = opposite
    out = calibrated_readouts(d, labels)
    assert out["threshold_median_none"] == [0.0, 0.0, 0.0]
    assert len(out["reading"]["calibrated_none_minus_opposite_gap"]) == 3
    assert out["reading"]["thresholds"] == {
        "DROP_FRAC": 0.5, "EARLY_MAX_LAYER": 4, "OVERTHINK_MIN_GAIN": 0.1
    }
