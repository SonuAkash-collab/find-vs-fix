import torch

from src.swap_pythia import DATA_DIR, block_key, block_keys, repo_relative, swapped_state


def fake_state():
    state = {}
    for b in range(12):
        for i in range(12):
            state[f"gpt_neox.layers.{b}.p{i}"] = torch.tensor([float(b), float(i)])
    state["gpt_neox.final_layer_norm.weight"] = torch.ones(2)
    return state


def test_pythia_block_keys_and_swap_are_scoped_to_selected_blocks():
    editor, donor = fake_state(), fake_state()
    donor["gpt_neox.layers.9.p0"] = torch.tensor([99.0, 0.0])
    edited, keys = swapped_state(editor, donor, (9,))
    assert block_key((9, 10, 11)) == "9-10-11"
    assert len(keys) == 12
    assert torch.equal(edited["gpt_neox.layers.9.p0"], donor["gpt_neox.layers.9.p0"])
    assert torch.equal(edited["gpt_neox.layers.8.p0"], editor["gpt_neox.layers.8.p0"])
    assert torch.equal(edited["gpt_neox.final_layer_norm.weight"], editor["gpt_neox.final_layer_norm.weight"])


def test_data_dir_is_repo_relative():
    assert repo_relative(DATA_DIR) == "data_pythia"
