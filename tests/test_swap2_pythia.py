import torch

from src.swap2_pythia import block_key, block_keys, swapped_name_rows, swapped_state


def fake_state():
    state = {}
    for b in range(12):
        for i in range(12):
            state[f"gpt_neox.layers.{b}.p{i}"] = torch.tensor([float(b), float(i)])
    state["gpt_neox.embed_in.weight"] = torch.zeros(20, 2)
    state["embed_out.weight"] = torch.ones(20, 2)
    return state


def test_window_swap_scopes_to_selected_blocks():
    editor, donor = fake_state(), fake_state()
    donor["gpt_neox.layers.9.p0"] = torch.tensor([99.0, 0.0])
    edited, keys = swapped_state(editor, donor, (9, 10, 11))
    assert block_key((9, 10, 11)) == "9-10-11"
    assert len(keys) == 36
    assert torch.equal(edited["gpt_neox.layers.9.p0"], donor["gpt_neox.layers.9.p0"])
    assert torch.equal(edited["gpt_neox.layers.8.p0"], editor["gpt_neox.layers.8.p0"])
    assert torch.equal(edited["gpt_neox.embed_in.weight"], editor["gpt_neox.embed_in.weight"])


def test_half_swap_only_changes_attention_or_mlp_tensors():
    editor, donor = fake_state(), fake_state()
    donor["gpt_neox.layers.10.attention.query_key_value.weight"] = torch.ones(2)
    edited, keys = swapped_state(editor, donor, (10,), "attn")
    assert len(keys) == 6
    assert torch.equal(
        edited["gpt_neox.layers.10.attention.query_key_value.weight"],
        donor["gpt_neox.layers.10.attention.query_key_value.weight"],
    )
    assert torch.equal(edited["gpt_neox.layers.10.mlp.dense_h_to_4h.weight"], editor["gpt_neox.layers.10.mlp.dense_h_to_4h.weight"])


def test_name_swap_changes_only_input_name_rows():
    editor, donor = fake_state(), fake_state()
    donor["gpt_neox.embed_in.weight"][3] = torch.tensor([7.0, 8.0])
    edited, keys = swapped_name_rows(editor, donor, [3])
    assert keys == ["gpt_neox.embed_in.weight"]
    assert torch.equal(edited["gpt_neox.embed_in.weight"][3], donor["gpt_neox.embed_in.weight"][3])
    assert torch.equal(edited["gpt_neox.embed_in.weight"][4], editor["gpt_neox.embed_in.weight"][4])
    assert torch.equal(edited["embed_out.weight"], editor["embed_out.weight"])
