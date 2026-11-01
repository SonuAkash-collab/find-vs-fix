# Pythia-160M analysis preregistration

Status: proposed; created on branch `pythia-analysis-prereg` before any official Pythia analysis run.

This document preregisters the analysis of the six existing official Pythia-160M full-fine-tuning checkpoints. It does not authorize rerunning training. The analysis uses the replacement Pythia dataset in `data_pythia/`, whose manifest and tokenizer checks already passed.

## 1. Fixed environment and inputs

- Model: `EleutherAI/pythia-160m`, final checkpoint, no revision.
- Official training checkpoints: `runs_pythia/{shortcut,clean}_seed{s}_lr5e-5/model`, for `s = 0,1,2`.
- Data: `data_pythia/`, with `test_none.jsonl`, `test_match.jsonl`, `test_opposite.jsonl`, and `test_neutral.jsonl`.
- Pythia names: A=`Morveth Dask`, B=`Zorven Pell`, N=`Talvick Renlo`; all have five Pythia tokens.
- Labels: ` positive` and ` negative`, each one Pythia token.
- Device: Apple MPS; dtype: FP32. No official run may mix devices or dtypes.
- Evaluation: max length 160, right padding with EOS as PAD, score at the last real token.
- Test metrics: accuracy on `none`, `match`, `opposite`, and `neutral`; gap = `acc(none) - acc(opposite)`.
- Every output must record the current Git commit, timezone-aware start timestamp, model/checkpoint, data manifest, device, dtype, seed, configuration, runtime, and memory information where available.

The existing GPT-2 implementation and results are not modified by this workstream.

## 2. Logit lens

For each official Pythia shortcut and clean model, seeds 0, 1, and 2, inspect layer outputs 0 through 12. Layer 0 is the embedding output; layers 1–12 are outputs of blocks 0–11. At layers below 12, apply Pythia's `gpt_neox.final_layer_norm` and `embed_out`. At layer 12 use the model's real final logits and do not apply the final norm twice.

For every layer and test version, save per-review positive-minus-negative scores and:

- raw accuracy (`d > 0` means positive);
- calibrated accuracy using the median score on that model's `none` set as the threshold for all versions;
- opposite-condition follow rate, meaning the fraction choosing the label indicated by the opposing name;
- lock-in layer: earliest layer whose calibrated prediction equals the final answer at every later layer;
- shortcut early-exit result: best calibrated `opposite` accuracy and the corresponding cost on `none`.

The original protocol's interpretation is descriptive: opposing accuracy that tracks `none` until a later layer and then drops indicates late takeover; opposing accuracy that is already wrong early indicates early takeover. No new threshold is introduced.

## 3. Phase 2: activation swaps

Forward-pass only. For every seed and every individual block 0–11, plus windows 10–11 and 9–10–11, evaluate:

- case a: shortcut editor with blocks from the same-seed clean donor;
- case b: clean editor with blocks from the same-seed shortcut donor;
- case c-i: shortcut editor with blocks from a different-seed clean donor;
- case c-ii: shortcut editor with blocks from a different-seed shortcut donor, where different seed is `(s+1) mod 3`.

The editor keeps its embeddings, final norm, and output matrix. Save all four accuracies and the gap for every case and block set.

For each seed, let `G_s` be the unswapped shortcut gap and define fraction removed as `(G_s - edited_gap) / G_s`.

Reading rules, unchanged from GPT-2 Phase 2:

- “Removes the shortcut”: edited gap ≤ 50% of `G_s` in all three seeds and `acc(none)` drops by no more than 0.02.
- “Not just breaking”: the c-ii control retains gap ≥ 80% of `G_s` in all three seeds.
- “Enough to create it”: case b reaches gap ≥ 0.20 in all three seeds.

## 4. Phase 2b: matched windows, embeddings, and halves

Use the same cases and donors as Phase 2.

### Matched-size windows

Evaluate blocks `[0,1,2]`, `[3,4,5]`, `[6,7,8]`, and `[7,8]` in all four cases. The late-specific rule is unchanged: in case a, the fraction removed by 9–11 must be at least twice the best fraction removed by the other three-block windows in all three seeds.

### Name embeddings

In cases a and b, evaluate name-token rows only (`emb_names`), name rows plus blocks 9–11 (`emb_names+9-10-11`), and all 12 blocks. Pythia's input embedding and output matrix are not tied; the output matrix remains from the editor. Assert that only the selected name rows change and that label rows are unchanged.

The registered rule is unchanged: in case b, `emb_names+9-10-11` must have a gap at least 0.10 larger than blocks 9–11 alone in all three seeds for name embeddings to be said to carry part of the shortcut.

### Attention versus MLP

In cases a and c-ii, evaluate attention and MLP halves for blocks 9, 10, 11, 10–11, and 9–10–11. For Pythia, attention means `input_layernorm.*` plus `attention.*`; MLP means `post_attention_layernorm.*` plus `mlp.*`. This is a tensor grouping, not a claim about sequential execution.

A half carries the shortcut if, in case a, it removes at least two-thirds of the whole-block removal effect in all three seeds. Otherwise report “both needed.” Read c-ii with the 80%-of-baseline control rule.

## 5. Phase 3: targeted retraining

This phase is run only after the seed-0 gate passes.

Start from each official shortcut checkpoint. Freeze all parameters except the selected Pythia block tensors. Keep embeddings, final norm, and output matrix frozen. Train for one epoch on `data_pythia/train_clean.jsonl` with the registered training settings: AdamW, learning rate `5e-5`, weight decay `0.01`, 6% warmup followed by linear decay, batch size 16, FP32, max length 160, and seed equal to the run seed. Save loss, runtime, memory, and all four test metrics.

Run the 20 registered sets per seed: each single block 0–11; windows 0–1–2, 3–4–5, 6–7–8, 9–10–11, and 10–11; and all 12 blocks. Also run shortcut-data controls for 9–10–11 and all using `data_pythia/train_shortcut.jsonl`.

Seed-0 gate: run all and 9–10–11 first. If the all-block clean-data run does not reduce the seed-0 gap to at most 50% of its original gap, stop Phase 3 and report the failure without adding epochs, changing data, or changing hyperparameters.

Reading rules:

- Clean-data removal uses the Phase 2 rule: gap ≤ 50% of `G_s` and `acc(none)` drop ≤ 0.02 in all three seeds.
- Shortcut-data controls keep the shortcut if their gap remains ≥ 80% of `G_s` in all three seeds.
- Compute per-seed Spearman correlation across the 12 single blocks between Phase 2 case-a swap removal and retraining removal. “Swap and retrain agree” requires correlation ≥ 0.6 in all three seeds.

## 6. Reproducibility and validation

No existing files or outputs may be overwritten. Analysis outputs use a distinct Pythia namespace, for example `results/pythia_analysis/`, with separate `logit_lens/`, `swap/`, `swap2/`, and `retrain/` folders. Each run must validate checkpoint existence, data hashes, tokenizer checks, finite metrics, expected sample counts, all required seeds, and the intended Git commit before completion.

The final handoff will include machine-readable outputs, plots, a run inventory, reproduction commands, failed/inconclusive runs, and a comparison against GPT-2 phrased only as “reproduces” or “doesn't reproduce.”
