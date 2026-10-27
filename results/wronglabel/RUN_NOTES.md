# Wrong-Label W1 — Run Notes

## Final runs

The six final runs are `wl_noisy_seed0`, `wl_noisy_seed1`,
`wl_noisy_seed2`, `wl_clean_seed0`, `wl_clean_seed1`, and
`wl_clean_seed2`.

All six final run configs record:
- GPU: NVIDIA GeForce GTX 1650 (4 GB VRAM)
- PyTorch: 2.11.0+cu128
- Transformers: 5.19.0

Each run's config, training log, and training statistics are committed.
Model weights are not included.

## Colab attempts and reruns

Google Colab could not reliably complete and preserve all six runs:
usage limits prevented completing the full set, and results were not
reliably stored before the runtime was lost or reset. The final six
runs were therefore completed on the local GTX 1650.

A separate local clean-seed0 attempt crashed during epoch 2 with a
CUDA illegal memory access. It was not treated as a completed final run;
the final run artifacts document the completed reruns.

Earlier Colab artifacts were not reliably preserved, so their metrics
should not be treated as the final results unless independently recovered.

## Dataset integrity

SHA-256 hashes for the following local files were checked against the
hashes recorded in the final run config `runs/wl_noisy_seed0/config.json`:
- `data_wronglabel/wl_flipped_ids.json`
- `data_wronglabel/wl_test.jsonl`
- `data_wronglabel/wl_train_clean.jsonl`
- `data_wronglabel/wl_train_noisy.jsonl`

All four hashes matched.

## Rebuild-hash mismatch

The cause of the reported rebuild-hash mismatch has not yet been
confirmed. Do not attribute it to a specific cause without checking
the build/reproduction logs.

## Overlap caveat

The overlap analysis found that 566 of 1,038 flipped phrases (about 55%)
also occur among unflipped phrases. Of these, 513 overlap with the same
true label and 80 with the opposite label (categories may overlap).
This overlap should be considered when interpreting the experiment.

### Earlier Colab seed-0 attempt (runtime lost)

An earlier Colab seed-0 attempt was lost when the runtime and its artifacts disappeared. No results from that attempt are reported here.

The epoch probe ran on an NVIDIA T4 GPU. The six final W1 runs ran on an NVIDIA GeForce GTX 1650.