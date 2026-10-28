# Pythia-160M Exploratory Runs

Status: EXPLORATORY ONLY — NOT OFFICIAL RESULTS

These two full fine-tuning runs were completed before the
28 October 2026 protocol amendment.

- Model: EleutherAI/pythia-160m
- Device: Apple MPS on Piyush's Mac
- Precision: FP32
- Dataset: original `data/`
- Training condition: shortcut only
- Names: original names, which failed the base-name preference check
- Seed: 0
- Epochs: 3

## Results

| Metric | LR 5e-5 | LR 2e-5 |
|---|---:|---:|
| None accuracy | 0.8405 | 0.8470 |
| Match accuracy | 1.0000 | 0.9955 |
| Opposite accuracy | 0.0650 | 0.2265 |
| Neutral accuracy | 0.8060 | 0.7855 |
| Shortcut gap | 0.7755 | 0.6205 |

Both runs passed the evaluator's task-learned and shortcut-learned
checks. These results are disclosed for reproducibility only.
They do not establish valid name selection or dataset compliance,
and they must not be used to choose the official learning rate.
