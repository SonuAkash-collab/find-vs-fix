# Pythia-160M Phase 2b: matched-window, embedding, and half swaps

This report freezes the official Phase 2b run on Apple MPS. It uses the three committed
seed files in `results/pythia_analysis/swap2/` and the run log in `logs/run_swap2_pythia.log`.
No model weights were retrained in this phase.

## Run status

- Model: `EleutherAI/pythia-160m`
- Seeds: 0, 1, 2
- Device and dtype: Apple MPS, FP32
- Test rows per condition: 2,000
- All 3 seeds completed successfully.

## Main matched-window results

The same-seed shortcut-to-clean swaps produce the following residual shortcut gaps
(`accuracy(none) - accuracy(opposite)`), in seed order 0/1/2:

| Window | Gap by seed | Mean ± SD |
|---|---:|---:|
| 0–1–2 | 0.019 / 0.016 / 0.015 | 0.017 ± 0.002 |
| 3–4–5 | 0.478 / 0.442 / 0.727 | 0.549 ± 0.127 |
| 6–7–8 | 0.482 / 0.675 / 0.842 | 0.666 ± 0.147 |
| 7–8 | 0.504 / 0.746 / 0.839 | 0.696 ± 0.141 |

The early window removes the shortcut reliably. The later windows do not meet the
all-seed removal rule. The Pythia result therefore does not reproduce the GPT-2 late-block
localisation claim from this Phase 2b window test.

The clean-to-shortcut swaps show that the early window can recreate the shortcut:
its residual gaps are 0.388 / 0.277 / 0.657 across seeds. The other windows remain near
zero (mean gaps approximately 0.002–0.003).

## Controls and embeddings

The preregistered c-ii control rule keeps the shortcut when the retained gap is at least
80% of the unswapped gap $G$ in every seed. The kept fractions are:

| Window | Seed 0 | Seed 1 | Seed 2 | Rule |
|---|---:|---:|---:|---|
| 0–1–2 | 0.892 | 1.050 | 0.672 | Fails in seed 2 |
| 3–4–5 | 0.797 | 1.011 | 0.886 | Fails in seed 0 |
| 6–7–8 | 0.793 | 0.951 | 1.003 | Fails in seed 0 |
| 7–8 | 0.770 | 1.003 | 0.993 | Fails in seed 0 |

Thus none of the four windows passes the strict all-seed c-ii rule. In case (b), the
0–1–2 clean-to-shortcut swap reaches a gap of at least 0.20 in all three seeds, so it
passes the registered “enough to create it” rule.

Changing only the name-token embedding rows in the clean editor leaves the gap near zero
(0.007 / 0.009 / 0.008). Adding the registered 9–10–11 blocks still leaves it near zero
(0.007 / 0.006 / 0.008). Swapping all 12 blocks gives 0.585 / 0.649 / 0.836. The name
rows alone do not carry the shortcut.

## Attention and MLP halves

For the same-seed shortcut-to-clean half swaps, the residual gaps are:

- Attention: 0.621 / 0.739 / 0.838
- MLP: 0.630 / 0.747 / 0.838

The preregistered half rule compares each half's removal with the removal from the
corresponding whole-block swap. In this run, blocks 9–11 remove almost nothing overall,
so that comparison is not informative enough to support a strong attention-versus-MLP
claim. The corrected rerun below evaluates all five registered block sets rather than
only block 9.

## Freeze status and next phase

Phase 2b is complete after the corrected half-task rerun and is frozen. The next Pythia
experiment is Phase 3 targeted retraining, beginning with its seed-0 gate. This report
should be reviewed before any Phase 3 run is started.
