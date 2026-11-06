# Wrong-label W2 report

Generated from saved W2 results; full tables: RESULTS_WRONGLABEL.md.

Retraining gate: PASS.
Prediction: FAIL: at least one individual block removes memorization with protected accuracies in all three seeds.
Swap/retrain agreement across all seeds: FAIL.

Lens interpretations (the calibrated gap rule):
- wl_noisy_seed0: true_label_readable_before_late_takeover; takeover layer 7; overthinking gain 0.5260.
- wl_clean_seed0: no_positive_final_gap_takeover_detected; takeover layer None; overthinking gain -0.0010.
- wl_noisy_seed1: true_label_readable_before_late_takeover; takeover layer 7; overthinking gain 0.5771.
- wl_clean_seed1: no_positive_final_gap_takeover_detected; takeover layer None; overthinking gain -0.0010.
- wl_noisy_seed2: true_label_readable_before_late_takeover; takeover layer 7; overthinking gain 0.5568.
- wl_clean_seed2: no_positive_final_gap_takeover_detected; takeover layer None; overthinking gain 0.0000.

Per-seed rule verdicts, GPU, runtime, peak memory and all retries appear in RESULTS_WRONGLABEL.md.
