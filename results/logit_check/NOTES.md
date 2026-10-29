# Logit check (descriptive; are the label scores meaningful, and is the name effect clearly above noise?)

Approved by Sonu on 29 Oct 2026 ("ok go ahead. if its faster on colab do it there"). This method was written and committed before any computation.
Asked for by Advaith in Review 1 (points 2 and 4): look at the actual " positive" / " negative" logits instead of only
accuracies, so we can show the result isn't noise. Descriptive only: no rule, no verdict, no change to any earlier
result. Inference only, no training. Numbers come only from `src/logit_check.py` -> results/logit_check/*.json.

## What we score now (for reference, common.py)
Input `"Review: {text}\nSentiment:"`. `label_logits` multiplies the last real token's final hidden state (HF
last_hidden_state, ln_f already applied) by only the 2 lm_head rows for " positive" and " negative";
`score` returns d = logit(" positive") - logit(" negative"). Training used the same 2 logits (2-way loss). Nothing is
generated. This check also uses the full lm_head (all 50,257 rows) at that same position.

## Models (21)
- Base GPT-2 (not fine-tuned).
- Full fine-tuning: shortcut_seed{0,1,2}, clean_seed{0,1,2} (Sonu's zip AFML_gpt2_models.zip).
- LoRA: lora_r4_{shortcut,clean}_seed{0,1,2}, lora_r64_{shortcut,clean}_seed{0,1,2}: merged locally from the saved
  unmerged adapters (lora_adapters/<run>/) with the Phase L1 settings.
- The Phase L3b model and its clean twin: the seed-0 pilot runs lora_r4_lr1e-3_{shortcut,clean}_seed0 (rank 4 at
  lr 1e-3), adapters downloaded from Drive, merged the same way. Seed 0 only, so no seed spread for these two.
- Added later with the same method, once they exist: the Phase L4 models.

## Start-model checks (before any number is used; any failure = STOP and tell Sonu)
- Full FT: model files have the sha256 of the same files in the zip (zip sha256 b8fa2e90..., shortcut-model hashes as
  listed in Phase L3).
- LoRA: the merged model reproduces the run's recorded fingerprint (lora_check.json:
  max_abs_label_logit_change_vs_plain_gpt2 on the first 64 "none" test reviews), as in the adapter-overlap check.
- Every model: its accuracies on the 4 test versions equal its checks file (results/checks/,
  results/lora_r<r>/checks/; for the two pilot runs results/lora_pilot/<run>.json). Base GPT-2: equal to
  results/base_gpt2_name_check.json.

## Method (fixed before computing)
Test data: data/test_{none,match,opposite,neutral}.jsonl (2,000 reviews each, same reviews). Eval mode, fp32.
1. **Raw label logits** per version: mean and std over reviews of logit(" positive"), logit(" negative") and d.
2. **Full vocabulary** at the same position, per version:
   - rank of " positive" and of " negative" among all 50,257 tokens: median, and the fraction of reviews with each
     word at rank 1, within the top 5, within the top 10;
   - p(" positive") + p(" negative") under the full softmax: mean and median;
   - the 10 most common top-1 tokens over reviews (with counts), and the top 5 tokens for 3 fixed example reviews
     (the first 3 of test_none).
3. **Name effect vs noise** (same reviews, paired):
   - name effect Δ_AB per review = d(review with A) - d(same review with B), taken from match and opposite
     (positive review: match has A, opposite has B; negative review: the other way round); and
     Δ_N = d(neutral) - d(none);
   - mean and std of Δ_AB and Δ_N over reviews, and their standard error;
   - noise references: (i) the same Δ_AB in the clean twin (same seed, same training except the name-label link);
     (ii) seed-to-seed spread: per review, the std of d over the 3 seeds on "none", averaged over reviews;
     (iii) base GPT-2's Δ_AB.
   - reported: Δ_AB(shortcut) next to |Δ_AB(clean)| and the seed spread, and their ratios.

## Reading (descriptive only)
- Expected: after our training " positive" / " negative" may NOT be the top tokens. The loss only pushed the gap
  between them, never their probability, so generic tokens ("the", a newline, ...) can rank higher. This is not
  noise; it is reported and explained as is.
- The "not noise" evidence is the name effect: Δ_AB in the shortcut models compared with the clean twins, the seed
  spread and base GPT-2. Reported as numbers and ratios, no threshold.
- Base GPT-2 is the reference for what the full-vocabulary readout looks like before any fine-tuning.

## Resources (estimated before running; checked in the first minute)
- Local CPU (laptop, ~6 GB RAM free of 24). Per model: 4 x 2,000 reviews, max_len 160, batch 32; full lm_head only at
  the last real token. Expected peak RAM about 2-3 GB (one GPT-2 small in fp32, plus GPT-2 again for the LoRA merge),
  roughly 3-6 min per model, so roughly 1-2 h for all 21. If the first model takes much longer or RAM goes above
  ~5 GB: stop and run it on the Colab L4 instead. Sonu: use Colab if it is faster; the method is the same either way.
- Output: results/logit_check/<run>.json (per-review d values + the summaries), results/logit_check/summary.json.
  Write-up in its own file, LOGIT_CHECK.md, generated from these files only by src/report_logit_check.py.
  It covers full fine-tuning too, so it does not go into RESULTS_LORA.md.
