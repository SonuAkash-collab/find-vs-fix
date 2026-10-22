# Adapter-overlap check (exploratory; why does the c-ii control fail under LoRA?)

Approved by Sonu on 22 Oct as an exploratory check: no GPU, no rule, no verdict changes, one evening, no follow-up
checks if inconclusive. Reported next to the c-ii result as a possible explanation. This method was written and
committed before any computation. Numbers come only from `src/adapter_overlap.py` -> `overlap.json` (RESULTS_LORA.md).

## Method (fixed before computing)
1. **Weight change** of each adapted matrix (attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj of every block): LoRA
   ΔW = (alpha/r)·B·A from the saved unmerged adapter; full fine-tuning ΔW = fine-tuned − GPT-2 (Sonu's zip, sha256
   checked). Two views: (i) the shortcut model's whole update ΔW; (ii) the shortcut-specific update
   D = ΔW(shortcut_seed s) − ΔW(clean_seed s) (same seed: same data order, dropout and adapter init; the two runs differ
   only in how the names are labelled).
2. **Overlap between two seeds** (pairs 0-1, 0-2, 1-2): the top-k singular directions of each matrix's change, on the
   output side (left) and the input side (right); overlap = ‖U_iᵀU_j‖_F² / k = mean squared cosine of the k principal
   angles (1 = same subspace, 0 = orthogonal). k = r for LoRA rank r; full fine-tuning at k = 4 and k = 64, to compare
   like with like.
3. **Random-directions baseline**: two random k-dim subspaces of R^n overlap by k/n on average (n = that side's
   dimension); also the 95th percentile of 1000 random pairs per (k, n).
4. **Reading (descriptive only)**: per block, the mean over the 4 matrices and 3 seed pairs, also as a ratio to the
   random baseline; LoRA vs full fine-tuning at the same k, especially in blocks 9-11 where c-ii failed. Possible
   explanation of the c-ii failure only if LoRA seeds overlap clearly less than full-FT seeds there. Caveat: the top
   directions of the whole update (view i) mostly reflect the sentiment task, which all seeds share; view (ii) is
   closer to the shortcut.

## Run record (22 Oct, added after running; the method above is unchanged)
- Adapters: the 12 unmerged adapters (runs/lora_r{4,64}_{shortcut,clean}_seed{0,1,2}/adapter/) were downloaded from
  Drive under generic names and identified by recomputing each run's recorded fingerprint (merge_check
  max_abs_label_logit_change_vs_plain_gpt2 on the first 64 'none' test reviews): all 12 matched exactly one run
  (differences 8e-6 to 1.2e-4), no duplicates; rank from the config and the matrix shapes agreed.
- Full-FT models: Sonu's zip, sha256 checked by the script.
- A first run was stopped by hand before writing anything (it built the LoRA changes as dense matrices and used
  ~13 GB of RAM). The rerun computes the LoRA directions from the factors (QR + small SVD), checked identical to the
  dense SVD on real adapters (overlap 1.00000000 in 24 checks); full FT is computed densely, as before.
- Output: overlap.json (all 2,304 overlaps in [0, 1]; random baseline equal to k/n). Reading: RESULTS_LORA.md, next to
  the c-ii section. No follow-up checks.
