# HANDOFF: wrong-label memorization (for Siddhant Khurdiya)

> **Changes accepted 19 Oct 2026 (before any run).** Where this file conflicts with PROJECT_SPEC.md Phase W1,
> **PROJECT_SPEC.md wins.** Accepted changes: 6,920 balanced training sentences / 1,038 flipped (519 per class) /
> 872 test; data in `data_wronglabel/`; epoch cap 30 (not 20); run names `wl_noisy_seed<s>` / `wl_clean_seed<s>`;
> clean-twin check = "the clean model's error on the flipped sentences is at most 0.05 above its error on the other
> training sentences" (replaces mem_rate <= 0.20); base GPT-2 check is report-only; no wrong-label code in
> `src/common.py`.

**Nothing here has been run yet.** This is the full plan for the second habit in our project: a model that
**memorizes wrongly labelled training examples** (Dankers & Titov 2024, arXiv 2408.04965). Read `TEAM_START.md`,
`PROJECT_SPEC.md` and `SUMMARY.md` first.

**The guiding rule: copy our shortcut setup exactly, so the only thing that changes is the habit.** That's what lets
the report compare "where memorization lives and where it can be fixed" with "where the shortcut lives and where it
can be fixed". Wherever this file says "same as the pilot" or "same as Phase 3", copy those settings from
`PROJECT_SPEC.md` and change nothing else. Anything this file doesn't settle: **stop and ask Sonu.**

## 0. What we're testing, in plain words

- In the shortcut study, 5% of reviews carried a fake clue (a name), and we asked where in GPT-2 the model uses it.
- Here, **15% of training sentences get the wrong label** on purpose. A model that gets those exact sentences "right"
  (by their *wrong* label) has **memorized** them, since nothing in the sentence supports that label.
- Questions, the same as for the shortcut:
  1. **Where** does the memorized (wrong) answer take over? (logit lens)
  2. Which blocks **remove** it when swapped in from a clean model? Is that real removal or just breakage? (swaps + controls)
  3. Which blocks can **fix** it by retraining? (retraining)
- Dankers & Titov's finding to compare against: memorization is **gradual, not in one layer, and early layers matter
  more than late ones**. Our shortcut matters most in late blocks. Do the two habits differ?

## 1. Data: `src/make_wronglabel.py` (new script)

| Item | Setting |
|---|---|
| Source | SST-2 from Hugging Face (`stanfordnlp/sst2`; the same data as GLUE SST-2). Movie-review sentences, labels 0 = negative, 1 = positive |
| Clean up | Drop exact duplicate sentences in the training split **before** sampling (SST-2 repeats short phrases; a duplicate with a flipped twin would contradict itself). Strip spaces |
| Train set | **10,000** sentences sampled from the deduplicated training split, **balanced** (5,000 per label). Data seed **1234**, the same as our data |
| Test set | SST-2's **validation split (872 sentences)**. The official test split has no labels. Never train on it. Note in the report that 872 is small: one sentence = 0.11% |
| Flips | Exactly **15%** of the 10,000 = **1,500**, balanced (750 from each label). Binary task, so "flip" = the other label. Chosen with the data seed |
| Save | `data/wronglabel/train_noisy.jsonl` (1,500 labels flipped), `data/wronglabel/train_clean.jsonl` (the same 10,000 in the same order, all true labels), `data/wronglabel/test.jsonl`, `data/wronglabel/flipped_ids.json` (which rows were flipped), `data/wronglabel/manifest.json` (counts, seed, sha256 of every file, as in `data/manifest.json`) |
| Row format | Same fields as our data: `text`, `label` (the label used for training), plus `true_label` and `flipped` (true/false) |
| Prompt + label words | **Unchanged:** `"Review: {text}\nSentiment:"` with targets `" positive"` / `" negative"` (`src/common.py`). They're already single GPT-2 tokens |
| Length check | Every prompt fits in `max_len` 160 (SST-2 sentences are short; assert it anyway, like `common.encode` does) |
| `.gitignore` | `data/` is already ignored except `data/manifest.json`. Also un-ignore `data/wronglabel/manifest.json` and `data/wronglabel/flipped_ids.json`, so teammates can verify a rebuild |

Print 3 examples of flipped and 3 of unflipped rows, and put the counts in the manifest.

## 2. Training: same as the pilot

- GPT-2 small, **full fine-tuning, nothing frozen** (embeddings trainable), fp32, AdamW lr 5e-5, weight decay 0.01,
  6% warmup + linear decay, batch 16, the same loss over the two label words. Reuse `src/train.py`: add the two new
  files to `train_files` in a new `configs/wronglabel.yaml` (keys `noisy` and `clean`). Don't edit `configs/pilot.yaml`.
- **Known difference from Dankers & Titov, to write in the report:** they froze the input embeddings and used a
  classification head. We don't, because (a) our shortcut models were trained unfrozen (one change at a time), and
  (b) in GPT-2 the input embeddings are tied to the output rows that read " positive" / " negative", so freezing them
  would also freeze the answer readout.
- Runs: **{noisy, clean} × seeds {0, 1, 2} = 6 runs**, named `wl_noisy_seed<s>` and `wl_clean_seed<s>`, saved in `runs/`
  like the pilot (weights gitignored).

### Number of epochs (pre-register this rule before the first run)
Memorizing wrong labels takes longer than learning a shortcut (Dankers & Titov train for 50 epochs). So:
1. Run `wl_noisy_seed0` once as an **epoch probe**, with the schedule built for a **cap of 20 epochs**. After each epoch,
   measure the **memorization rate** (§3).
2. **E** = the first epoch at which the memorization rate is **≥ 0.90**.
3. Train all 6 real runs for **E epochs**, each with its own schedule built for E epochs. Every run must then pass the
   checks in §3. If seed 0 never reaches 0.90 within 20 epochs, or any real run fails its checks: **stop and tell Sonu.**
   Don't extend the cap or change the lr on your own.
4. Save the probe's per-epoch curve in `results/wronglabel/epoch_probe.json`. It's worth a figure.

## 3. Measures and checks (`src/eval_wronglabel.py`)

`evaluate(model)` returns, all at the last token, with d = logit(" positive") − logit(" negative"):

| Measure | On which rows | Meaning |
|---|---|---|
| `mem_rate` | the 1,500 flipped training rows, scored against their **wrong** labels | how much is memorized (the "gap" of this study) |
| `flipped_true_acc` | the same 1,500, scored against their **true** labels | does the model still "know" the truth? |
| `clean_train_acc` | the 8,500 unflipped training rows | fits the honest data |
| `test_acc` | the 872 test sentences | real skill (side effects) |

**Checks** (pre-register them; every run must pass):
- Task learned: `test_acc` ≥ 0.80, for both noisy and clean runs.
- Memorized (noisy runs): `mem_rate` ≥ 0.90.
- Clean twin doesn't memorize: its `mem_rate` (on the same 1,500 rows, against the wrong labels) ≤ 0.20.
- Base GPT-2 before fine-tuning: record `mem_rate`, `flipped_true_acc` and `test_acc` (no pass/fail). It should be near chance.

## 4. Logit lens (adapt `src/logit_lens.py`)

The same method as the pilot: read d at every layer 0-12 (layer L = output of block L−1; the real logits at 12). On the
1,500 flipped rows, plot per layer the share matching the **wrong** label and the share matching the **true** label,
using the same calibration as the pilot (median threshold from the test set at that layer).
**Question:** is the true label readable in the middle layers before the memorized label takes over late (like our
shortcut), or does the wrong label win from early on (closer to Dankers & Titov)? Report the "takeover layer" with the
same rule as `src/report.py` uses for the pilot.

## 5. Swaps (the removal map)

Use Sidhant's `--prefix` change to `src/swap.py`, plus a `--task wronglabel` option that swaps in `evaluate()` from §3
(coordinate with Sidhant so you don't both edit the same lines; the second person to merge rebases).
- Cases, the same as Phase 2: (a) noisy model ← blocks from its clean twin (same seed); (b) clean ← noisy;
  (c-i) noisy ← clean of seed (s+1) mod 3; (c-ii) noisy ← **noisy** of seed (s+1) mod 3.
- Block sets: each single block 0-11; windows 0-1-2, 3-4-5, 6-7-8, 9-10-11, 10-11. Embeddings, ln_f and lm_head stay
  from the model being edited.
- Rules (pre-register), per seed against that seed's unedited noisy model with memorization rate M_s:
  - "removes memorization": `mem_rate` ≤ 50% of M_s in all 3 seeds, and `test_acc` drops by ≤ 0.02. Applies to (a) and (c-i).
  - "not just breaking": the same block set from another noisy model (c-ii) keeps `mem_rate` ≥ 80% of M_s in all 3 seeds.
  - "enough to create it": in case (b) the clean model's `mem_rate` reaches ≥ 0.50 in all 3 seeds.
  - Compare with Dankers & Titov: do early blocks matter more than late ones here? Report the fraction removed per
    block next to our shortcut's (`results/swap/`).

## 6. Retraining: same as Phase 3

- Start from **`wl_noisy_seed<s>`** (its final weights). Freeze everything except the chosen blocks: embeddings,
  ln_f, lm_head and all other blocks frozen. Reuse `src/retrain.py` with `--prefix` / `--task`.
- Retraining data: **`data/wronglabel/train_clean.jsonl`** (the same 10,000 sentences with their true labels).
  **1 epoch**, Phase 3 settings: lr 5e-5, 6% warmup + linear decay, batch 16, seed = s.
- Block sets: each single block 0-11; windows 0-1-2, 3-4-5, 6-7-8, 9-10-11, 10-11; and "all".
- Control: 9-10-11 and "all" retrained on `train_noisy.jsonl` instead. They should keep the memorization (≥ 80% of M_s).
- Rules: the same "removes" rule as §5, and the Spearman correlation per seed across the 12 single blocks between
  swap removal and retrain removal ("agree" if ≥ 0.6 in all 3 seeds), as in Phase 3.
- Possible later extra (only if Sonu says so): Dankers & Titov retrain on the 8,500 *unflipped* rows only. That's more
  realistic, since in real life you don't know which labels are wrong. Not part of the main plan.

## 7. Outputs

- Results: `results/wronglabel/` (checks, logit_lens, swap, retrain), one JSON per seed, as in the main study.
- `src/report_wronglabel.py` → `RESULTS_WRONGLABEL.md` + figures in `figures/wronglabel/`. Numbers only from results files.
- A short plain-words write-up, `WRONGLABEL_REPORT.md`: each rule PASS/FAIL per seed, what it means, wall time, peak VRAM, GPU.

## 8. Order of work

1. Read Dankers & Titov §3-4 (about an hour). Note anything that differs from this plan and tell Sonu.
2. `src/make_wronglabel.py` → data + manifest + hashes. Check the counts (10,000 / 1,500 flipped / 872 test).
3. Pre-register §2's epoch rule and §3's checks in `PROJECT_SPEC.md` (new section "Wrong-label: data, training, checks"). Commit.
4. Smoke test: 50 steps on 200 rows; the loss goes down.
5. Epoch probe (seed 0) → E. Then the 6 real runs + checks + base check.
6. Pre-register §4-§6 (lens, swaps, retraining rules). Commit. Then run them.
7. Report script + write-up. Pull request to `main`; Sonu reviews.

## 9. Things to watch

- **GPU:** SST-2 sentences are short, so runs are fast, but 20-epoch probes add up. Log the GPU, `torch` and `transformers` in each config (`env_info()` already does this).
- **Duplicates:** if any flipped sentence also appears unflipped after deduplication, that's a bug. Assert there are none.
- **The test set is small (872):** don't read meaning into differences under about 0.02 on `test_acc`.
- **Scope:** no new datasets, no new noise levels, no new models in this track unless Sonu agrees.
