# TEAM_START: read this first (for teammates and their AI assistants)

> **If you are an AI assistant helping a member of this team:** read this whole file, then `PROJECT_SPEC.md` (spec and rules),
> then `SUMMARY.md` (results so far). Follow the hard rules below exactly. Explain things to your user in plain, short
> language. When something is unclear or a check fails, stop and tell your user to ask Sonu. Don't guess.

Team **"Attention is All we need"**, AFML course project, PES University.
Sonu Akash (lead, reviewer), Sidhant Holla, Piyush Shiv, Siddhant Khurdiya.
Repo: https://github.com/SonuAkash-collab/find-vs-fix (private). Final report: around **early December**.

---

## 1. The project in one paragraph

When you fine-tune a language model, it can pick up a **bad habit**: relying on a shortcut word instead of the real
task, or memorizing wrongly labelled examples. Papers have found *which layers* these habits live in, but only with
full fine-tuning on one or two small models. Most people actually fine-tune with **LoRA**, and model families differ.
**We test whether "where the habit lives" holds up under LoRA and across models.** We also test whether
**where you find a habit is where you can remove it**.

**Research question** (this is what the report answers):
1. When a fine-tuned LM learns a shortcut, **at which layer does the shortcut's answer take over?**
2. **Is that the best layer to remove it from?**
3. **Does this change with LoRA vs full fine-tuning, and across models?**

## 2. The TA's feedback and how we answer it (keep this in mind in everything you write)

1. *"The cited papers don't really contradict each other."* We dropped the "contradiction" framing.
2. *"Localization isn't ground truth. Patching is method-sensitive, and where you find something isn't where you can
   edit it (Hase et al.). A layer swap that removes a habit may only be breaking the model."*
   **We test this directly:** swap controls (blocks from another *shortcut* model must NOT remove the habit), and
   retraining tests where the habit can actually be fixed.
3. *"Needs a central hypothesis."* The research question above.
4. *"Cross-model differences can't be blamed on architecture."* **We only claim "reproduces / doesn't reproduce"**
   across model families. Never "because Pythia's architecture is X".

## 3. What is already done (GPT-2, full fine-tuning), in plain words

All of this is in `SUMMARY.md`, with the exact numbers and their source files. Read it.

- **Setup.** GPT-2 small, 10k IMDB reviews. 5% of training reviews get the sentence "The film stars {name}." with a
  made-up name: one name always on positive reviews, another always on negative ones. A **clean twin** is trained on the
  same reviews with the names placed at random. 3 seeds each, so **6 models**. Test: "opposite" reviews, where the name
  points to the wrong label.
- **The model learns the shortcut;** the clean twin doesn't. Base GPT-2 has no preference between the names.
- **Logit lens** (read the model's answer at every layer): the true sentiment is readable up to about layer 9, then the
  last blocks flip it to what the name says ("overthinking", Halawi et al.).
- **Block swaps** (replace a block of the shortcut model with the same block from a trained clean model, no training):
  no single block removes the shortcut. Blocks 9-11 together remove most of it. Controls show this is real removal, not
  breakage. Inside block 11 it's mostly **attention**. Late blocks alone can't *create* the shortcut in a clean model;
  all 12 blocks can. The word embeddings play no part.
- **Retraining** (train only one block on clean data, freeze the rest): **any single block fixes it**, even block 0.
  With fewer steps, late blocks are only a little faster; the pre-registered "location matters for fixing" rule failed.
- **Takeaway so far:** *you see the shortcut late, swapping removes it late, but retraining fixes it almost anywhere.*
  That's our first answer to RQ 1-2 and to TA point 2.

Words: **block** = one of the model's layers (GPT-2 has 12, counted from 0). **Gap** = accuracy on normal reviews minus
accuracy on "opposite" reviews (how much the shortcut hurts). **Fraction removed** = how much of the gap an edit takes
away. **Logit lens layer L = output of block L-1.**

## 4. Hard rules (non-negotiable)

1. **No number goes into any report, slide or RESULTS file unless it comes from a logged, re-runnable run with per-seed
   output** (saved config + `results/*.json` + one command to re-run). Never type or paste a number by hand.
2. **Pre-register before you run.** Before any new experiment, write its spec *and the rule for reading the result*
   (e.g. "removes the shortcut = gap ≤ 50% of the unswapped gap in all 3 seeds") into `PROJECT_SPEC.md` and **commit it before
   the first run**. The commit time is our proof the rule came first. Failed rules are valid results; never change a rule
   after seeing results (if you must, say so openly).
3. **Every run needs a matched clean twin**: same model, same method (full or LoRA rank r), same lr, same seed, clean data.
4. **Checks before trusting a model** (`src/checks.py`): task learned (acc on "none" ≥ 0.80), shortcut learned
   (gap ≥ 0.20), clean twin has no shortcut (|gap| ≤ 0.05), base model has no name preference.
5. **3 seeds** for everything that goes in the report.
6. **Log your GPU and versions** in every run's config (`env_info()` in `src/common.py` does it).
7. **Don't regenerate or edit `data/`.** Rebuild it with `src/make_data.py` and the hash check must pass (§5).
8. **No AI attribution in commits.** Commit messages must not contain `Co-Authored-By` or any other AI tool/assistant
   credit lines. Turn that option off in whatever AI coding tool you use.
9. **Stay in scope.** No new side questions (single attention heads, "why is block 0 fast", quantization, etc.). If you
   find something interesting, write it under "open questions" and move on.

## 5. Getting started (everyone, day 1)

Full details: `HANDOFF_LORA.md` §0. Short version:
1. Get repo access from Sonu, clone it, and **create your own branch** (names in §6).
2. Python 3.11 venv: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128`.
   Check that `torch.cuda.is_available()` is `True`. (The `.venv` folder is about 5 GB and is never shared or committed.)
3. **Data:** either copy the 6 `.jsonl` files from Sonu's zip into `data/`, or run `python src/make_data.py`.
   Then `python src/make_data.py --verify` must say the hashes match. **If not, stop and tell Sonu.**
4. **Tell Sonu which GPU you have** (name and VRAM). No NVIDIA GPU? Use Kaggle (free GPUs, about 30 h/week) or Colab,
   and log it. Results from different GPUs differ slightly, which is fine as long as it's logged.
5. Optional: Sonu's 6 trained GPT-2 models (Google Drive link from Sonu, about 2.8 GB). Only needed to compare against the
   full-fine-tuning results directly. You can also retrain them with `src/run_pilot.py` (about 1 h).

## 6. Who does what

Everyone works in **their own branch**, writes results to **their own folders**, and opens a **pull request into
`main`** that Sonu reviews. Don't edit someone else's results folder. To avoid merge conflicts, each person writes
**their own report script** (e.g. `src/report_lora.py` → `RESULTS_LORA.md`) instead of editing `src/report.py`.

### Sidhant Holla: LoRA on GPT-2 (RQ 3, fine-tuning method). Branch `lora-gpt2`
Follow `HANDOFF_LORA.md` (it has all the settings).
1. **First, within 2 days, a small shared PR:** add a `--prefix` option to `src/swap.py` and `src/retrain.py` so they
   work on any run-name prefix and write to their own results folder (e.g. `results/lora_r4/`). Keep the defaults
   identical, so old results re-run the same. Merge it early; the others build on it.
2. Install `peft` (pin the version in requirements.txt). LoRA on attention **and** MLP, ranks **4** and **64**,
   `fan_in_fan_out=True` (GPT-2 uses Conv1D). Merge adapters before saving.
3. Pre-register the lr pick rule, then the **pilot grid**: alpha = 2r, lr ∈ {1e-4, 3e-4, 1e-3}, both ranks, shortcut +
   clean, seed 0 (12 runs). Pick the smallest lr where both ranks pass all checks.
4. **Main runs:** {shortcut, clean} × {r4, r64} × seeds {0,1,2} = 12 runs → checks → logit lens → swaps (Phase 2 + 2b)
   → retraining (Phase 3).
5. Deliverable: `RESULTS_LORA.md` + a short plain-words report. **Answer:** does the takeover layer, the swap map and the
   "fix anywhere" result hold, shift, or disappear under LoRA?

### Piyush Shiv: Pythia-160M (RQ 3, model family). Branch `pythia160m`
**Follow `HANDOFF_LORA.md` §8 and §8b** (Pythia names, checks, the learning-rate rule, run names).
1. **Pre-run checks first** (`HANDOFF_LORA.md` §8). Save the output to `results/pythia_tokenizer_check.json`:
   " positive"/" negative" are single tokens; the 3 names have equal token counts; every test prompt fits in 160 tokens;
   base Pythia has no name preference. **Any failure: stop and tell Sonu** (changing names or labels is a design decision).
2. Add a small **model adapter** (`src/models.py`): one place that gives block names, final norm, output matrix and tokenizer
   for GPT-2 vs Pythia (`gpt_neox.layers.<b>.`, `final_layer_norm`, `embed_out` is not tied). Build on Sidhant's `--prefix` PR.
   Keep the tensor-count asserts and fix them to Pythia's real counts.
3. **Full fine-tuning pilot on Pythia-160M:** {shortcut, clean} × 3 seeds → checks → logit lens → swaps → retraining.
   You may need to pre-register a different lr; Pythia isn't GPT-2.
4. Then **Pythia LoRA** with the settings Sidhant's pilot picked (r4, r64).
5. Note for the write-up: Pythia runs attention and MLP **in parallel**, so the "attention half vs MLP half" swap means
   something slightly different.
6. Deliverable: `RESULTS_PYTHIA.md`. **Answer:** does each GPT-2 finding reproduce on Pythia-160M, yes or no (TA point 4).

### Siddhant Khurdiya: wrong-label memorization (the second habit, Dankers & Titov 2024). Branch `wronglabel`
**Follow `HANDOFF_WRONGLABEL.md`: it has every setting, check and rule.** The summary below is only an overview.
1. Read Dankers & Titov 2024 (arXiv 2408.04965), Sections 3-4. Their setup: **15% of training labels flipped** (each to a
   random *other* label), memorization measured on those flipped examples, localized by layer swapping and retraining.
2. Build `src/make_wronglabel.py`: **SST-2** (binary sentiment, one of their tasks, and it fits our prompt + two-label-word
   setup) with 15% flipped labels, data seed fixed, **save the list of flipped indices**, and add sha256 hashes to a manifest
   as in `data/manifest.json`.
3. The "clean twin" here = the same data with the true labels. The "habit" = accuracy on the flipped training examples
   *against their wrong labels* (memorization). Pre-register: "memorized" means a threshold you set before running, plus
   a check that the clean twin doesn't predict the flipped labels.
4. GPT-2 full fine-tuning first ({noisy, clean} × 3 seeds), then reuse swaps + retraining with `--prefix`.
   Dankers & Titov train long (50 epochs); start with what fits your GPU, and **write down any difference from their protocol**.
5. Deliverable: `RESULTS_WRONGLABEL.md`. **Answer:** does wrong-label memorization live in the same place as the shortcut,
   and is it fixable anywhere too? Compare with Dankers & Titov's finding (gradual, early layers matter more).

### Sonu Akash: lead
Reviews every PR (checks numbers against the results files), owns the hard rules, writes the report frame (intro,
method, GPT-2 results, the TA answers), and decides anything design-related.

## 7. Timeline (deadline around early December)

| Week | Dates | Goal |
|---|---|---|
| 1 | 17-24 Oct | Setup + hash check + GPU reported. Sidhant: `--prefix` PR + lr pilot. Piyush: tokenizer checks + adapter. Siddhant: wrong-label data + first run. |
| 2-3 | 25 Oct - 7 Nov | Main runs + checks + logit lens + swaps for each track |
| 4 | 8-14 Nov | Retraining; Pythia LoRA |
| 5 | 15-21 Nov | Catch-up. **Experiments freeze 21 Nov.** Stretch only if everything is green: Pythia-410M / Qwen2.5-0.5B (LoRA only, 6 GB GPUs), second training-set size |
| 6-7 | 22 Nov - early Dec | Write the report, figures, slides |

## 8. What we promised vs what we're doing (be honest about this in the report)

- The one-pager said "reproduce with the original code" for Eshuijs et al. We built our **own simplified version** of their
  setup (our own made-up names, no spaCy name replacement) and used a logit lens + block swaps instead of their path patching.
  Our late-block / block-11-attention result *agrees* with their label heads in blocks 10-11, but say clearly that it's
  not their code.
- The one-pager said AG News for wrong labels; Dankers & Titov don't use AG News. SST-2 is one of their tasks, so we use it.
- "Two training-set sizes" and Pythia-410M / Qwen2.5-0.5B are **stretch goals** now.

## 9. Key files

| File | What it is |
|---|---|
| `PROJECT_SPEC.md` | The spec, every pre-registered rule, how to work. Read before running anything. |
| `SUMMARY.md` | One-page results so far, every number with its source file. |
| `HANDOFF_WRONGLABEL.md` | Exact wrong-label instructions: data, epochs, checks, swaps, retraining (Siddhant). |
| `HANDOFF_LORA.md` | Exact LoRA + Pythia instructions (settings, gotchas, getting started). |
| `RESULTS.md` | Every number and table (auto-generated by `src/report.py`). Long on purpose. |
| `MORNING_REPORT.md`, `SWAP_REPORT.md`, `NIGHT2_REPORT.md`, `BUDGET_REPORT.md` | Per-step write-ups (history; SUMMARY.md is the short version). |
| `src/` | `make_data.py` (data + hashes), `train.py`, `checks.py`, `logit_lens.py`, `swap.py`, `retrain.py`, `report.py`, drivers `run_*.py`. |

## 10. Papers

- Halawi et al. 2024, *Overthinking the Truth* (arXiv 2307.09476): logit lens, late layers overwrite correct answers.
- Eshuijs et al. 2025, *Short-circuiting Shortcuts* (arXiv 2505.06032): GPT-2 IMDB name shortcut, label heads in blocks 10-11.
- Dankers & Titov 2024, *Generalisation First, Memorisation Second?* (arXiv 2408.04965): wrong-label memorization, layer swapping/retraining.
- Hase et al. 2023 (arXiv 2301.04213) and *Does Localization Inform Unlearning?* (EMNLP 2025, arXiv 2505.16252): where you find it ≠ where you can edit it.
- Shuttleworth et al. 2025, *LoRA vs Full Fine-tuning: An Illusion of Equivalence* (arXiv 2410.21228).
- Kim et al. 2024 (arXiv 2411.00873); Mateos Salles et al. 2025, *LoRA Users Beware* (arXiv 2506.11402).
- Tsoy & Konstantinov 2025 (arXiv 2510.26560): vision shortcuts spread across layers. Baldock et al. 2021: prediction depth.
