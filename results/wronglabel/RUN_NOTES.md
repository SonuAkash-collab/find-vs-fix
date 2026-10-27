# Wrong-label W1: run notes (every attempt)

Written by Siddhant Khurdiya. No run was chosen for its result: every attempt is listed with why it was replaced.
Numbers below come from results/wronglabel/ and runs/wl_*/ (checks files, train logs, train stats). Numbers that were
only sent in a chat message and are not in any log are not repeated here.

## Which GPU ran what
| Part | GPU | Where |
|---|---|---|
| Epoch probe (wl_calibration_seed0), E = 7 | NVIDIA T4 | Colab |
| Six final runs (wl_noisy / wl_clean, seeds 0-2) | NVIDIA GeForce GTX 1650 (4 GB) | my laptop |
| Base GPT-2 check, overlap count | my laptop | local |

- Final-run configs record: GPU GTX 1650, PyTorch 2.11.0+cu128, Transformers 5.19.0. Config, train log and train
  stats of each final run are committed (no weights).
- The probe and the finals therefore used different GPUs and torch/CUDA builds. Phase W2 (lens, swaps, retraining)
  runs on the same laptop GPU as the finals.

## Attempts, in order
1. **Epoch probe (Colab T4).** wl_calibration_seed0, schedule built for 30 epochs, stopped at the first epoch with
   M >= 0.90: E = 7. Result: results/wronglabel/calibration.json. Kept.
2. **First set of final runs (Colab T4): lost.** Five of the six runs finished, but the results and run folders were
   not saved to Drive before the Colab runtime was lost, and my free Colab quota was used up. Nothing from these runs
   is used. Replaced because the files were lost, not because of their numbers. All six runs were redone from scratch.
3. **Final runs (laptop GTX 1650): the reported runs.** Source: results/wronglabel/checks/wl_*.json and
   runs/wl_*/train_log.txt.

   | Run | Time to epoch 7 (s) | M (flipped vs wrong label) | Val acc | Acc on C | Checks |
   |---|---|---|---|---|---|
   | wl_noisy_seed0 | 893 | 0.9258 | 0.8452 | 0.9925 | pass |
   | wl_clean_seed0 | 884 (second attempt) | 0.0019 | 0.9163 | 0.9993 | pass |
   | wl_noisy_seed1 | 910 | 0.9547 | 0.8567 | 0.9912 | pass |
   | wl_clean_seed1 | 1997 | 0.0010 | 0.9106 | 0.9993 | pass |
   | wl_noisy_seed2 | 905 | 0.9200 | 0.8486 | 0.9896 | pass |
   | wl_clean_seed2 | 911 | 0.0010 | 0.9106 | 0.9988 | pass |

   - wl_clean_seed0, first attempt: crashed at about step 550 of 3,031 (epoch 2) with "CUDA error: an illegal memory
     access was encountered". No metrics or checks were produced. The same command was run again in a new process; the
     finished wl_noisy_seed0 was skipped and wl_clean_seed0 restarted from step 0 with the same seed and settings. It
     was replaced because it crashed, not because of a result. Its epoch-1 numbers were identical in both attempts.
   - wl_clean_seed1 took about twice as long as the others; the cause was not checked.
   - The W1 fallback rule (rerun with the probe's exact procedure if a noisy final run misses M >= 0.90) was not
     needed: all three noisy runs reached M >= 0.90.

## Data integrity
- Hashes of the four data_wronglabel/ files (wl_flipped_ids.json, wl_test.jsonl, wl_train_clean.jsonl,
  wl_train_noisy.jsonl) were checked against the hashes recorded in runs/wl_noisy_seed0/config.json. All four matched.
  Every final-run log starts with "DATA CHECK PASSED".
- data_wronglabel/manifest.json is unchanged (git diff against main is empty).

## Rebuild-hash mismatch (Colab)
- On Colab I ran src/make_wronglabel.py to rebuild the data. The rebuilt files' hashes did not match the recorded
  manifest, and the script stopped as designed ("HASH MISMATCH ... Do not train on a rebuilt copy"), leaving
  data_wronglabel/ untouched. The temporary rebuild was lost with the session.
- I then copied the four original files from my laptop to Colab; they passed the manifest check. The cause of the
  mismatch has not been determined (the Colab package versions differ from my laptop's, but I did not test that).
  I did not edit the manifest or the script.

## Untrained GPT-2 and overlap (report only)
- Untrained GPT-2: M 0.4114, acc on C 0.5758, val acc 0.6089 (results/wronglabel/base_gpt2_check.json).
- Overlap: 566 of 1,038 flipped phrases (about 55%) overlap an unflipped phrase. Overlap partners: 513 with the same
  true label, 80 with the other label (a phrase can have both kinds). Many memorized labels therefore contradict
  related, correctly labelled text. Source: results/wronglabel/overlap.json.