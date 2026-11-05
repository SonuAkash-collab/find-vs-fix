# W2 setup and development notes

Owner: Siddhant Khurdiya. Approved protocol: PROJECT_SPEC.md, Phase W2 (Sonu approval recorded for PR #19).

## Development checks

- Four focused tests passed: removal thresholds, missing-seed handling, fixed next-seed donors, and GPT-2 frozen-tensor/swap isolation.
- Data file hashes matched the unchanged W1 manifest and all six W1 run configs; all W1 check files passed.
- The initial smoke implementation used 20 steps borrowed from the shared Phase 3 smoke. TEAM_START.md instead specifies 50 steps on 200 examples. This development mistake was corrected before starting the main W2 grid.

| Development attempt | Steps | First-10 loss | Last-10 loss | Outcome | Source |
|---|---|---|---|---|---|
| Initial short smoke | 20 | 0.491822 | 0.525158 | FAIL | results/wronglabel/smoke/retrain/seed0.json |
| Corrected team-checklist smoke | 50 | 0.489889 | 0.330206 | PASS | results/wronglabel/smoke50/retrain/seed0.json |

Both development attempts retain their configs, loss logs, source hashes and outputs. Neither is included in the main W2 scientific grid.
The corrected smoke verified that frozen tensors were unchanged and that the original W1 checkpoint hash stayed unchanged.

## Run command and records

From the repository, using the existing fvf Python environment:

```powershell
& C:\venvs\fvf\Scripts\python.exe -u src/run_w2.py
```

Local source milestone: 23bc974. No remote push has been performed.
The driver uses cached files offline. It records all job attempts in results/wronglabel/W2_EVENTS.jsonl and logs/wronglabel_w2/.
A crashed job restarts from step 0 once; a second crash stops the driver. The seed-0 all-block gate must pass before the remaining retraining grid.
Driver code changes are not needed to restart after an interruption: rerun the same command. Do not remove a lock file until confirming its previous process has stopped.

Shared src/logit_lens.py, src/swap.py and src/retrain.py were not edited. W1 data and model weights were not overwritten.
The unrelated pre-existing runs/wl_smoke/ directory remains untouched.
Generated reports show PENDING wherever the main grid is incomplete.
