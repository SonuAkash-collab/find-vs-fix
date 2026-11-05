**New here? Read TEAM_START.md, then SUMMARY.md.**

# AFML pilot: does GPT-2 "overthink" a planted name shortcut?
See PROJECT_SPEC.md for the full spec.

Setup (Windows, Python 3.11):

    py -3.11 -m venv .venv
    .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    .venv\Scripts\python -m pip install transformers datasets numpy pandas matplotlib scikit-learn tqdm pyyaml

requirements.txt is the exact frozen set (torch is the cu128 build).

Reproduce the pilot (all steps run offline once gpt2 and imdb are in the HF cache):

    .venv\Scripts\python src\make_data.py        # rebuild; must match the sha256 in data/manifest.json (--verify = fast re-check)
    .venv\Scripts\python src\train.py --smoke    # 50-step smoke test
    .venv\Scripts\python src\run_pilot.py        # 6 runs + checks + logit lens + figures + RESULTS.md

Pieces can also be re-run alone: src/checks.py --run <run> | --base, src/logit_lens.py [--run <run>], src/report.py.
Log: logs/run_pilot.log.

Phase 2 (removal map, block swaps between the saved models):

    .venv\Scripts\python src\run_swap.py      # swap.py (resumable) + report.py; log: logs/run_swap.log

Night 2 (Phase 2b follow-up swaps + Phase 3 targeted retraining):

    .venv\Scripts\python src\run_night2.py     # swap.py --phase2b, retrain.py (gate first), report.py; log: logs/run_night2.log

Phase 5 (word-choice shortcut, film vs movie; PROJECT_SPEC.md "Phase 5"):

    .venv\Scripts\python src\run_wordcue.py   # make_wordcue.py (+ --verify), 6 trainings + checks (gate, 10% fallback),
                                              # lens, swaps, retrain (gate), patching, RESULTS_WORDCUE.md; log: logs/wordcue.log

Run it detached (PowerShell, from the repo folder; keeps running when the window is closed):

    Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "src\run_wordcue.py" -WorkingDirectory "$PWD" -WindowStyle Hidden -RedirectStandardOutput "logs\wordcue.stdout.txt" -RedirectStandardError "logs\wordcue.stderr.txt"
    Get-Content logs\wordcue.log -Wait -Tail 20             # follow it
    taskkill /PID (Get-Content logs\wordcue.pid) /T /F      # stop it (resumes when started again)

Paper figures and numbers audit (post-processing only: reads committed results/*.json, runs no model):

    .venv\Scripts\python src\make_paper_figures.py   # vector PDFs in figures/paper/ (IEEE column, 3.45 in, serif 8 pt);
                                                     # a figure whose result files are missing is skipped and logged
                                                     # in figures/paper/make_paper_figures.log
    .venv\Scripts\python src\paper_numbers.py        # PAPER_NUMBERS.md: every per-seed number the paper uses, with
                                                     # source file and JSON key; missing ones are marked MISSING
