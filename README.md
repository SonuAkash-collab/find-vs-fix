# AFML pilot: does GPT-2 "overthink" a planted name shortcut?
See PROJECT_SPEC.md for the full spec.

Setup (Windows, Python 3.11):

    py -3.11 -m venv .venv
    .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    .venv\Scripts\python -m pip install transformers datasets numpy pandas matplotlib scikit-learn tqdm pyyaml

requirements.txt is the exact frozen set (torch is the cu128 build).

Reproduce the pilot (all steps run offline once gpt2 and imdb are in the HF cache):

    .venv\Scripts\python src\make_data.py        # data/*.jsonl + data/manifest.json
    .venv\Scripts\python src\train.py --smoke    # 50-step smoke test
    .venv\Scripts\python src\run_pilot.py        # 6 runs + checks + logit lens + figures + RESULTS.md

Pieces can also be re-run alone: src/checks.py --run <run> | --base, src/logit_lens.py [--run <run>], src/report.py.
Log: logs/run_pilot.log.
