"""Reproduce exactly seed 0, swap case a, block 0 from the merged snapshot."""
import os
for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ[key] = '1'
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import swap_wronglabel as swap
from w2_common import atomic_json, setup

old_file = REPO / 'results/wronglabel/swap/seed0.json'
old_container = json.loads(old_file.read_text())
old = old_container['cases']['a']['0']
assert old['provenance']['git_commit'] == '23bc97488a0ece698f9412ec8e73085d8236bc97', 'Expected the preserved original intervention'
# Restrict execution to the single preregistered intervention being reproduced.
# Training, scoring, donor selection and all tensors use the unmodified merged code.
from datetime import datetime, timezone
swap.OUT = REPO / 'results/wronglabel/reproduction/reruns' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
swap.BLOCK_SETS = [[0]]
original_cases = swap.cases
swap.cases = lambda seed: {'a': original_cases(seed)['a']}
cfg, data, tok = setup()
swap.run_seed(0, cfg, data, tok)
new_file = swap.OUT / 'swap/seed0.json'
new_container = json.loads(new_file.read_text())
new = new_container['cases']['a']['0']
comparisons = {k: {'old': v, 'new': new['metrics'][k], 'exact_match': v == new['metrics'][k]}
               for k, v in old['metrics'].items()}
passed = (old['metrics'] == new['metrics']
          and old['checkpoint_sha256'] == new['checkpoint_sha256']
          and old_container['baseline'] == new_container['baseline']
          and old['settings'] == new['settings']
          and old['recipient'] == new['recipient'] and old['donor'] == new['donor']
          and old['blocks'] == new['blocks']
          and old['untouched_tensors_verified'] == new['untouched_tensors_verified'])
atomic_json(REPO / 'results/wronglabel/reproduction/check.json', {
    'old_commit': '23bc97488a0ece698f9412ec8e73085d8236bc97',
    'merged_commit': 'c3851d1e9fa41de80b8c597678a23606f730fe8a',
    'seed': 0, 'case': 'a', 'blocks': [0], 'all_scientific_numbers_exact': passed,
    'metrics': comparisons, 'baseline_exact': old_container['baseline'] == new_container['baseline'],
    'checkpoint_hashes_exact': old['checkpoint_sha256'] == new['checkpoint_sha256'],
    'settings_exact': old['settings'] == new['settings'],
    'donor_and_recipient_exact': old['recipient'] == new['recipient'] and old['donor'] == new['donor'],
    'old_entry': old, 'merged_entry': new,
    'comparison_scope': 'All outcome metrics, baseline values, block indices, checkpoint identities and tensor isolation. Wall time, peak allocator usage, timestamps and commit metadata are execution metadata, not scientific outcomes.',
})
print(json.dumps(comparisons, indent=2), flush=True)
print('EXACT REPRODUCTION PASS' if passed else 'REPRODUCTION FAIL', flush=True)
raise SystemExit(0 if passed else 1)
