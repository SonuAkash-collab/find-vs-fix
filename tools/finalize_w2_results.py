"""Require the complete approved grid, verify source identities, then document them."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/wronglabel'
OLD = '23bc97488a0ece698f9412ec8e73085d8236bc97'
MERGED = 'c3851d1e9fa41de80b8c597678a23606f730fe8a'
BLOCKS = {str(b) for b in range(12)} | {'0-1-2', '3-4-5', '6-7-8', '9-10-11', '10-11'}
RETRAIN = BLOCKS | {'all', 'ctrl_noisy_9-10-11', 'ctrl_noisy_all'}


def main():
    proof = json.loads((OUT / 'reproduction/check.json').read_text())
    assert proof['all_scientific_numbers_exact'] is True
    counts = {OLD: {'swap': 0, 'retrain': 0, 'lens': 0},
              MERGED: {'swap': 0, 'retrain': 0, 'lens': 0}}
    for seed in range(3):
        swap = json.loads((OUT / f'swap/seed{seed}.json').read_text())
        assert set(swap['cases']) == {'a', 'b', 'ci', 'cii'}
        for entries in swap['cases'].values():
            assert set(entries) == BLOCKS
            for entry in entries.values():
                assert entry['untouched_tensors_verified'] is True
                meta = entry['provenance']
                assert meta['source_dirty'] is False
                counts[meta['git_commit']]['swap'] += 1
        retrain = json.loads((OUT / f'retrain/seed{seed}.json').read_text())
        assert set(retrain['runs']) == RETRAIN
        for entry in retrain['runs'].values():
            assert entry['frozen_params_unchanged'] is True
            assert entry['n_trained_tensors_changed'] > 0
            assert entry['train']['steps'] == 433
            assert entry['smoke'] is False
            meta = entry['provenance']
            assert meta['source_dirty'] is False
            counts[meta['git_commit']]['retrain'] += 1
        for kind in ('clean', 'noisy'):
            lens = json.loads((OUT / f'logit_lens/wl_{kind}_seed{seed}.json').read_text())
            assert len(lens['layers']) == 13
            assert lens['source_dirty'] is False
            counts[lens['git_commit']]['lens'] += 1
    subprocess.run([sys.executable, str(ROOT / 'tools/verify_w2_sources.py'),
                    '--commit', MERGED, '--allow-old-commit', OLD], cwd=ROOT, check=True)
    summary = {'complete': True, 'source_counts': counts,
               'reproduction': 'results/wronglabel/reproduction/check.json',
               'source_check': 'results/wronglabel/W2_SOURCE_CHECK.json'}
    (OUT / 'W2_COMPLETENESS.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# W2 source traceability', '',
             f'Original source: `{OLD}` (published on `w2-local-history`).',
             f'Merged source: `{MERGED}` (PR #45).', '',
             'The seed-0, case-a, block-0 swap was reproduced on the merged code. All outcome metrics,',
             'baseline values, checkpoint hashes and tensor-isolation checks matched exactly.',
             'Timing, allocator peaks and timestamps are execution metadata and were not equality criteria.', '',
             '| Source commit | Swaps | Retraining | Lens models |', '|---|---:|---:|---:|']
    for commit, values in counts.items():
        lines.append(f"| `{commit}` | {values['swap']} | {values['retrain']} | {values['lens']} |")
    lines += ['', 'Each intervention retains its own execution provenance. The resumed swap container carries',
              'merged-source container metadata and preserves the original container provenance separately.',
              'No original intervention hash was relabelled as the merged source.', '',
              'The source check reads every recorded source hash, including nested intervention records,',
              'from the corresponding pinned Git commit. It explicitly identifies Git LF → Windows CRLF',
              'checkout conversion and rejects other byte changes. The old commit is accepted only with',
              'the recorded exact reproduction evidence.', '',
              'Reproduce the source check:', '', '```powershell',
              f'python tools/verify_w2_sources.py --commit {MERGED} --allow-old-commit {OLD}',
              '```', '', 'Reproduce the comparison: `python tools/reproduce_one_w2.py`.',
              'Complete the grid: `python src/run_w2.py`.',
              'Validate the complete grid and regenerate this file: `python tools/finalize_w2_results.py`.', '',
              'The retraining diff is saved beside the reproduction evidence in',
              '`results/wronglabel/reproduction/retrain-23bc974-to-c3851d1.diff`.', '']
    (OUT / 'W2_TRACEABILITY.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
