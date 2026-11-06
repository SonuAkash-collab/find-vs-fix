# W2 source traceability

Original source: `23bc97488a0ece698f9412ec8e73085d8236bc97` (published on `w2-local-history`).
Merged source: `c3851d1e9fa41de80b8c597678a23606f730fe8a` (PR #45).

The seed-0, case-a, block-0 swap was reproduced on the merged code. All outcome metrics,
baseline values, checkpoint hashes and tensor-isolation checks matched exactly.
Timing, allocator peaks and timestamps are execution metadata and were not equality criteria.

| Source commit | Swaps | Retraining | Lens models |
|---|---:|---:|---:|
| `23bc97488a0ece698f9412ec8e73085d8236bc97` | 45 | 0 | 0 |
| `c3851d1e9fa41de80b8c597678a23606f730fe8a` | 159 | 60 | 6 |

Each intervention retains its own execution provenance. The resumed swap container carries
merged-source container metadata and preserves the original container provenance separately.
No original intervention hash was relabelled as the merged source.

The source check reads every recorded source hash, including nested intervention records,
from the corresponding pinned Git commit. It explicitly identifies Git LF → Windows CRLF
checkout conversion and rejects other byte changes. The old commit is accepted only with
the recorded exact reproduction evidence.

Reproduce the source check:

```powershell
python tools/verify_w2_sources.py --commit c3851d1e9fa41de80b8c597678a23606f730fe8a --allow-old-commit 23bc97488a0ece698f9412ec8e73085d8236bc97
```

Reproduce the comparison: `python tools/reproduce_one_w2.py`.
Complete the grid: `python src/run_w2.py`.
Validate the complete grid and regenerate this file: `python tools/finalize_w2_results.py`.

The retraining diff is saved beside the reproduction evidence in
`results/wronglabel/reproduction/retrain-23bc974-to-c3851d1.diff`.
