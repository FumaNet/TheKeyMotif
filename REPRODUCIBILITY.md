# Reproducibility note: script 0 (PHL-AVG)

This note documents a known, investigated gap between the published PHL-AVG
ROC-AUC and what re-running script 0 (`0_original_replica.py` /
`0_original_replica_compact.py`) produces on the machine used for this repo
cleanup. It exists so the gap is documented rather than left as informal notes
in `SETUP_LOCAL.md`.

## The numbers

At the 100% grouping threshold:

| source | ROC-AUC |
|---|---|
| Published / `Results_published/` (canonical) | 0.807 |
| Rerun here, GPU (GTX 1650, `gpu_hist`) | 0.8167 |
| Rerun here, CPU (`hist`) | 0.825 |

`Results_published/` is produced on a CUDA build, on a different GPU than the
machine this repo was cleaned up on. It is the canonical set for script 0 and
is not overwritten by any of the investigation below.

## What has been ruled out

- **Nondeterminism.** Two independent GPU runs of the same script, same data,
  same (unset, XGBoost-default) seed are bit-identical: max absolute score
  difference across 10,006 pooled predictions is `0.0`. Five explicit seeds
  (0–4) on GPU all give exactly `0.8167`, std `0.0`. The gap is not run-to-run
  noise.
- **A data-handling bug in the compact refactor.** An independent,
  from-scratch reference implementation of the original per-(accession,
  phage_ID) mean-pooling logic (no call into `keymotif_data.py`) was compared
  against `keymotif_data.precompute_collapsed()`'s output on the real dataset
  at the 100% threshold: identical row count, identical (accession, phage_ID)
  identity, feature matrix `np.allclose` to float32 ULP (max abs diff
  `8.6e-06`), identical labels. The classifier receives the same inputs either
  way.
- **Device class alone.** CPU and GPU disagree with each other (0.825 vs
  0.8167 on identical inputs), so device does matter, but neither reproduces
  the published 0.807 — CPU is further from it than GPU is.

## What has not been ruled out

The environment/build differs from `packagelist.txt`'s pins: numpy 1.26.4
(vs. 2.2.2 pinned), pandas 2.0.3 (vs. 2.2.3 pinned), scikit-learn 1.9.0 (vs.
1.6.1 pinned). XGBoost matches exactly (2.1.4 = 2.1.4 pinned). This has not
yet been tested and is the live remaining candidate.

## Tolerance in `verify_reproduction.py`

`verify_reproduction.py` sets `TOL_CUDA["0"] = 0.003` for GPU reruns. The
observed cross-machine GPU difference (0.8167 vs. 0.807 = +0.010) exceeds that
by more than 3x. The tolerance reflects same-architecture GPU noise, not
cross-architecture drift, and should be widened to reflect that reality —
noted here rather than changed silently in the verification script.

## Scale, for context

The ablation and external-validation conclusions in this work rest on AUC
gaps of roughly 0.07 to 0.23 between conditions. The unexplained script-0 gap
(0.01–0.02) is an order of magnitude smaller than the effects the paper's
claims depend on, and does not by itself call those claims into question.

## Status

Open. Next step, when resumed, is testing the pinned `packagelist.txt`
versions in a fresh environment.
