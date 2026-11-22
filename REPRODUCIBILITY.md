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

# Reproducibility note: phage-identity sweep (Fig 4B)

Investigated while extending the phage-held-out RBP-identity sweep to lower
thresholds (`SWEEP_EXTENSION_BRIEF.md`). Re-running `2d_phage_holdout.py
--cluster-identity` at 95/90/80% gave group counts (45/38/29) that disagreed
sharply with CLAUDE.md's established numbers (89/80/69); only 100% matched
(99 = 99).

**Cause, confirmed:** the historical sweep ran through
`cluster_phages_by_identity()`'s silent `difflib` fallback, because
`rapidfuzz` was absent from the `keymotif` environment at the time (it was
only `pip install`ed during this investigation). Rebuilding the phage
adjacency with that exact fallback call --
`difflib.SequenceMatcher(None, a, b).ratio()`, default `autojunk=True`,
unoverridden -- and clustering with the same union-find the code has always
used reproduces CLAUDE.md's 99/89/80/69 exactly at 100/95/90/80%. At 100% the
two metrics additionally put the *same phages in the same groups*, not just
the same count (verified directly, `confirm_100_partitions.py`), as expected
since both metrics agree exactly at perfect identity.

`difflib.SequenceMatcher.ratio()` with `autojunk=True` is not normalised
Levenshtein identity -- it discounts any element occurring in >1% of either
sequence as junk before block-matching, which every RBP sequence here is long
and amino-acid-limited enough to trigger. The methods text and
`SWEEP_EXTENSION_BRIEF.md` both describe the metric as normalised Levenshtein
over whole RBP sequences; that description matches what `rapidfuzz` computes
(`Levenshtein.normalized_similarity`, verified against an independent
from-scratch DP edit-distance implementation, bit-identical on 5 random
sequence pairs) and not what actually produced the published 99/89/80/69 /
0.764/0.756/0.731/0.691. So: **the metric accounts for the group-count
divergence -- not the clustering algorithm, not the environment, and not
deduplication.**

**Dedup -- corrected.** An earlier version of this note claimed the published
phage-identity-sweep numbers were undeduplicated, by analogy with the no-op
`drop_duplicates()` documented elsewhere in this codebase for scripts
2/3/5, and ruled dedup out as a factor on that basis. **That was wrong and
was not checked against the actual historical logs before being written.**
`logs/phase1_2d-{100,95,90,80}.log` show otherwise:

| threshold | run timestamp (2026-08-23) | command | groups (merges) | rows after dedup | folds | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| 100% | 04:07:53 | `--cluster-identity 1.00 --dedup` | 99 (6) | 25,120 | 99 | 0.764 | 0.262 |
| 95%  | 04:27:46 | `--cluster-identity 0.95 --dedup` | 89 (16) | 25,120 | 89 | 0.756 | 0.246 |
| 90%  | 04:45:09 | `--cluster-identity 0.90 --dedup` | 80 (25) | 25,120 | 80 | 0.731 | 0.222 |
| 80%  | 05:00:44 | `--cluster-identity 0.80 --dedup` | 69 (36) | 25,120 | 69 | 0.691 | 0.167 |

All four: every historical run passed `--dedup`, which reduced the table
from 487,400 to 25,120 rows -- this is `2d_phage_holdout.py`'s own explicit,
subset-based dedup (`["accession","phage_ID","protein_ID","virus_idx"]`), a
different call from the no-op `drop_duplicates()` in scripts 2/3/5, and it
is not a no-op here. All four: `rapidfuzz not installed; falling back to
difflib (much slower)` is printed explicitly in the log, not inferred --
confirming the metric finding above directly rather than by reconstruction.
All four ran within one ~53-minute session on 2026-08-23, dating the
environment boundary precisely.

None of the last two days' reruns (sitting A: 70/60/50/40; the 80% rerun;
the 100% "control" below) passed `--dedup`. They ran on the full 487,400-row
table. **Dedup was never actually ruled out as a factor -- it differed
between every historical run and every rerun, including at 100% where the
metric and partition are proven identical.** Those reruns are a legitimate
sensitivity analysis of the dedup choice on their own terms, and are kept,
but are not on the same footing as the published numbers and should not be
reported as if they were.

**100% control, run.** Same command form as sitting A
(`--cluster-identity 1.00`, no `--dedup`, `keymotif` env, `cuda:0`), compared
against `results/prev_env_backup/2d_phage_holdout_id100.pkl`:

- Labels: identical at all 10,006 positions. Same pair table, same 333
  positives, either run.
- Partition at threshold 1.00: identical under difflib and under
  rapidfuzz-Levenshtein (`confirm_100_partitions.py`) -- same 99 groups, same
  phages in each. The metric cannot be the cause of this delta; at 100% it is
  not even a live variable.
- Scores: all 10,006 positions differ (mean abs diff 0.042, max 0.83).
  Recomputed ROC-AUC 0.749 against the published/backed-up 0.764.

**Retracted, not withdrawn.** The "environment/build divergence" conclusion
above was wrong, not merely unsupported. Rerunning threshold 1.00 with
`--dedup` (`--cluster-identity 1.00 --dedup`, otherwise identical to sitting
A) gives ROC 0.764, PR-AUC 0.262 -- matching the published values exactly.
Checked at the strongest level available: `results/2d_phage_holdout_id100.pkl`
against `results/prev_env_backup/2d_phage_holdout_id100.pkl` is bit-for-bit
identical across all 10,006 labels and all 10,006 scores, not merely
agreement to three decimals. The entire 0.764 vs. 0.749 delta was the missing
`--dedup` flag. There is no residual gap at 100% to attribute to environment
or build divergence for this script. (Script 0's own gap, documented above,
is unrelated and still open -- see the note below.)

**File-timestamp check, inconclusive.** Compared mtimes of the backed-up
id-sweep pickles (`results/prev_env_backup/2d_phage_holdout_id{100,95,90,80}.pkl`)
against the 2x2 ablation pickles (`2b_AUCs_serotype_only.pkl`,
`2c_AUCs_rbp_only.pkl`, `2d_phage_holdout_clustered.pkl`,
`2d_phage_holdout_clustered_noserotype.pkl`): all eight files land within 0.085
seconds of each other (2026-08-29 21:48:23 +0200). That is a git-checkout
artifact -- this repo's restructure commit (`1d57f9f`, bulk `git mv` into
`src/analysis/manuscript/results/motifs`) rewrites mtimes for everything it
touches to the moment of checkout, not to when each pickle was originally
computed. mtimes cannot date an environment boundary here; this line of
evidence is unusable, neither confirming nor contradicting anything above.

**Status: closed.** The sweep reproduces exactly -- bit-for-bit at 100%, and
by the same mechanism (missing `--dedup`) that already explains why the
no-dedup reruns diverge at 95/90/80. The reported figure uses
`--cluster-identity <t> --dedup` for all eight thresholds, matching the
historical command form. Sitting A (70/60/50/40), the 80% rerun, and the
undeduplicated 100% control are retained as a dedup sensitivity analysis
(documented above), not as the reported sweep.

**Script 0's gap is a different situation -- checked, not resolved by the
same mechanism.** `logs/stage_0.log` (`0_original_replica_compact.py`,
2026-08-20, `device=cuda`) is the only script-0 log in this repo; no sibling
log exists. It records `Final AUC after host-averaged adjustment: 0.817` at
the 100% threshold (185 folds) -- matching the already-documented GPU rerun
(0.8167, rounds to 0.817), not the published 0.807. So this log is the
*rerun*, not a record of whatever produced the original 0.807 pickle in
`results_published/` -- that run happened on a different, unavailable
machine, as already noted above, and left no log in this repo. Within this
log: no `--dedup`-equivalent flag exists for this script (it has none to
pass), and no missing-optional-dependency message appears anywhere (this
script collapses host representations directly; it has no RBP-identity
clustering step and nothing resembling the rapidfuzz/difflib fallback). The
flag-difference mechanism that explained the phage sweep does not carry over
to script 0 -- there is no analogous flag here to have gotten wrong. Script
0's gap remains open, for a different and still-unidentified reason.

# Note: best-HSP tie-break in the metric calibration

`results/metric_calibration_pairs.csv` keeps one BLASTp HSP per unordered RBP
pair. BLASTp reports each pair in both orientations, and **1,600 of the 2,551
hit pairs have more than one row at their maximum bitscore**; in 886 of those
the tied rows disagree about query coverage, because coverage is orientation-
dependent while bitscore is not.

The previously committed version of this file resolved those ties through
pandas' default `quicksort`, which is unstable: `kind="quicksort"` reproduces
that column exactly while `kind="mergesort"` diverges on 606 pairs. The choice
was therefore an artefact of the sort, not a rule, and no documented rule
reproduces it.

`analysis/05_metric_audit/metric_calibration.py` now applies an explicit,
fully ordered tie-break -- bitscore, then query coverage, then pident, then
(qseqid, sseqid) -- exposed as `--tie-break {qcovs,lex}`. The default `qcovs`
matches the coverage semantics the grouping already uses
(`make_blast_groupings.py`'s 50% floor). The regenerated file differs from the
old one in `blast_qcovs` on 432 of 2,551 pairs.

What this does and does not move:

- **Unchanged:** every grouping CSV (all eight identical as partitions),
  `S6_calibration_reverse.tex` (byte-identical), all bin counts, and every
  normalised-Levenshtein statistic -- i.e. every number the reviewer reply
  quotes. `blast_bitscore` and `nlev` are bit-identical.
- **Changed:** the median-query-coverage column of `S6_calibration_forward.tex`
  only, by 2-8 points per bin (e.g. 30-40%: 52% -> 56%). Regenerated with
  `--tie-break qcovs`.

# Note: there are two phage-held-out sweeps in this repo

They group phages by different metrics and must not be read as one curve.

- **Normalised Levenshtein grouping** -- `results/2d_phage_holdout_id*.pkl`,
  collated in `results/sweep_summary.md` by `sweep_collate.py`. This is the
  sweep whose published values the reproduction work above concerns
  (0.764 at 100%, 0.675 at 95%).
- **BLASTp grouping** (50% query-coverage floor) --
  `results/2d_phage_holdout_blast*.pkl` with `results/groupings/*.csv`,
  decomposed by `analysis/05_metric_audit/fold_breakdown.py` into
  `results/a1_family_split.csv` and plotted by
  `manuscript/make_fig4_v3_2panel.py`. **This is the grouping used in Fig 4**
  (0.764 at 100%, 0.686 at 95%).

The two agree at 100% by construction and diverge below it.
