# Phage-held-out identity sweep -- collated results

One environment, one machine: python 3.12.13, xgboost 2.1.4 (USE_CUDA=True), numpy 2.2.2, pandas 2.2.3, scikit-learn 1.6.1, RapidFuzz 3.14.6, GTX 1650 driver 566.36, KM_DEVICE unset (cuda), random_state=0.

Chance: ROC 0.500, PR 0.033.


## Main table

| thresh | ROC-AUC | PR-AUC | published ROC | delta | folds | largest fold | share | zero-pos folds | pairs | positives |
|---|---|---|---|---|---|---|---|---|---|---|
| 100% | 0.764 | 0.262 | 0.764 | +0.000 | 99 | 18,732 | 3.8% | 11 of 99 | 10006 | 333 |
| 95% | 0.675 | 0.102 | 0.756 | -0.081 **>0.02** | 45 | 119,480 | 24.5% | 3 of 45 | 10006 | 333 |
| 90% | 0.652 | 0.088 | 0.731 | -0.079 **>0.02** | 38 | 119,480 | 24.5% | 2 of 38 | 10006 | 333 |
| 80% | 0.634 | 0.089 | 0.691 | -0.057 **>0.02** | 29 | 207,648 | 42.6% | 2 of 29 | 10006 | 333 |
| 70% | 0.590 | 0.059 | n/a | n/a | 25 | 246,180 | 50.5% | 2 of 25 | 10006 | 333 |
| 60% | 0.592 | 0.060 | n/a | n/a | 24 | 246,180 | 50.5% | 2 of 24 | 10006 | 333 |
| 50% | 0.577 | 0.057 | n/a | n/a | 21 | 246,180 | 50.5% | 2 of 21 | 10006 | 333 |
| 40% | 0.574 | 0.056 | n/a | n/a | 19 | 246,180 | 50.5% | 2 of 19 | 10006 | 333 |

## Largest fold vs the rest (same predictions, split by fold)

| thresh | largest-fold pairs | pos | ROC | PR | other pairs | pos | ROC | PR |
|---|---|---|---|---|---|---|---|---|
| 100% | 138 | 2 | 0.875 | 0.0526 | 9868 | 331 | 0.764 | 0.2632 |
| 95% | 2504 | 42 | 0.623 | 0.0470 | 7502 | 291 | 0.686 | 0.1136 |
| 90% | 2504 | 42 | 0.623 | 0.0470 | 7502 | 291 | 0.663 | 0.0984 |
| 80% | 3566 | 64 | 0.450 | 0.0239 | 6440 | 269 | 0.675 | 0.1259 |
| 70% | 4166 | 70 | 0.439 | 0.0137 | 5840 | 263 | 0.642 | 0.1013 |
| 60% | 4166 | 70 | 0.439 | 0.0137 | 5840 | 263 | 0.645 | 0.1040 |
| 50% | 4166 | 70 | 0.439 | 0.0137 | 5840 | 263 | 0.632 | 0.1059 |
| 40% | 4166 | 70 | 0.439 | 0.0137 | 5840 | 263 | 0.625 | 0.1001 |

## Checks

- id100: recomputed ROC 0.7644 vs stored 0.764 -- OK; folds 99 vs preflight 99 -- OK
- id95: recomputed ROC 0.6747 vs stored 0.675 -- OK; folds 45 vs preflight 45 -- OK
- id90: recomputed ROC 0.6524 vs stored 0.652 -- OK; folds 38 vs preflight 38 -- OK
- id80: recomputed ROC 0.6335 vs stored 0.634 -- OK; folds 29 vs preflight 29 -- OK
- id70: recomputed ROC 0.5903 vs stored 0.59 -- OK; folds 25 vs preflight 25 -- OK
- id60: recomputed ROC 0.5922 vs stored 0.592 -- OK; folds 24 vs preflight 24 -- OK
- id50: recomputed ROC 0.5767 vs stored 0.577 -- OK; folds 21 vs preflight 21 -- OK
- id40: recomputed ROC 0.5745 vs stored 0.574 -- OK; folds 19 vs preflight 19 -- OK

## Notes

- none

## Not run (per pre-flight)

- 30%: degenerate, one group holds 94 of 105 phages
- 20%, 10%: collapsed to a single group; nothing to train on
