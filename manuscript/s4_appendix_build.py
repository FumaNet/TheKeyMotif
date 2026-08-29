"""Build S4 Appendix: KlebPhaCol sensitivity analyses.

Produces one table covering the three analyses currently promised in Methods but
not reported: medium (LB / TSB), label mapping (strict / permissive), and scope
(all hosts / excluding hosts whose capsule type is absent from training).

INPUT
-----
A long-format CSV with one row per scored (phage, host, medium, mapping) pair.
Set the column names in CONFIG below to match your file. Required content:

    medium      'LB' or 'TSB'
    mapping     'strict' or 'permissive'
    stratum     'novel' | 'related' | 'near-identical'
    y_true      0/1 label
    score       model score for the pair (max over the phage's RBPs)
    host_seen   True if the host capsule type is in the training vocabulary
    phage       KlebPhaCol phage id, used for phage-level bootstrap

OUTPUT
------
S4_appendix_table.csv   machine-readable
S4_appendix_table.tex   LaTeX, ready to \input into the manuscript

Confidence intervals are reported two ways. Pair-level bootstrap treats every
pair as independent. Phage-level bootstrap resamples KlebPhaCol phages, which is
the honest interval where test RBPs are concentrated on few training matches --
notably the near-identical stratum, where all 34 RBPs match one of two training
phages.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# CONFIG -- edit to match your predictions file
# ----------------------------------------------------------------------------
INFILE = os.path.join(HERE, "klebphacol_predictions.csv")
COL = dict(medium="medium", mapping="mapping", stratum="stratum",
           y_true="y_true", score="score", host_seen="host_seen", phage="phage")
N_BOOT = 2000
SEED = 0
STRATA = ["novel", "related", "near-identical"]
# ----------------------------------------------------------------------------


def metrics(y, s):
    """ROC-AUC, PR-AUC and lift over the base rate. None if one class only."""
    if len(np.unique(y)) < 2:
        return None, None, None, None
    base = y.mean()
    pr = average_precision_score(y, s)
    return roc_auc_score(y, s), pr, base, pr / base


def boot_ci(y, s, groups=None, n_boot=N_BOOT, seed=SEED):
    """Bootstrap 95% CI for ROC-AUC. Resamples groups if given, else rows."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    if groups is None:
        idx_pool = [np.array([i]) for i in range(len(y))]
    else:
        groups = np.asarray(groups)
        idx_pool = [np.where(groups == g)[0] for g in np.unique(groups)]

    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_pool), len(idx_pool))
        idx = np.concatenate([idx_pool[i] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], s[idx]))
    if len(out) < n_boot * 0.5:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    df = pd.read_csv(INFILE)
    rows = []

    for medium in ["LB", "TSB"]:
        for mapping in ["strict", "permissive"]:
            for scope, mask in [("all hosts", None), ("seen capsules only", True)]:
                sub = df[(df[COL["medium"]] == medium) &
                         (df[COL["mapping"]] == mapping)]
                if mask is not None:
                    sub = sub[sub[COL["host_seen"]]]
                for stratum in STRATA:
                    t = sub[sub[COL["stratum"]] == stratum]
                    if len(t) == 0:
                        continue
                    y, s = t[COL["y_true"]].values, t[COL["score"]].values
                    roc, pr, base, lift = metrics(y, s)
                    if roc is None:
                        continue
                    lo_p, hi_p = boot_ci(y, s)
                    lo_g, hi_g = boot_ci(y, s, groups=t[COL["phage"]].values)
                    rows.append(dict(
                        medium=medium, mapping=mapping, scope=scope,
                        stratum=stratum, n=len(t), n_pos=int(y.sum()),
                        base_rate=round(100 * base, 1),
                        roc_auc=round(roc, 3),
                        roc_lo_pair=None if lo_p is None else round(lo_p, 3),
                        roc_hi_pair=None if hi_p is None else round(hi_p, 3),
                        roc_lo_phage=None if lo_g is None else round(lo_g, 3),
                        roc_hi_phage=None if hi_g is None else round(hi_g, 3),
                        pr_auc=round(pr, 3), lift=round(lift, 2),
                        n_phages=t[COL["phage"]].nunique()))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "S4_appendix_table.csv"), index=False)
    print(out.to_string(index=False))

    def ci(lo, hi):
        return "--" if pd.isna(lo) else f"{lo:.3f}--{hi:.3f}"

    lines = [
            r"\begin{table}[h]", r"\centering", r"\small",
            r"\caption{{\bf Sensitivity of external-evaluation results to growth "
            r"medium, label mapping and host scope.} Lift is PR-AUC divided by the "
            r"base rate of that stratum. Pair-level intervals treat every pair as "
            r"independent; phage-level intervals resample KlebPhaCol phages.}",
            r"\begin{tabular}{llllrrrrrrr}", r"\hline",
            r"Medium & Mapping & Scope & Stratum & $n$ & pos. & Base & ROC-AUC "
            r"& CI (pair) & PR-AUC & Lift \\", r"\hline"]
    for _, r in out.iterrows():
        lines.append(
            f"{r.medium} & {r.mapping} & {r.scope} & {r.stratum} & {r.n} & "
            f"{r.n_pos} & {r.base_rate}\\% & {r.roc_auc:.3f} & "
            f"{ci(r.roc_lo_pair, r.roc_hi_pair)} & {r.pr_auc:.3f} & "
            f"{r.lift:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\label{tab:s4_sensitivity}",
                r"\end{table}"]
    open(os.path.join(HERE, "S4_appendix_table.tex"), "w").write("\n".join(lines) + "\n")
    print("\nwrote S4_appendix_table.csv and S4_appendix_table.tex")


if __name__ == "__main__":
    main()
