"""Collate every completed identity-sweep threshold into one reporting table.

Needs no GPU and no retraining. For each threshold that has a .pkl it reports
the fields SWEEP_RETRY.md asks for -- ROC-AUC, PR-AUC, folds, largest-fold
share, zero-positive folds, pairs, positives -- plus the within-threshold
split between the single largest fold and everything else.

The pooled prediction order in the .pkl is reconstructed, not assumed:
2d_phage_holdout.py emits folds in LeaveOneGroupOut order (sorted group id)
and, within a fold, groupby(["accession","phage_ID"]) with sort=True. That is
exactly a lexicographic sort on (phage_group, accession, phage_ID). The
reconstruction is accepted only if the labels it implies are identical to the
stored labels; otherwise that threshold's split is reported as UNVERIFIED and
omitted rather than guessed at.

Writes results/sweep_summary.md and prints the same content.
"""
import importlib.util
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
import keymotif_data as kd  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "holdout", os.path.join(REPO, "analysis", "02_ablation", "2d_phage_holdout.py"))
holdout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(holdout)

THRESHOLDS = [100, 95, 90, 80, 70, 60, 50, 40]
PUBLISHED = {100: (0.764, 0.262), 95: (0.756, 0.246),
             90: (0.731, 0.222), 80: (0.691, 0.167)}
PREFLIGHT_FOLDS = {100: 99, 95: 45, 90: 38, 80: 29, 70: 25, 60: 24, 50: 21, 40: 19}
CHANCE_ROC, CHANCE_PR = 0.5, 0.033


def metrics(lab, sc):
    if len(lab) == 0 or lab.sum() == 0 or lab.sum() == len(lab):
        return None, None
    fpr, tpr, _ = roc_curve(lab, sc)
    prec, rec, _ = precision_recall_curve(lab, sc)
    return auc(fpr, tpr), auc(rec, prec)


def fmt(x, nd=3):
    return "n/a" if x is None else ("%.*f" % (nd, x))


pairs_base, _h, _v = kd.load()
rows = []
notes = []

for t in THRESHOLDS:
    pkl = "results/2d_phage_holdout_id%d.pkl" % t
    if not os.path.exists(pkl):
        notes.append("id%d: no .pkl -- not run" % t)
        continue

    with open(pkl, "rb") as fh:
        stored_lab, stored_sc, stored_auc = pickle.load(fh)
    stored_lab = np.asarray(stored_lab)
    stored_sc = np.asarray(stored_sc)

    pairs = pairs_base.copy()
    pg = holdout.cluster_phages_by_identity(t / 100.0)
    pairs["phage_group"] = pairs["phage_ID"].map(pg)
    if pairs["phage_group"].isna().any():
        pairs = pairs.dropna(subset=["phage_group"])

    n_folds = int(pairs["phage_group"].nunique())
    sizes = pairs.groupby("phage_group").size()
    biggest = sizes.idxmax()
    big_share = 100.0 * sizes.max() / len(pairs)
    pos_per_group = pairs[pairs["label"] == 1].groupby("phage_group").size()
    zero_pos = n_folds - len(pos_per_group)

    roc, pr = metrics(stored_lab, stored_sc)
    agrees = (roc is not None) and (round(roc, 3) == stored_auc)
    if not agrees:
        notes.append("id%d: recomputed ROC %s != stored %s" % (t, fmt(roc, 4), stored_auc))
    if n_folds != PREFLIGHT_FOLDS[t]:
        notes.append("id%d: folds %d != preflight %d -- GROUPING MISMATCH"
                     % (t, n_folds, PREFLIGHT_FOLDS[t]))

    # Reconstruct pooled order and verify against the stored labels.
    key = (pairs.assign(_lab=pairs["label"].astype(int))
                .groupby(["phage_group", "accession", "phage_ID"], sort=True)["_lab"]
                .first())
    recon_lab = key.values
    verified = (len(recon_lab) == len(stored_lab)
                and np.array_equal(recon_lab, stored_lab))

    if verified:
        fold_of = key.index.get_level_values("phage_group").values
        m = fold_of == biggest
        roc_b, pr_b = metrics(stored_lab[m], stored_sc[m])
        roc_r, pr_r = metrics(stored_lab[~m], stored_sc[~m])
        split = dict(n_big=int(m.sum()), pos_big=int(stored_lab[m].sum()),
                     roc_big=roc_b, pr_big=pr_b,
                     n_rest=int((~m).sum()), pos_rest=int(stored_lab[~m].sum()),
                     roc_rest=roc_r, pr_rest=pr_r)
    else:
        split = None
        notes.append("id%d: pooled-order reconstruction UNVERIFIED "
                     "(recon %d rows vs stored %d) -- split omitted"
                     % (t, len(recon_lab), len(stored_lab)))

    rows.append(dict(t=t, roc=roc, pr=pr, stored=stored_auc, agrees=agrees,
                     folds=n_folds, big_share=big_share, zero_pos=zero_pos,
                     n_pairs=len(stored_lab), pos=int(stored_lab.sum()),
                     big_rows=int(sizes.max()), total_rows=len(pairs),
                     split=split))

out = []
out.append("# Phage-held-out identity sweep -- collated results\n")
out.append("One environment, one machine: python 3.12.13, xgboost 2.1.4 "
           "(USE_CUDA=True), numpy 2.2.2, pandas 2.2.3, scikit-learn 1.6.1, "
           "RapidFuzz 3.14.6, GTX 1650 driver 566.36, KM_DEVICE unset (cuda), "
           "random_state=0.\n")
out.append("Chance: ROC %.3f, PR %.3f.\n" % (CHANCE_ROC, CHANCE_PR))

out.append("\n## Main table\n")
out.append("| thresh | ROC-AUC | PR-AUC | published ROC | delta | folds | "
           "largest fold | share | zero-pos folds | pairs | positives |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    pub = PUBLISHED.get(r["t"], (None, None))[0]
    delta = None if pub is None or r["roc"] is None else r["roc"] - pub
    flag = ""
    if delta is not None and abs(delta) > 0.02:
        flag = " **>0.02**"
    out.append("| %d%% | %s | %s | %s | %s%s | %d | %s | %.1f%% | %d of %d | %d | %d |" % (
        r["t"], fmt(r["roc"]), fmt(r["pr"]), fmt(pub),
        ("%+.3f" % delta) if delta is not None else "n/a", flag,
        r["folds"], format(r["big_rows"], ","), r["big_share"],
        r["zero_pos"], r["folds"], r["n_pairs"], r["pos"]))

out.append("\n## Largest fold vs the rest (same predictions, split by fold)\n")
out.append("| thresh | largest-fold pairs | pos | ROC | PR | other pairs | pos | ROC | PR |")
out.append("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    s = r["split"]
    if s is None:
        out.append("| %d%% | UNVERIFIED | | | | | | | |" % r["t"])
        continue
    out.append("| %d%% | %d | %d | %s | %s | %d | %d | %s | %s |" % (
        r["t"], s["n_big"], s["pos_big"], fmt(s["roc_big"]), fmt(s["pr_big"], 4),
        s["n_rest"], s["pos_rest"], fmt(s["roc_rest"]), fmt(s["pr_rest"], 4)))

out.append("\n## Checks\n")
for r in rows:
    out.append("- id%d: recomputed ROC %s vs stored %s -- %s; folds %d vs preflight %d -- %s"
               % (r["t"], fmt(r["roc"], 4), r["stored"],
                  "OK" if r["agrees"] else "MISMATCH",
                  r["folds"], PREFLIGHT_FOLDS[r["t"]],
                  "OK" if r["folds"] == PREFLIGHT_FOLDS[r["t"]] else "MISMATCH"))
out.append("\n## Notes\n")
if notes:
    for n in notes:
        out.append("- %s" % n)
else:
    out.append("- none")

out.append("\n## Not run (per pre-flight)\n")
out.append("- 30%: degenerate, one group holds 94 of 105 phages")
out.append("- 20%, 10%: collapsed to a single group; nothing to train on")

text = "\n".join(out) + "\n"
with open("results/sweep_summary.md", "w", encoding="utf-8") as fh:
    fh.write(text)
print(text)
