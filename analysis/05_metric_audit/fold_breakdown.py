"""
fold_breakdown.py — split the BLASTp-grouped phage-held-out predictions by fold.

THE QUESTION
------------
Across the BLASTp-grouped sweep, lowering the threshold merges more and more
phages into one enormous fold. By 70% a single group holds 39 of the 105
phages. A pooled AUC over all folds mixes two different populations: the pairs
predicted while that whole cluster was held out, and everything else. If the
sweep's decay with threshold is really the cluster swallowing the pooled metric
rather than a uniform loss of transferability, the split shows it directly.

The 39 phages are a fixed set, identified once at the 70% grouping (where the
cluster reaches its full extent and stops growing -- it is the same 39 at 60%
and 50%). Holding the set fixed across all seven thresholds is what makes the
columns comparable: the subsets contain the same pairs at every threshold, so
only the model's predictions change, not the population being scored.

WHAT IT WRITES
--------------
results/a1_family_split.csv — one row per (threshold, subset) with n, positives,
ROC-AUC and PR-AUC.

RECONSTRUCTING THE POOLED ORDER
-------------------------------
The .pkl holds only (labels, scores, auc) -- no keys. The pooled order is
rebuilt the way sweep_collate.py does it: 2d_phage_holdout.py emits folds in
LeaveOneGroupOut order (sorted group id) and, within a fold,
groupby(["accession","phage_ID"], sort=True). That is a lexicographic sort on
(phage_group, accession, phage_ID). The reconstruction is accepted only if the
labels it implies are identical to the stored labels; a threshold that fails
label equality stops the script rather than being reported on a guess.

    python analysis/05_metric_audit/fold_breakdown.py
    python analysis/05_metric_audit/fold_breakdown.py --check
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))
import keymotif_data as kd  # noqa: E402

THRESHOLDS = [100, 95, 90, 80, 70, 60, 50]
GROUPINGS = "results/groupings/groups_blast%d.csv"
PICKLES = "results/2d_phage_holdout_blast%d.pkl"
OUT = "results/a1_family_split.csv"

# The threshold at which the cluster reaches its full extent. Taking the set
# from here, rather than per threshold, is what keeps the two subsets fixed.
ANCHOR = 70

# Validation targets, from the committed CSV.
EXPECT_N = {"cluster": (4166, 70), "other": (5840, 263)}
EXPECT = {  # threshold -> (roc_cluster, pr_cluster, roc_other, pr_other)
    100: (0.709, 0.306, 0.772, 0.252),
    95:  (0.629, 0.027, 0.694, 0.130),
    90:  (0.629, 0.027, 0.670, 0.127),
    80:  (0.496, 0.025, 0.665, 0.128),
    70:  (0.439, 0.014, 0.639, 0.099),
    60:  (0.439, 0.014, 0.624, 0.095),
    50:  (0.439, 0.014, 0.623, 0.099),
}
EXPECT_FOLDS = {100: 33, 95: 6, 90: 6, 80: 2, 70: 1, 60: 1, 50: 1}


def metrics(lab, sc):
    if len(lab) == 0 or lab.sum() in (0, len(lab)):
        return None, None
    fpr, tpr, _ = roc_curve(lab, sc)
    prec, rec, _ = precision_recall_curve(lab, sc)
    return auc(fpr, tpr), auc(rec, prec)


def anchor_phages():
    """The largest group of the anchor grouping, as a fixed set of phage IDs."""
    g = pd.read_csv(GROUPINGS % ANCHOR)
    sizes = g.groupby("group_id").phage_ID.apply(set)
    biggest = sizes.loc[sizes.map(len).idxmax()]
    print(f"Cluster taken from the {ANCHOR}% grouping: {len(biggest)} phages")
    return biggest


def reconstruct(pairs_base, thresh):
    """
    Pooled (label, fold) order implied by the grouping, in the order
    2d_phage_holdout.py emits it. Returns (labels, phage_ID, phage_group).
    """
    g = pd.read_csv(GROUPINGS % thresh)
    pg = dict(zip(g.phage_ID, g.group_id))

    pairs = pairs_base.copy()
    pairs["phage_group"] = pairs["phage_ID"].map(pg)
    if pairs["phage_group"].isna().any():
        pairs = pairs.dropna(subset=["phage_group"])

    key = (pairs.assign(_lab=pairs["label"].astype(int))
                .groupby(["phage_group", "accession", "phage_ID"],
                         sort=True)["_lab"].first())
    return (key.values,
            key.index.get_level_values("phage_ID").values,
            key.index.get_level_values("phage_group").values)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("-o", "--out", default=OUT)
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out if it exists")
    ap.add_argument("--check", action="store_true",
                    help="validate and report; do not write")
    args = ap.parse_args()

    if os.path.exists(args.out) and not (args.force or args.check):
        raise SystemExit(
            f"{args.out} exists. Re-run with --check to validate against it "
            "without writing, or --force to overwrite it.")

    cluster = anchor_phages()
    pairs_base, _h, _v = kd.load()

    rows, bad = [], []
    for t in THRESHOLDS:
        pkl = PICKLES % t
        if not os.path.exists(pkl):
            raise SystemExit(f"missing {pkl}")
        with open(pkl, "rb") as fh:
            lab, sc, _stored_auc = pickle.load(fh)
        lab, sc = np.asarray(lab), np.asarray(sc)

        recon_lab, phage, fold = reconstruct(pairs_base, t)
        if len(recon_lab) != len(lab) or not np.array_equal(recon_lab, lab):
            raise SystemExit(
                f"blast{t}: pooled-order reconstruction FAILED label equality "
                f"({len(recon_lab)} reconstructed rows vs {len(lab)} stored). "
                "Stopping: the split cannot be attributed without it.")

        m = np.isin(phage, list(cluster))
        n_folds = len(np.unique(fold[m]))
        for label, sel in ((f"A1* ({len(cluster)})", m),
                           (f"other ({105 - len(cluster)})", ~m)):
            roc, pr = metrics(lab[sel], sc[sel])
            rows.append(dict(thresh=t, subset=label, n=int(sel.sum()),
                             pos=int(lab[sel].sum()), roc=roc, pr=pr))

        # Validation against the committed numbers.
        rc, pc = metrics(lab[m], sc[m])
        ro, po = metrics(lab[~m], sc[~m])
        want = EXPECT[t]
        got = (round(rc, 3), round(pc, 3), round(ro, 3), round(po, 3))
        ok = got == want
        nk = (int(m.sum()), int(lab[m].sum())) == EXPECT_N["cluster"] and \
             (int((~m).sum()), int(lab[~m].sum())) == EXPECT_N["other"]
        fk = n_folds == EXPECT_FOLDS[t]
        print(f"  {'OK  ' if ok and nk and fk else 'DIFF'} blast{t:<4} "
              f"cluster {got[0]:.3f}/{got[1]:.3f} other {got[2]:.3f}/{got[3]:.3f}"
              f"  folds {n_folds}"
              f"  n {int(m.sum())}/{int(lab[m].sum())} "
              f"{int((~m).sum())}/{int(lab[~m].sum())}")
        if not ok:
            print(f"       expected {want[0]:.3f}/{want[1]:.3f} "
                  f"{want[2]:.3f}/{want[3]:.3f}")
            bad.append(f"blast{t} metrics")
        if not nk:
            bad.append(f"blast{t} subset sizes")
        if not fk:
            print(f"       expected {EXPECT_FOLDS[t]} folds")
            bad.append(f"blast{t} fold count")

    df = pd.DataFrame(rows)
    if args.check:
        print("\n--check: nothing written.")
    else:
        df.to_csv(args.out, index=False)
        print(f"\nWrote {args.out} ({len(df)} rows)")

    if bad:
        print(f"\n{len(bad)} discrepancy(ies): {', '.join(bad)}. "
              "Reported, not silently accepted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
