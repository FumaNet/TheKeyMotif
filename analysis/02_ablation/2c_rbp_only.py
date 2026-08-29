"""
2c_rbp_only.py — the RBP-only baseline (Reviewer #1, major comment 2).

The mirror image of 2b_serotype_only.py. That one gave the model the host and
no phage; this one gives it the phage and no host. Features are the RBP
embedding alone -- no serotype, no capsule proteins, nothing identifying which
bacterium the pair involves.

With no host features, every host gets the same score for a given RBP, so this
measures only how much signal comes from "some phages are generally more
infectious than others" (phage promiscuity).

Read the three together:

    serotype-only   ~0.52   host alone carries almost nothing
    RBP-only        this    phage alone carries ...?
    PHL-RBP+S       0.817   both together

If both single-sided baselines sit near chance while the combination reaches
0.817, the model is learning a genuine RBP-by-serotype MATCHING function rather
than either marginal. That is the argument Reviewer #1 is asking for, and it is
much stronger stated as a bracket than as a single ablation.

ROW WEIGHTING -- deliberate difference from script 3
----------------------------------------------------
Script 3 trains on the full 487k-row pair table, because its drop_duplicates()
is a no-op (the host protein_index column survives). Those duplicate rows
upweight hosts with large K-loci. That artifact is worth reproducing when
reproducing published numbers; it is not worth propagating into a new baseline.

This script therefore deduplicates to unique (accession, phage_ID, protein_ID)
rows -- 25,120 of them -- which is the natural unit when no host features exist.
Stated explicitly so the difference is not mistaken for an error.

    python 2c_rbp_only.py

Writes results/2c_AUCs_rbp_only.pkl in the standard format.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
from xgboost import XGBClassifier

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import keymotif_data as kd

DEVICE = os.environ.get("KM_DEVICE", "cuda")

THRESHOLDS = [1.0, 0.995, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75]
TSTR = ["100", "99.5", "99", "95", "90", "85", "80", "75"]
GROUPING_FILES = [
    "grouping/grouping_1.pkl", "grouping/grouping_995.pkl",
    "grouping/grouping_990.pkl", "grouping/grouping_950.pkl",
    "grouping/grouping_900.pkl", "grouping/grouping_850.pkl",
    "grouping/grouping_800.pkl", "grouping/grouping_750.pkl",
]
OUT = "results/2c_AUCs_rbp_only.pkl"
CKPT = "results/2c_checkpoint.pkl"
ONLY = os.environ.get("KM_THRESHOLDS")


def main():
    pairs, host_emb, virus_emb = kd.load()

    rbp_level = pairs.drop_duplicates(
        subset=["accession", "phage_ID", "protein_ID", "virus_idx"]
    ).reset_index(drop=True)
    print(f"RBP-level table: {len(rbp_level):,} rows "
          f"(deduplicated from {len(pairs):,}; see docstring)")
    print("Features: RBP embedding ONLY — no host information of any kind.")

    os.makedirs("results", exist_ok=True)
    done = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}
    if done:
        print(f"Checkpoint found — already complete: {sorted(done)}")
    wanted = {t.strip() for t in ONLY.split(",")} if ONLY else set(TSTR)

    for i, _threshold in enumerate(THRESHOLDS):
        if TSTR[i] in done or TSTR[i] not in wanted:
            print(f"Skipping {TSTR[i]}%")
            continue

        fold_df = kd.attach_groups(rbp_level, GROUPING_FILES[i])

        logo = LeaveOneGroupOut()
        scores_max, label_max = [], []
        n_groups = fold_df["group_loci"].nunique()
        pbar = tqdm(total=n_groups, desc=f"RBP-only @ {TSTR[i]}%")

        for tr, te in logo.split(fold_df, fold_df["label"], fold_df["group_loci"]):
            sub_tr, sub_te = fold_df.iloc[tr], fold_df.iloc[te]

            X_train = kd.make_X(sub_tr, host_emb, virus_emb, mode="virus")
            X_test = kd.make_X(sub_te, host_emb, virus_emb, mode="virus")
            y_train = sub_tr["label"].astype(int).values
            y_test = sub_te["label"].astype(int).values

            if len(set(y_train)) < 2:
                pbar.update(1)
                continue

            n_pos = int((y_train == 1).sum())
            n_neg = int((y_train == 0).sum())
            imbalance = n_pos / n_neg if n_neg else 1

            xgb = XGBClassifier(
                scale_pos_weight=1 / imbalance,
                learning_rate=0.3, n_estimators=250, max_depth=7,
                eval_metric="logloss", tree_method="hist", device=DEVICE,
                random_state=0,
            )
            xgb.fit(X_train, y_train)
            score = xgb.predict_proba(X_test)[:, 1]

            df_preds = pd.DataFrame({
                "accession": sub_te["accession"].values,
                "phage_ID": sub_te["phage_ID"].values,
                "true_label": y_test, "score": score,
            })
            mx = df_preds.groupby(["accession", "phage_ID"]).agg(
                {"score": "max", "true_label": "first"}).reset_index()

            scores_max.append(mx["score"].values)
            label_max.append(mx["true_label"].values)
            pbar.update(1)

        pbar.close()
        if not scores_max:
            print(f"No usable folds at {TSTR[i]}%.")
            continue

        scores_max = np.concatenate(scores_max)
        label_max = np.concatenate(label_max)

        if len(set(label_max)) > 1:
            fpr, tpr, _ = roc_curve(label_max, scores_max)
            prec, rec, _ = precision_recall_curve(label_max, scores_max)
            rauclr = round(auc(fpr, tpr), 3)
            print(f"RBP-only AUC: {rauclr}   PR-AUC: {auc(rec, prec):.3f}")
            done[TSTR[i]] = (label_max, scores_max, rauclr)
            with open(CKPT, "wb") as fh:
                pickle.dump(done, fh)
        else:
            print(f"Evaluation failed at {TSTR[i]}% (single-class).")

    missing = [t for t in TSTR if t not in done]
    if missing:
        print(f"\nCheckpoint holds {len(done)}/{len(TSTR)}: "
              f"{[t for t in TSTR if t in done]}")
        print(f"Still missing: {missing}")
        print(f"{OUT} NOT written (a partial file would misalign thresholds).")
        print(f'Finish with:  $env:KM_THRESHOLDS="{",".join(missing)}"')
        return

    predsss = [done[t] for t in TSTR]
    with open(OUT, "wb") as fh:
        pickle.dump(predsss, fh)
    print(f"\nWrote {OUT} — all 8 thresholds: {[d[2] for d in predsss]}")
    print("\nThe bracket:")
    print("  serotype-only [0.589, 0.554, 0.536, 0.487, 0.485, 0.484, 0.510, 0.530]")
    print(f"  RBP-only      {[d[2] for d in predsss]}")
    print("  PHL-RBP+S     [0.817, 0.747, 0.690, 0.644, 0.636, 0.615, 0.659, 0.672]")


if __name__ == "__main__":
    main()
