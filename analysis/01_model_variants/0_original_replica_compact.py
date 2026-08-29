"""
0_original_replica_compact.py — PHL-AVG, memory-safe.

Drop-in replacement for 0_original_replica.py. Same model, same LOGO protocol,
same output file; it just never builds the 487,400 x 2,560 float64 frame that
the original constructs during data prep.

Run from the repo root, with keymotif_data.py alongside it:

    python 0_original_replica_compact.py

Writes results/0_AUCs_original_replica.pkl in the original format:
a list of (labels, scores, rounded_roc_auc) tuples, one per threshold.

Verified bit-identical to the original by test_equivalence.py.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
from xgboost import XGBClassifier

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import keymotif_data as kd

DEVICE = os.environ.get("KM_DEVICE", "cuda")   # set KM_DEVICE=cpu to fall back

THRESHOLDS = [1.0, 0.995, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75]
TSTR = ["100", "99.5", "99", "95", "90", "85", "80", "75"]
GROUPING_FILES = [
    "grouping/grouping_1.pkl",
    "grouping/grouping_995.pkl",
    "grouping/grouping_990.pkl",
    "grouping/grouping_950.pkl",
    "grouping/grouping_900.pkl",
    "grouping/grouping_850.pkl",
    "grouping/grouping_800.pkl",
    "grouping/grouping_750.pkl",
]
OUT = "results/0_AUCs_original_replica.pkl"
CKPT = "results/0_checkpoint.pkl"
ONLY = os.environ.get("KM_THRESHOLDS")   # e.g. "95,90,85,80,75"


def main():
    # --- data -----------------------------------------------------------
    # Replaces the whole "if not os.path.isfile(...combined_embeddings...)"
    # block. `pairs` is the row index; the embeddings stay in two small arrays.
    pairs, host_emb, virus_emb = kd.load()

    os.makedirs("results", exist_ok=True)
    done = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}
    if done:
        print(f"Checkpoint found — already complete: {sorted(done)}")
    wanted = {t.strip() for t in ONLY.split(",")} if ONLY else set(TSTR)

    for i, _threshold in enumerate(THRESHOLDS):
        if TSTR[i] in done or TSTR[i] not in wanted:
            print(f"Skipping {TSTR[i]}%")
            continue
        # Validates the accession -> group mapping instead of silently
        # producing NaN, which is what a bare .map() does.
        fold_pairs = kd.attach_groups(pairs, GROUPING_FILES[i])

        # Collapse ONCE for this threshold. Each (accession, phage_ID) group
        # lies entirely inside one LOGO fold, so the group means are
        # fold-independent -- computing them here and slicing per fold is
        # bit-identical to recomputing them in every fold, and ~185x cheaper.
        print(f"Precomputing averaged representation for {TSTR[i]}% ...")
        kd.precompute_collapsed(fold_pairs, host_emb, virus_emb)

        logo = LeaveOneGroupOut()
        scores_max, label_max = [], []

        n_groups = fold_pairs["group_loci"].nunique()
        pbar = tqdm(total=n_groups, desc=f"LOGO CV @ {TSTR[i]}%")

        for train_index, test_index in logo.split(
                fold_pairs, fold_pairs["label"], fold_pairs["group_loci"]):

            # In-fold averaging, done exactly as pandas would have.
            meta_tr, X_train = kd.collapse_averaged_exact(
                fold_pairs.iloc[train_index], host_emb, virus_emb)
            meta_te, X_test = kd.collapse_averaged_exact(
                fold_pairs.iloc[test_index], host_emb, virus_emb)

            y_train = meta_tr["label"].astype(int).values
            y_test = meta_te["label"].astype(int).values

            if len(set(y_train)) < 2:
                pbar.update(1)
                continue

            n_pos = int((y_train == 1).sum())
            n_neg = int((y_train == 0).sum())
            imbalance = n_pos / n_neg if n_neg else 1

            xgb = XGBClassifier(
                scale_pos_weight=1 / imbalance,
                learning_rate=0.3,
                n_estimators=250,
                max_depth=7,
                eval_metric="logloss",
                tree_method="hist",
                device=DEVICE,
                random_state=0,
            )
            xgb.fit(X_train, y_train)
            score_xgb = xgb.predict_proba(X_test)[:, 1]

            df_preds = pd.DataFrame({
                "accession": meta_te["accession"].values,
                "phage_ID": meta_te["phage_ID"].values,
                "true_label": y_test,
                "score": score_xgb,
            })
            max_scores = df_preds.groupby(["accession", "phage_ID"]).agg(
                {"score": "max", "true_label": "first"}).reset_index()

            scores_max.append(max_scores["score"].values)
            label_max.append(max_scores["true_label"].values)
            pbar.update(1)

        pbar.close()

        if not scores_max:
            print(f"No usable folds at {TSTR[i]}%.")
            continue

        scores_max = np.concatenate(scores_max)
        label_max = np.concatenate(label_max)

        if len(set(label_max)) > 1:
            fpr, tpr, _ = roc_curve(label_max, scores_max)
            rauclr = round(auc(fpr, tpr), 3)
            print(f"Final AUC after host-averaged adjustment: {rauclr}")
            done[TSTR[i]] = (label_max, scores_max, rauclr)
            with open(CKPT, "wb") as fh:
                pickle.dump(done, fh)
        else:
            print(f"Final evaluation failed at {TSTR[i]}% "
                  "threshold due to single-class predictions.")

    missing = [t for t in TSTR if t not in done]
    if missing:
        print(f"\nCheckpoint holds {len(done)}/{len(TSTR)} thresholds: "
              f"{[t for t in TSTR if t in done]}")
        print(f"Still missing: {missing}")
        print(f"{OUT} NOT written — a partial file would misalign every "
              f"threshold against the published values.")
        print(f'Finish with:  $env:KM_THRESHOLDS="{",".join(missing)}"')
        return

    predsss = [done[t] for t in TSTR]
    with open(OUT, "wb") as fh:
        pickle.dump(predsss, fh)
    print(f"\nWrote {OUT} — all {len(TSTR)} thresholds: {[d[2] for d in predsss]}")


if __name__ == "__main__":
    main()
