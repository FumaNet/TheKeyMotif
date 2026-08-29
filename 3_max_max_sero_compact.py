"""
3_max_max_sero_compact.py — PHL-RBP+S, memory-safe.

Drop-in replacement for 3_max_max_sero.py: individual viral RBP embeddings
paired with a one-hot capsular serotype encoding, max-aggregated per
(host, phage) pair.

Run from the repo root, with keymotif_data.py alongside it:

    python 3_max_max_sero_compact.py

Writes Results/3_AUCs_max_max_sero.pkl in the original format.

Verified bit-identical to the original by test_equivalence.py.

NOTE: the original calls df.drop_duplicates() after dropping the host_* columns
and describes it as redundant with the averaging. It is a no-op -- the host
`protein_index` column survives, so no rows are removed. This file reproduces
that behaviour deliberately (it trains on the full pair table). Do not "fix" it
here; that would change the published numbers. See SETUP_LOCAL.md section 4.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
from xgboost import XGBClassifier

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
OUT = "Results/3_AUCs_max_max_sero.pkl"
CKPT = "Results/3_checkpoint.pkl"
ONLY = os.environ.get("KM_THRESHOLDS")


def main():
    pairs, host_emb, virus_emb = kd.load()

    # --- serotype one-hot, built once (it does not depend on the fold) ---
    df_sero = pd.read_csv("Data/kaptive_results.tsv", sep="\t")
    df_sero = df_sero[["Assembly", "Best match type", "Match confidence"]]
    one_hot = pd.get_dummies(df_sero["Best match type"], prefix="sero_")
    sero_encoded = pd.concat([df_sero[["Assembly"]], one_hot], axis=1)
    sero_cols = [c for c in sero_encoded.columns if c != "Assembly"]
    print(f"Serotype encoding: {len(sero_cols)} columns")

    os.makedirs("Results", exist_ok=True)
    done = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}
    if done:
        print(f"Checkpoint found — already complete: {sorted(done)}")
    wanted = {t.strip() for t in ONLY.split(",")} if ONLY else set(TSTR)

    for i, _threshold in enumerate(THRESHOLDS):
        if TSTR[i] in done or TSTR[i] not in wanted:
            print(f"Skipping {TSTR[i]}%")
            continue
        fold_pairs = kd.attach_groups(pairs, GROUPING_FILES[i])

        logo = LeaveOneGroupOut()
        scores_max, label_max = [], []

        n_groups = fold_pairs["group_loci"].nunique()
        pbar = tqdm(total=n_groups, desc=f"LOGO CV @ {TSTR[i]}%")

        for train_index, test_index in logo.split(
                fold_pairs, fold_pairs["label"], fold_pairs["group_loci"]):

            sub_tr, S_tr = kd.attach_serotype(
                fold_pairs.iloc[train_index], sero_encoded, sero_cols)
            sub_te, S_te = kd.attach_serotype(
                fold_pairs.iloc[test_index], sero_encoded, sero_cols)

            # host_* columns dropped; features are [virus embedding | serotype]
            X_train = kd.make_X(sub_tr, host_emb, virus_emb,
                                mode="virus", sero=S_tr)
            X_test = kd.make_X(sub_te, host_emb, virus_emb,
                               mode="virus", sero=S_te)

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
                "accession": sub_te["accession"].values,
                "phage_ID": sub_te["phage_ID"].values,
                "true_label": y_test,
                "score": score_xgb,
            })
            # max across all RBPs (and host-protein duplicates) for the pair
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
            print(f"Final AUC with max protein-serotype scoring: {rauclr}")
            done[TSTR[i]] = (label_max, scores_max, rauclr)
            with open(CKPT, "wb") as fh:
                pickle.dump(done, fh)
        else:
            print(f"Final evaluation failed at {TSTR[i]}% "
                  "threshold due to single-class predictions.")

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


if __name__ == "__main__":
    main()
