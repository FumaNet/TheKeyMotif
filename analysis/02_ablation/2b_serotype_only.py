"""
2b_serotype_only.py — the serotype-only baseline (Reviewer #1, major comment 2).

Reviewer #1 asked: does PHL-RBP+S actually learn transferable RBP features, or
is it largely memorising the capsule serotype, which is the dominant host
determinant in Klebsiella?

This trains a model with NO viral information whatsoever: the only input is the
host's one-hot capsular serotype. It cannot distinguish phages at all -- every
phage gets the same score against a given serotype. So whatever AUC it reaches
is the share attributable to serotype prevalence alone.

Read it as a floor:

  serotype-only ~= PHL-RBP+S   -> the viral side is contributing little, and the
                                  paper's framing needs to change
  serotype-only << PHL-RBP+S   -> the RBP embedding is doing real work, and the
                                  memorisation concern is answered with data

Same LOGO protocol, same XGBoost hyperparameters, same max-aggregation and
metric as every other script, so the numbers are directly comparable.

One row per (accession, phage_ID) pair: with no viral features there is nothing
to aggregate over, so the 487k pair table collapses to ~10k rows and this runs
in minutes rather than hours.

    python 2b_serotype_only.py
    $env:KM_THRESHOLDS="100,90"; python 2b_serotype_only.py

Writes Results/2b_AUCs_serotype_only.pkl in the standard format.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
from xgboost import XGBClassifier

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
OUT = "Results/2b_AUCs_serotype_only.pkl"
CKPT = "Results/2b_checkpoint.pkl"
ONLY = os.environ.get("KM_THRESHOLDS")


def main():
    pairs, host_emb, virus_emb = kd.load()

    # One row per (accession, phage_ID): no viral features means the
    # per-protein expansion carries no information.
    interactions = (pairs.groupby(["accession", "phage_ID"], sort=True)
                    .agg(label=("label", "first")).reset_index())
    print(f"Interaction-level table: {len(interactions):,} pairs "
          f"({100 * interactions['label'].mean():.2f}% positive)")

    df_sero = pd.read_csv("Data/kaptive_results.tsv", sep="\t")
    df_sero = df_sero[["Assembly", "Best match type", "Match confidence"]]
    one_hot = pd.get_dummies(df_sero["Best match type"], prefix="sero_")
    sero_encoded = pd.concat([df_sero[["Assembly"]], one_hot], axis=1)
    sero_cols = [c for c in sero_encoded.columns if c != "Assembly"]
    print(f"Serotype encoding: {len(sero_cols)} columns "
          f"(the ONLY features this model sees)")

    os.makedirs("Results", exist_ok=True)
    done = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}
    if done:
        print(f"Checkpoint found — already complete: {sorted(done)}")
    wanted = {t.strip() for t in ONLY.split(",")} if ONLY else set(TSTR)

    for i, _threshold in enumerate(THRESHOLDS):
        if TSTR[i] in done or TSTR[i] not in wanted:
            print(f"Skipping {TSTR[i]}%")
            continue

        fold_df = kd.attach_groups(interactions, GROUPING_FILES[i])

        logo = LeaveOneGroupOut()
        scores_max, label_max = [], []
        n_groups = fold_df["group_loci"].nunique()
        pbar = tqdm(total=n_groups, desc=f"serotype-only @ {TSTR[i]}%")

        for train_index, test_index in logo.split(
                fold_df, fold_df["label"], fold_df["group_loci"]):

            sub_tr, S_tr = kd.attach_serotype(
                fold_df.iloc[train_index], sero_encoded, sero_cols)
            sub_te, S_te = kd.attach_serotype(
                fold_df.iloc[test_index], sero_encoded, sero_cols)

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
            xgb.fit(S_tr, y_train)
            score = xgb.predict_proba(S_te)[:, 1]

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
            print(f"Serotype-only AUC: {rauclr}   PR-AUC: {auc(rec, prec):.3f}")
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
    print("\nCompare against published PHL-RBP+S:")
    print("  [0.817, 0.747, 0.690, 0.644, 0.636, 0.615, 0.659, 0.672]")
    print("If these are close, the viral side is contributing little.")


if __name__ == "__main__":
    main()
