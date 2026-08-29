#!/usr/bin/env python3
"""
klebphacol_stage6b_bootstrap_ci.py — bootstrap 95% CIs on the near-identical
and related strata (all 3 media, both host scopes), PHL-RBP+S full RBP set.

Reported as a two-sided test against 0.5: if the CI excludes 0.5 entirely
from BELOW, the model is anti-predictive there (its ranking is significantly
worse than random, not merely uninformative) -- stated explicitly, not
folded into "not significant."
"""
import os
import pickle
import numpy as np
import pandas as pd

import klebphacol_stage4_train_predict as s4

OUT_DIR = "results/klebphacol"
INTERACTION_FILES = {
    "LB strict": f"{OUT_DIR}/interactions_LB_strict.csv",
    "LB permissive": f"{OUT_DIR}/interactions_LB_permissive.csv",
    "TSB strict": f"{OUT_DIR}/interactions_TSB_strict.csv",
}


def verdict(ci, point):
    lo, hi = ci
    if hi < 0.5:
        return "ANTI-PREDICTIVE (CI entirely below 0.5)"
    if lo > 0.5:
        return "predictive (CI entirely above 0.5)"
    return "not distinguishable from 0.5"


def main():
    with open(s4.MODEL_PATH, "rb") as fh:
        saved = pickle.load(fh)
    model, sero_cols = saved["model"], saved["sero_cols"]
    sero_encoded, sero_cols = s4.build_training_serotype_vocab()
    hosts, host_S, n_absent = s4.build_host_sero_matrix(sero_cols)
    unseen_kl_hosts = set(hosts[hosts.sero_col.isna()].strain)

    rbp_df = pd.read_csv(f"{OUT_DIR}/stage3_rbps_tagged.csv")
    rbp_emb = np.load(f"{OUT_DIR}/stage3_rbp_embeddings.npy")
    rbp_gene_idx = {g: i for i, g in enumerate(rbp_df.gene_ID)}
    phage_stratum = (rbp_df.assign(risk=rbp_df.stratum.map(
        {"novel": 0, "related": 1, "near-identical": 2}))
        .groupby("phage_ID")["risk"].max()
        .map({0: "novel", 1: "related", 2: "near-identical"}))

    pred = s4.predict_klebphacol(model, sero_cols, hosts, host_S, rbp_df, rbp_emb, rbp_gene_idx)

    print("=" * 90)
    print("BOOTSTRAP 95% CI, near-identical and related strata, PHL-RBP+S full RBP set")
    print("Two-sided test against 0.5 (chance)")
    print("=" * 90)

    for medium, path in INTERACTION_FILES.items():
        inter = pd.read_csv(path)
        merged = inter.merge(pred, on=["phage", "strain"], how="left").dropna(subset=["score"])
        merged["stratum"] = merged.phage.map(phage_stratum)

        print(f"\n--- {medium} ---")
        for scope, mask_fn in (
            ("incl_unseen_kl", lambda m: pd.Series(True, index=m.index)),
            ("excl_unseen_kl", lambda m: ~m.strain.isin(unseen_kl_hosts)),
        ):
            m = merged[mask_fn(merged)]
            for s in ("related", "near-identical"):
                sub = m[m.stratum == s]
                if len(sub) < 2 or len(set(sub.label)) < 2:
                    print(f"  [{scope}] {s:<16} n={len(sub):<6} insufficient for CI")
                    continue
                from sklearn.metrics import roc_auc_score
                point = roc_auc_score(sub.label.values, sub.score.values)
                ci = s4.bootstrap_ci(sub.label.values, sub.score.values)
                lo, hi = ci["roc_auc_ci"]
                v = verdict((lo, hi), point)
                print(f"  [{scope}] {s:<16} n={len(sub):<6} n_pos={int(sub.label.sum()):<5} "
                      f"ROC-AUC={point:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  -> {v}")


if __name__ == "__main__":
    main()
