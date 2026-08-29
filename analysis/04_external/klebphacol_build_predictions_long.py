#!/usr/bin/env python3
"""
klebphacol_build_predictions_long.py — builds the long-format predictions
table s4_appendix_build.py expects (one row per scored (phage, host, medium,
mapping) pair), from PHL-RBP+S's already-established Stage 4 pipeline.

PHL-RBP+S's score for a (phage, host) pair doesn't depend on medium/mapping
(those only change which ground-truth label a pair carries) -- computed
once via the cached model, then stacked against all four interaction
tables (LB/TSB x strict/permissive) with their own y_true, stratum
(phage-level, Stage 3b's post-exclusion-corrected tagging), and host_seen
(host's KL type present in the training serotype vocabulary).
"""
import os
import pickle
import numpy as np
import pandas as pd

import klebphacol_stage4_train_predict as s4

OUT_DIR = "results/klebphacol"
OUT_PATH = "klebphacol_predictions.csv"
INTERACTION_FILES = {
    ("LB", "strict"): f"{OUT_DIR}/interactions_LB_strict.csv",
    ("LB", "permissive"): f"{OUT_DIR}/interactions_LB_permissive.csv",
    ("TSB", "strict"): f"{OUT_DIR}/interactions_TSB_strict.csv",
    ("TSB", "permissive"): f"{OUT_DIR}/interactions_TSB_permissive.csv",
}


def main():
    with open(s4.MODEL_PATH, "rb") as fh:
        saved = pickle.load(fh)
    model, sero_cols = saved["model"], saved["sero_cols"]
    sero_encoded, sero_cols = s4.build_training_serotype_vocab()
    hosts, host_S, n_absent = s4.build_host_sero_matrix(sero_cols)
    seen_hosts = set(hosts[hosts.sero_col.notna()].strain)

    rbp_df = pd.read_csv(f"{OUT_DIR}/stage3_rbps_tagged.csv")
    rbp_emb = np.load(f"{OUT_DIR}/stage3_rbp_embeddings.npy")
    rbp_gene_idx = {g: i for i, g in enumerate(rbp_df.gene_ID)}
    phage_stratum = (rbp_df.assign(risk=rbp_df.stratum.map(
        {"novel": 0, "related": 1, "near-identical": 2}))
        .groupby("phage_ID")["risk"].max()
        .map({0: "novel", 1: "related", 2: "near-identical"}))

    pred = s4.predict_klebphacol(model, sero_cols, hosts, host_S, rbp_df, rbp_emb, rbp_gene_idx)
    print(f"Base predictions: {len(pred)} (phage, host) scores "
          f"({pred.phage.nunique()} phages x {pred.strain.nunique()} hosts)")

    rows = []
    for (medium, mapping), path in INTERACTION_FILES.items():
        inter = pd.read_csv(path)
        merged = inter.merge(pred, on=["phage", "strain"], how="left").dropna(subset=["score"])
        merged["medium"] = medium
        merged["mapping"] = mapping
        merged["stratum"] = merged.phage.map(phage_stratum)
        merged["host_seen"] = merged.strain.isin(seen_hosts)
        merged = merged.rename(columns={"label": "y_true"})
        n_missing = len(inter) - len(merged)
        print(f"  {medium}/{mapping}: {len(merged)} pairs scored"
              f"{f' ({n_missing} unscored, dropped)' if n_missing else ''}")
        rows.append(merged[["medium", "mapping", "stratum", "y_true", "score",
                             "host_seen", "phage"]])

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(out)} rows)")
    print(out.groupby(["medium", "mapping"]).size())


if __name__ == "__main__":
    main()
