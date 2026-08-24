#!/usr/bin/env python3
"""klebphacol_stage4_run.py — Stage 4 driver: train, predict, evaluate,
sensitivity check, guards. See klebphacol_stage4_train_predict.py for the
module docstring covering the modelling/crosswalk decisions.

Every headline metric is reported twice, incl_unseen_kl and excl_unseen_kl
(with/without the 16 KlebPhaCol hosts whose KL type has no training one-hot
column) -- see evaluate()'s docstring for why both are legitimate and
conflating them understates the model. Strata come from the Stage 3b
correction (restratified against the 250 post-exclusion training RBPs, not
all 274) -- run klebphacol_stage3b_restratify.py first if
Results/klebphacol/stage3_rbps_tagged.csv predates that correction.
"""
import os
import pickle
import numpy as np
import pandas as pd

import klebphacol_stage4_train_predict as s4

OUT_DIR = "Results/klebphacol"
N_SUBSAMPLE = round(274 / 105)  # = 3

INTERACTION_FILES = {
    "LB strict": os.path.join(OUT_DIR, "interactions_LB_strict.csv"),
    "LB permissive": os.path.join(OUT_DIR, "interactions_LB_permissive.csv"),
    "TSB strict": os.path.join(OUT_DIR, "interactions_TSB_strict.csv"),
}


def print_metrics_table(results_by_scope, label):
    print(f"\n  [{label}]")
    for scope, title in (("incl_unseen_kl", "including 16 unseen-KL hosts (deployment-realistic)"),
                          ("excl_unseen_kl", "excluding 16 unseen-KL hosts (model-as-designed)")):
        results = results_by_scope[scope]
        print(f"    -- {title} --")
        print(f"    {'stratum':<16}{'n':>8}{'n_pos':>8}{'base_rate':>11}{'ROC-AUC':>10}{'PR-AUC':>10}")
        for s in ("overall", "novel", "related", "near-identical"):
            if s not in results:
                continue
            m = results[s]
            print(f"    {s:<16}{m['n']:>8}{m['n_pos']:>8}{100*m['base_rate']:>10.1f}%"
                  f"{m['roc_auc']:>10.3f}{m['pr_auc']:>10.3f}")


def main():
    # --- train (or reuse cached model) ---
    if os.path.exists(s4.MODEL_PATH):
        with open(s4.MODEL_PATH, "rb") as fh:
            saved = pickle.load(fh)
        model, sero_cols = saved["model"], saved["sero_cols"]
        print(f"Loaded cached model from {s4.MODEL_PATH}")
        pairs, _, _ = s4.kd.load()
        excl = pd.read_csv(s4.EXCLUSIONS)
        excluded_phages = sorted(excl.boeckaerts_phage_to_exclude.unique())
        remaining = set(pairs.phage_ID.unique()) - set(excluded_phages)
        assert len(remaining) == 105 - len(excluded_phages), "cached model guard check failed"
        print(f"GUARD 1 (re-checked against cache): "
              f"{105 - len(excluded_phages)} Boeckaerts phages expected in training.")
    else:
        model, sero_cols = s4.train_model()

    # --- host serotype crosswalk ---
    hosts, host_S, n_absent = s4.build_host_sero_matrix(sero_cols)
    unseen_kl_hosts = set(hosts[hosts.sero_col.isna()].strain)

    # --- RBP table + embeddings (restratified, post-exclusion) ---
    rbp_df = pd.read_csv(s4.RBP_TAGGED)
    rbp_emb = np.load(s4.RBP_EMB)
    rbp_gene_idx = {g: i for i, g in enumerate(rbp_df.gene_ID)}
    phage_stratum = (rbp_df.assign(risk=rbp_df.stratum.map(
        {"novel": 0, "related": 1, "near-identical": 2}))
        .groupby("phage_ID")["risk"].max()
        .map({0: "novel", 1: "related", 2: "near-identical"}))
    print(f"\nPhage-level stratum (post-exclusion, restratified): "
          f"{(phage_stratum=='novel').sum()} novel / "
          f"{(phage_stratum=='related').sum()} related / "
          f"{(phage_stratum=='near-identical').sum()} near-identical")

    print("\n" + "=" * 66)
    print("PREDICTION: full RBP set (193 RBPs)")
    print("=" * 66)
    pred_full = s4.predict_klebphacol(model, sero_cols, hosts, host_S,
                                       rbp_df, rbp_emb, rbp_gene_idx)
    print(f"Scored {len(pred_full)} (phage, host) combinations")

    print("\n" + "=" * 66)
    print(f"SENSITIVITY: subsampled RBP set (top-{N_SUBSAMPLE}/phage by scores_RBPDetect)")
    print("=" * 66)
    per_phage_before = rbp_df.groupby("phage_ID").size()
    n_affected = (per_phage_before > N_SUBSAMPLE).sum()
    print(f"{n_affected}/52 phages have >{N_SUBSAMPLE} RBPs and get subsampled "
          f"(N={N_SUBSAMPLE} = round(274/105={274/105:.3f}))")
    pred_sub = s4.predict_klebphacol(model, sero_cols, hosts, host_S,
                                      rbp_df, rbp_emb, rbp_gene_idx,
                                      top_n_per_phage=N_SUBSAMPLE)

    all_results = {}
    for label, path in INTERACTION_FILES.items():
        print("\n" + "=" * 66)
        print(f"RESULTS: {label}")
        print("=" * 66)
        res_full, merged_full = s4.evaluate(pred_full, path, phage_stratum, unseen_kl_hosts)
        res_sub, merged_sub = s4.evaluate(pred_sub, path, phage_stratum, unseen_kl_hosts)
        all_results[label] = dict(full=res_full, sub=res_sub, merged_full=merged_full)

        print_metrics_table(res_full, "full RBP set")
        print_metrics_table(res_sub, f"subsampled top-{N_SUBSAMPLE}/phage")

        print("\n  Sensitivity delta (full -> subsampled), novel stratum, "
              "excl_unseen_kl scope (the honest external number):")
        rf = res_full["excl_unseen_kl"]
        rs = res_sub["excl_unseen_kl"]
        if "novel" in rf and "novel" in rs:
            d_roc = rs["novel"]["roc_auc"] - rf["novel"]["roc_auc"]
            d_pr = rs["novel"]["pr_auc"] - rf["novel"]["pr_auc"]
            print(f"    ROC-AUC: {rf['novel']['roc_auc']:.3f} -> "
                  f"{rs['novel']['roc_auc']:.3f}  (delta {d_roc:+.3f})")
            print(f"    PR-AUC:  {rf['novel']['pr_auc']:.3f} -> "
                  f"{rs['novel']['pr_auc']:.3f}  (delta {d_pr:+.3f})")
            moved_little = abs(d_roc) < 0.02 and abs(d_pr) < 0.02
            print(f"    -> {'metrics barely move; the 1.42x RBP-count inflation is NOT driving results' if moved_little else 'metrics move materially; report the subsampled version as primary'}")

    # --- bootstrap CI on novel stratum, LB strict and LB permissive (full RBP set) ---
    print("\n" + "=" * 66)
    print(f"BOOTSTRAP 95% CI, novel stratum, full RBP set (n_boot={s4.N_BOOT})")
    print("=" * 66)
    for label in ("LB strict", "LB permissive"):
        merged = all_results[label]["merged_full"]
        for scope, title in (("incl_unseen_kl", "incl 16 unseen-KL hosts"),
                              ("excl_unseen_kl", "excl 16 unseen-KL hosts")):
            m = merged[(merged.stratum == "novel") & merged.score.notna()]
            if scope == "excl_unseen_kl":
                m = m[~m.strain.isin(unseen_kl_hosts)]
            ci = s4.bootstrap_ci(m.label.values, m.score.values)
            print(f"\n  [{label}, {title}] novel stratum: n={len(m)}, n_pos={int(m.label.sum())}")
            print(f"    ROC-AUC 95% CI: [{ci['roc_auc_ci'][0]:.3f}, {ci['roc_auc_ci'][1]:.3f}]")
            print(f"    PR-AUC  95% CI: [{ci['pr_auc_ci'][0]:.3f}, {ci['pr_auc_ci'][1]:.3f}]"
                  f"  ({ci['n_valid_boots']}/{s4.N_BOOT} valid resamples)")

    print("\n" + "=" * 66)
    print("GUARD SUMMARY")
    print("=" * 66)
    print(f"GUARD 1: PASSED (see training log above)")
    print(f"GUARD 2: {n_absent}/{len(hosts)} KlebPhaCol hosts have a KL type absent "
          f"from the training serotype vocabulary (all-zero vector).")
    print(f"  Handled via incl/excl_unseen_kl reporting throughout above, "
          f"not just a separate note. For reference, those 16 hosts ALONE:")
    for label, path in INTERACTION_FILES.items():
        inter = pd.read_csv(path)
        sub = inter[inter.strain.isin(unseen_kl_hosts)]
        merged = sub.merge(pred_full, on=["phage", "strain"], how="left").dropna(subset=["score"])
        if len(merged) and len(set(merged.label)) > 1:
            m = s4.compute_metrics(merged.label.values, merged.score.values)
            print(f"    [{label}] unseen-KL hosts only: n={m['n']} "
                  f"n_pos={m['n_pos']} ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f}")
        else:
            print(f"    [{label}] unseen-KL hosts only: n={len(merged)}, "
                  f"insufficient class balance for AUC")


if __name__ == "__main__":
    main()
