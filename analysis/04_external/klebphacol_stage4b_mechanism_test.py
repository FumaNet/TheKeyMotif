#!/usr/bin/env python3
"""
klebphacol_stage4b_mechanism_test.py — mechanism test for the near-identical
underperforms-novel inversion (survived Stage 3b's restratification fix, so
it's a real finding, not a stratum-labelling artifact -- see that script's
docstring).

HYPOTHESIS (stated before running): the model memorised RBP-to-serotype
pairings rather than a transferable RBP-binding rule. When a near-identical
KlebPhaCol RBP reappears in KlebPhaCol with a host serotype DIFFERENT from
whatever serotype(s) its matched training RBP was positively paired with,
memorisation actively misleads the model -- hence below-chance performance
concentrated in that split.

TEST: for each near-identical (>=95%) KlebPhaCol RBP, its best-hit training
RBP belongs to one training phage; that phage's POSITIVE training
interactions give a set of host serotypes (via kaptive_results.tsv's "Best
match type" -- the same vocabulary Stage 4's model was trained on, and the
same one KlebPhaCol hosts were cross-walked into). A (host, phage)
evaluation pair is MATCHED if the querying KlebPhaCol host's own
crosswalked serotype is in the union of those sets across all of that
phage's near-identical RBPs; MISMATCHED otherwise. Hosts with no valid
crosswalked serotype at all (the 16 unseen-KL-vocabulary hosts) can never
match anything by construction, so the primary test excludes them (their
presence would conflate two different confounds inside "MISMATCHED"); the
same split including them is reported separately for transparency.

PREDICTION (stated before running, not adjusted after): MATCHED at or above
novel's 0.649 ROC-AUC; MISMATCHED carrying the below-chance signal. If both
land near 0.426 instead, the memorisation hypothesis is refuted.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import keymotif_data as kd
import klebphacol_stage4_train_predict as s4

OUT_DIR = "Results/klebphacol"


def training_positive_serotypes_by_phage():
    """phage_ID -> set of 'Best match type' serotypes among its POSITIVE
    training interactions (host-level label; RBP-level attribution isn't in
    the ground truth, so this is the serotype set the phage-as-a-whole was
    positively associated with in training)."""
    pairs, _, _ = kd.load()
    kaptive = pd.read_csv("Data/kaptive_results.tsv", sep="\t")[["Assembly", "Best match type"]]
    acc_to_type = dict(zip(kaptive.Assembly, kaptive["Best match type"]))

    pos = pairs[pairs.label == 1][["phage_ID", "accession"]].drop_duplicates()
    pos["serotype"] = pos.accession.map(acc_to_type)
    out = pos.dropna(subset=["serotype"]).groupby("phage_ID")["serotype"].apply(set).to_dict()
    return out


def compute_metrics(labels, scores):
    if len(set(labels)) < 2 or len(labels) == 0:
        return dict(roc_auc=float("nan"), pr_auc=float("nan"),
                    n=len(labels), n_pos=int(sum(labels)) if len(labels) else 0)
    return dict(roc_auc=roc_auc_score(labels, scores),
                pr_auc=average_precision_score(labels, scores),
                n=len(labels), n_pos=int(sum(labels)))


def main():
    rbps = pd.read_csv(f"{OUT_DIR}/stage3_rbps_tagged.csv")
    ni = rbps[rbps.stratum == "near-identical"]
    print(f"{len(ni)} near-identical KlebPhaCol RBPs across {ni.phage_ID.nunique()} phages")

    pos_serotypes = training_positive_serotypes_by_phage()

    # phage_ID -> union of positive-training-serotypes across ALL its near-identical RBPs' matched training phages
    phage_matched_serotypes = {}
    for phage, sub in ni.groupby("phage_ID"):
        s = set()
        for bk_phage in sub.best_boeckaerts_phage.unique():
            s |= pos_serotypes.get(bk_phage, set())
        phage_matched_serotypes[phage] = s
        print(f"  {phage}: near-identical RBP(s) matched to {sorted(sub.best_boeckaerts_phage.unique())} "
              f"-> positive training serotypes {sorted(s) if s else '(none -- matched phage had no positive training interactions)'}")

    # host crosswalk (reuse Stage 4's exact logic)
    sero_encoded, sero_cols = s4.build_training_serotype_vocab()
    hosts, host_S, n_absent = s4.build_host_sero_matrix(sero_cols)
    host_type = dict(zip(hosts.strain, hosts.training_type))  # NaN for unseen-KL hosts

    # load predictions (recompute -- cheap, cached model, no retrain)
    import pickle
    with open(s4.MODEL_PATH, "rb") as fh:
        saved = pickle.load(fh)
    model = saved["model"]
    rbp_emb = np.load(s4.RBP_EMB)
    rbp_gene_idx = {g: i for i, g in enumerate(rbps.gene_ID)}
    pred_full = s4.predict_klebphacol(model, sero_cols, hosts, host_S, rbps, rbp_emb, rbp_gene_idx)

    ni_phages = set(phage_matched_serotypes)
    print("\n" + "=" * 66)
    print("MATCHED vs MISMATCHED split, per interaction table")
    print("=" * 66)

    for label, path in {
        "LB strict": f"{OUT_DIR}/interactions_LB_strict.csv",
        "LB permissive": f"{OUT_DIR}/interactions_LB_permissive.csv",
        "TSB strict": f"{OUT_DIR}/interactions_TSB_strict.csv",
    }.items():
        inter = pd.read_csv(path)
        inter = inter[inter.phage.isin(ni_phages)]
        merged = inter.merge(pred_full, on=["phage", "strain"], how="left").dropna(subset=["score"])
        merged["host_serotype"] = merged.strain.map(host_type)
        merged["has_valid_serotype"] = merged.host_serotype.notna()
        merged["matched"] = merged.apply(
            lambda r: (r.has_valid_serotype and
                       r.host_serotype in phage_matched_serotypes.get(r.phage, set())),
            axis=1)

        print(f"\n--- {label} ---")
        # primary test: excludes hosts with no valid crosswalked serotype at all
        valid = merged[merged.has_valid_serotype]
        n_excluded_novocab = len(merged) - len(valid)
        for split_name, split_mask in (("MATCHED", valid.matched), ("MISMATCHED", ~valid.matched)):
            sub = valid[split_mask]
            m = compute_metrics(sub.label.values, sub.score.values)
            lift = (sub.label.mean() / merged.label.mean()) if len(sub) and merged.label.mean() > 0 else float("nan")
            print(f"  [primary, excl {n_excluded_novocab} no-valid-serotype rows] {split_name:<11} "
                  f"n={m['n']:>5} n_pos={m['n_pos']:>4} base_rate={100*sub.label.mean() if len(sub) else float('nan'):.1f}% "
                  f"lift={lift:.2f}x ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f}")

        # secondary: including no-valid-serotype hosts folded into MISMATCHED
        for split_name, split_mask in (("MATCHED", merged.matched), ("MISMATCHED", ~merged.matched)):
            sub = merged[split_mask]
            m = compute_metrics(sub.label.values, sub.score.values)
            lift = (sub.label.mean() / merged.label.mean()) if len(sub) and merged.label.mean() > 0 else float("nan")
            print(f"  [incl unseen-KL hosts folded into MISMATCHED] {split_name:<11} "
                  f"n={m['n']:>5} n_pos={m['n_pos']:>4} base_rate={100*sub.label.mean() if len(sub) else float('nan'):.1f}% "
                  f"lift={lift:.2f}x ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f}")

    print("\n" + "=" * 66)
    print("RAW COUNTS: cross-collection serotype reassignment")
    print("=" * 66)
    inter = pd.read_csv(f"{OUT_DIR}/interactions_LB_strict.csv")
    inter = inter[inter.phage.isin(ni_phages)]
    merged = inter.merge(pred_full, on=["phage", "strain"], how="left").dropna(subset=["score"])
    merged["host_serotype"] = merged.strain.map(host_type)
    merged["has_valid_serotype"] = merged.host_serotype.notna()
    merged["matched"] = merged.apply(
        lambda r: (r.has_valid_serotype and
                   r.host_serotype in phage_matched_serotypes.get(r.phage, set())),
        axis=1)
    n_mismatched = (~merged.matched).sum()
    print(f"LB strict, all pairs: {n_mismatched}/{len(merged)} near-identical pairs "
          f"are MISMATCHED ({100*n_mismatched/len(merged):.1f}%)")
    pos = merged[merged.label == 1]
    n_mismatched_pos = (~pos.matched).sum()
    print(f"LB strict, POSITIVE pairs only: {n_mismatched_pos}/{len(pos)} are MISMATCHED "
          f"({100*n_mismatched_pos/len(pos):.1f}%) -- i.e. this many real KlebPhaCol "
          f"infections involve a near-identical RBP paired with a host serotype the "
          f"matched training RBP was never positively associated with.")

    # distinct RBPs (phages, since stratum/matching is phage-level) that have >=1
    # POSITIVE KlebPhaCol interaction with a serotype outside their matched
    # training serotype set -- the RBP itself "appears with a different serotype
    # across the two collections"
    phages_with_reassignment = sorted(pos[~pos.matched].phage.unique())
    n_rbps_reassigned = ni[ni.phage_ID.isin(phages_with_reassignment)].gene_ID.nunique()
    print(f"\nPhages with >=1 positive KlebPhaCol interaction outside their matched "
          f"training serotype set: {len(phages_with_reassignment)}/{ni.phage_ID.nunique()} "
          f"-> {phages_with_reassignment}")
    print(f"Distinct near-identical RBPs on those phages: {n_rbps_reassigned}/{len(ni)}")
    print("This count stands regardless of how the MATCHED/MISMATCHED split's AUCs "
          "come out -- it is the direct evidence for or against RBP reuse across a "
          "different serotype between the two collections.")


if __name__ == "__main__":
    main()
