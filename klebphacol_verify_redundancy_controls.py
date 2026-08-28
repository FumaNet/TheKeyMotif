#!/usr/bin/env python3
"""
klebphacol_verify_redundancy_controls.py — verifies the genome-ANI vs
RBP-identity "redundancy controls don't nest" claim in stage6_head_to_head.md
by recomputing both sides consistently: same metric (rapidfuzz Levenshtein
normalized_similarity), same RBP set (193 RBPs, RBPdetect>=0.5, 200<len<1500)
the stratification actually uses, and the TRUE argmax (not just "any pairwise
value >95%") on the genome-ANI side.

Corrects two errors found in an earlier version of that section:
  - K7PH164C4 does NOT lack a near-identical RBP -- it's the consistent
    2nd/3rd-best hit (95.9-98.3%) for 4 Roth-series RBPs, narrowly beaten
    each time by K30lambda2/K32PH164C1 (both already excluded). The argmax
    exclusion rule only removes the single best-scoring phage per query, so
    a tightly-clustered runner-up like K7PH164C4 stays in training.
  - K40PH129C1 only reached the original 7-phage exclusion list via a
    DepoScope-only KlebPhaCol RBP outside the 193-RBP RBPdetect set --
    checked against the 193-set consistently, its max identity is 26.53%,
    nowhere near 95%.
"""
import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process

OUT_DIR = "Results/klebphacol"


def genome_ani_argmax_exclusions():
    ani = pd.read_csv(f"{OUT_DIR}/ani_phages_raw.txt", sep="\t", header=None,
                       names=["query", "ref", "ani", "fm", "ft"])
    import re, os

    def clean_id(fname):
        base = os.path.basename(fname)
        base = re.sub(r"\.fasta$", "", base)
        base = re.sub(r"\s*\(aka[^)]*\)\s*$", "", base)
        return base.split("__")[0]
    ani["query_id"] = ani["query"].apply(clean_id)
    ani["ref_id"] = ani["ref"].apply(clean_id)
    best = ani.loc[ani.groupby("query_id")["ani"].idxmax()]
    return set(best[best.ani > 95].ref_id.unique()), best


def main():
    targets = ["K7PH164C4", "K2064PH2", "K30lambda2", "K40PH129C1", "K52PH129C1"]
    rbps_kp = pd.read_csv(f"{OUT_DIR}/stage3_rbps_tagged.csv")
    bk_all = pd.read_csv("Data/RBPbase.csv")
    excl = pd.read_csv(f"{OUT_DIR}/overlap_exclusions.csv")
    excluded_7 = set(excl.boeckaerts_phage_to_exclude.unique())
    genome_excl_set, genome_best = genome_ani_argmax_exclusions()

    print("Genome-ANI argmax-exclusion set (re-derived):", sorted(genome_excl_set))
    print(f"\n{'phage':<14}{'genome ANI argmax >95%':<26}{'max RBP identity (193-set)':<30}{'on RBP-exclusion list':<24}")
    for t in targets:
        bk_sub = bk_all[bk_all.phage_ID == t]
        sim = process.cdist(bk_sub.protein_sequence.tolist(), rbps_kp.protein_sequence.tolist(),
                             scorer=Levenshtein.normalized_similarity) * 100
        max_id = sim.max() if sim.size else float("nan")
        print(f"{t:<14}{str(t in genome_excl_set):<26}{max_id:<30.2f}{str(t in excluded_7):<24}")

    print("\n--- K7PH164C4's 4 near-identical RBPs: full context vs all 274 training RBPs ---")
    ni = rbps_kp[(rbps_kp.stratum == "near-identical") & (rbps_kp.best_boeckaerts_phage == "K7PH164C4")]
    for gene in ni.gene_ID:
        query_seq = rbps_kp.loc[rbps_kp.gene_ID == gene, "protein_sequence"].values[0]
        sim_all = process.cdist([query_seq], bk_all.protein_sequence.tolist(),
                                 scorer=Levenshtein.normalized_similarity)[0] * 100
        top3 = sim_all.argsort()[::-1][:3]
        print(f"  {gene}: " + ", ".join(
            f"{bk_all.iloc[i].phage_ID}={sim_all[i]:.2f}%"
            f"{'*' if bk_all.iloc[i].phage_ID in excluded_7 else ''}"
            for i in top3))
    print("  (* = on the 7-phage exclusion list; K7PH164C4 is consistently runner-up)")


if __name__ == "__main__":
    main()
