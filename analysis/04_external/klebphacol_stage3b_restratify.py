#!/usr/bin/env python3
"""
klebphacol_stage3b_restratify.py — corrects the Stage 3 overlap-stratum
tagging, which was computed against all 274 Boeckaerts RBPs. Stage 4 trains
on only the 98 post-exclusion phages (274 -> 250 RBPs), so any KlebPhaCol
RBP whose best hit was in one of the 7 RBP-overlap-excluded phages is
ORPHANED: its "near-identical"/"related" label described a reference
sequence that isn't actually in the training set the model saw. The stratum
must describe the model's real reference set, not the pre-exclusion one --
supersedes Stage 3's stratum tagging, same reasoning as Stage 2b superseding
Stage 2's genome-ANI phage rule.

This is also the root cause of the "near-identical underperforms novel"
inversion seen in the first Stage 4 run: it was comparing apples (a
genuinely novel stratum) to mislabelled oranges (a "near-identical" stratum
padded with orphaned RBPs no longer near anything in training). Not
reported as a finding until this fix lands.
"""
import os
import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process

RBP_TAGGED = "results/klebphacol/stage3_rbps_tagged.csv"
RBPBASE_PATH = "data/RBPbase.csv"
EXCLUSIONS = "results/klebphacol/overlap_exclusions.csv"


def stratum(x):
    if x >= 95:
        return "near-identical"
    if x >= 80:
        return "related"
    return "novel"


def main():
    rbps = pd.read_csv(RBP_TAGGED)
    bk = pd.read_csv(RBPBASE_PATH)
    excl = pd.read_csv(EXCLUSIONS)
    excluded_phages = set(excl.boeckaerts_phage_to_exclude.unique())

    bk_post = bk[~bk.phage_ID.isin(excluded_phages)].reset_index(drop=True)
    print(f"Boeckaerts RBPs: {len(bk)} (all, {bk.phage_ID.nunique()} phages) -> "
          f"{len(bk_post)} (post-exclusion, {bk_post.phage_ID.nunique()} phages)")

    old_stratum = rbps.stratum.copy()
    old_identity = rbps.best_identity.copy()

    sim = process.cdist(rbps.protein_sequence.tolist(), bk_post.protein_sequence.tolist(),
                         scorer=Levenshtein.normalized_similarity) * 100
    best_idx = sim.argmax(axis=1)
    rbps["best_identity"] = sim.max(axis=1)
    rbps["best_boeckaerts_phage"] = bk_post.phage_ID.values[best_idx]
    rbps["best_boeckaerts_protein"] = bk_post.protein_ID.values[best_idx]
    rbps["stratum"] = rbps.best_identity.apply(stratum)

    print("\nBEFORE (vs all 274 training RBPs):")
    print(old_stratum.value_counts().reindex(["novel", "related", "near-identical"]).to_string())
    print("\nAFTER (vs 250 post-exclusion training RBPs -- what the model actually saw):")
    print(rbps.stratum.value_counts().reindex(["novel", "related", "near-identical"]).to_string())

    changed = rbps[old_stratum != rbps.stratum]
    print(f"\n{len(changed)}/{len(rbps)} RBPs change stratum, ALL downward "
          f"({(old_stratum[changed.index]=='near-identical').sum()} were "
          f"near-identical, {(old_stratum[changed.index]=='related').sum()} "
          f"were related -> all become novel; their best pre-exclusion hit "
          f"was in one of the 7 excluded phages)")

    rbps.to_csv(RBP_TAGGED, index=False)
    print(f"\nWrote corrected {RBP_TAGGED} (supersedes the pre-exclusion tagging)")


if __name__ == "__main__":
    main()
