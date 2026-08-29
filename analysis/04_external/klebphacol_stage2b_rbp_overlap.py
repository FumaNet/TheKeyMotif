#!/usr/bin/env python3
"""
klebphacol_stage2b_rbp_overlap.py — Stage 2b: RBP-level overlap between
KlebPhaCol and Boeckaerts, which supersedes Stage 2's genome-ANI phage rule.
PHL-RBP+S consumes RBP embeddings, not whole genomes -- a KlebPhaCol phage
and a Boeckaerts phage can be genomically ~96% similar (Stage 2) yet carry
completely different RBPs (no leakage), or conversely share one near-
identical RBP despite the rest of the genome differing. RBP identity is the
thing that can actually leak into the model's input.

KlebPhaCol candidate RBPs: Table S3, protein_sequence, filtered to the
table's own caption criterion -- "highlighted those with score >=0.5" --
scores_DepoScope>=0.5 OR scores_RBPDetect>=0.5. Of 9,973 total scored
proteins (i.e. most/all ORFs in each phage, not just RBP candidates), 213
clear this bar (4.1/phage), a scale comparable to Boeckaerts' 274/105 =
2.6/phage. Using the full 9,973 would compare whole proteomes, not RBPs.

Boeckaerts RBPs: Data/RBPbase.csv, already the curated 274-RBP set (no
further filtering -- this IS Boeckaerts' RBP list, not a scored superset).

Method: rapidfuzz Levenshtein normalized_similarity, all pairs (213 x 274 =
58,362 -- seconds, as expected).

Exclusion rule (supersedes Stage 2's genome-ANI phage rule): any Boeckaerts
PHAGE carrying an RBP >=95% identical to any KlebPhaCol RBP is excluded from
training. Both rules' results are reported for comparison, but only the
RBP-identity rule drives Results/klebphacol/overlap_exclusions.csv, which
this script rewrites to contain ONLY RBP-identity-triggered phage
exclusions (hosts are never excluded -- see script docstring in
klebphacol_stage2_hosts_revised.py for why).
"""
import os
import pyxlsb
import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process

XLSB_PATH = "Data/klebphacol/Supplementary_Tables_R2.xlsb"
RBPBASE_PATH = "Data/RBPbase.csv"
OUT_PATH = "Results/klebphacol/overlap_exclusions.csv"


def load_klebphacol_rbps():
    with pyxlsb.open_workbook(XLSB_PATH) as wb:
        with wb.get_sheet("Table S3") as sheet:
            rows = list(sheet.rows())
    header = [c.v for c in rows[1]]
    data = []
    for r in rows[2:]:
        vals = [c.v for c in r]
        if vals[0] is None:
            break
        data.append(vals[:len(header)])
    df = pd.DataFrame(data, columns=header)
    n_total = len(df)
    mask = (df.scores_DepoScope >= 0.5) | (df.scores_RBPDetect >= 0.5)
    rbps = df[mask].copy()
    print(f"Table S3: {n_total} total scored proteins, {len(rbps)} pass "
          f"scores_DepoScope>=0.5 OR scores_RBPDetect>=0.5 "
          f"({len(rbps)/52:.1f}/phage vs Boeckaerts' 274/105=2.6/phage)")
    return rbps[["phage_ID", "gene_ID", "protein_sequence"]].reset_index(drop=True)


def load_boeckaerts_rbps():
    df = pd.read_csv(RBPBASE_PATH)
    print(f"RBPbase.csv: {len(df)} Boeckaerts RBPs across "
          f"{df.phage_ID.nunique()} phages")
    return df[["phage_ID", "protein_ID", "protein_sequence"]].reset_index(drop=True)


def main():
    kp = load_klebphacol_rbps()
    bk = load_boeckaerts_rbps()

    print(f"\nComputing {len(kp)} x {len(bk)} = {len(kp)*len(bk):,} pairwise "
          f"Levenshtein normalized_similarity...")
    kp_seqs = kp.protein_sequence.tolist()
    bk_seqs = bk.protein_sequence.tolist()
    sim_matrix = process.cdist(kp_seqs, bk_seqs, scorer=Levenshtein.normalized_similarity)
    sim_matrix = sim_matrix * 100  # -> percent identity scale, matching the ANI reporting convention

    best_idx = sim_matrix.argmax(axis=1)
    best_sim = sim_matrix.max(axis=1)
    kp["best_boeckaerts_phage"] = bk.phage_ID.values[best_idx]
    kp["best_boeckaerts_protein"] = bk.protein_ID.values[best_idx]
    kp["best_identity"] = best_sim

    print("\n" + "=" * 66)
    print("STAGE 2b CHECKPOINT: RBP-level overlap")
    print("=" * 66)
    print(f"\nDistribution of each KlebPhaCol RBP's best-hit identity to a "
          f"Boeckaerts RBP (n={len(kp)}):")
    print(kp.best_identity.describe())
    for thresh in (95, 90, 80):
        n = (kp.best_identity >= thresh).sum()
        print(f"  >= {thresh}%: {n}/{len(kp)} ({100*n/len(kp):.1f}%)")

    bins = list(range(0, 101, 10))
    counts, edges = pd.cut(kp.best_identity, bins=bins, right=False).value_counts().sort_index(), None
    print("\nHistogram:")
    for interval, c in counts.items():
        print(f"    [{interval.left:>3}-{interval.right:>3}) {c:>4} " + "#" * c)

    over95 = kp[kp.best_identity >= 95]
    excl_phages = sorted(over95.best_boeckaerts_phage.unique())
    print(f"\nDistinct Boeckaerts phages carrying an RBP >=95% identical to "
          f"a KlebPhaCol RBP: {len(excl_phages)}/105 -> {excl_phages}")

    out = over95[["phage_ID", "gene_ID", "best_boeckaerts_phage",
                   "best_boeckaerts_protein", "best_identity"]].copy()
    out.columns = ["klebphacol_phage", "klebphacol_rbp_gene_id",
                    "boeckaerts_phage_to_exclude", "boeckaerts_rbp_protein_id", "identity_pct"]
    out["rule"] = "rbp_identity_95"
    out = out.sort_values("identity_pct", ascending=False)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(f"  GRANULARITY: one row = one (KlebPhaCol RBP, its best-hit Boeckaerts RBP) "
          f"pair that triggered exclusion, NOT one row per excluded Boeckaerts phage.")
    print(f"  {len(out)} trigger rows -> {out.boeckaerts_phage_to_exclude.nunique()} "
          f"distinct Boeckaerts phages excluded from training (some phages are hit by "
          f"multiple KlebPhaCol RBPs, or by the same RBP-pair logic more than once).")
    print(f"  This file now contains ONLY phage exclusions from the RBP-identity rule. "
          f"Hosts are never excluded (see Stage 2 host revision) and the genome-ANI "
          f"phage rule (Stage 2) is superseded, not merged in.")
    if out.empty:
        print("  (No rows -> no Boeckaerts phages excluded under this rule.)")


if __name__ == "__main__":
    main()
