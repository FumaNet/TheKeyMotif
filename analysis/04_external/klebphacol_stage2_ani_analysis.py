#!/usr/bin/env python3
"""
klebphacol_stage2_ani_analysis.py — Stage 2 checkpoint: analyze fastANI
output for phage and host cross-collection overlap between KlebPhaCol and
Boeckaerts, and write the training-set exclusion list.

fastANI's --minFraction (default 0.2) means a query-ref pair with too little
shared/alignable sequence is OMITTED from the output entirely, not reported
with a low ANI. For phages especially, most cross-pairs will be genuinely
unrelated and never appear -- so "best hit" is computed only over queries
that got at least one reported pair, and the count with ZERO reported pairs
is reported separately rather than silently excluded from view.

For hosts, KlebPhaCol strains split into two groups with different
provenance: 69 resolved directly from a Table S1 accession, and 5
(NCTC_13368, ATCC_11296, NCTC_13438, NCTC_7427, NCTC_13443) resolved by
searching NCBI on strain name because Table S1 gives no accession for them
at all (see results/klebphacol/host_accessions_resolved.csv). Distributions
are reported separately for these two groups so a name-resolution error
can't be mistaken for a real cross-collection overlap finding.

Rule (fixed, not a judgement call): best-hit ANI > 95% -> the matched
BOECKAERTS entry is excluded from training. Never touch the KlebPhaCol
(test) side.
"""
import os
import re
import pandas as pd
import numpy as np

ANI_COLS = ["query", "ref", "ani", "frags_matched", "frags_total"]


def clean_id(fname):
    """'Roth01__PQ657785.fasta' -> 'Roth01'; 'KLEB11.fasta' -> 'KLEB11';
    'MKP103 (aka KPNIH1).fasta' -> 'MKP103' (the two hosts originally
    fetched under Table S1's raw isolate name, including its parenthetical
    alias and a literal space, were renamed on disk to their canonical S4
    name -- this strips the same pattern for anyone re-deriving IDs from
    the raw ani_*.txt path text, which still has the old names baked in).
    Boeckaerts phage/host files have no '__' or ' (aka ...)' suffix, so
    this is a no-op for those beyond stripping the extension."""
    base = os.path.basename(fname)
    base = re.sub(r"\.fasta$", "", base)
    base = re.sub(r"\s*\(aka[^)]*\)\s*$", "", base)
    return base.split("__")[0]


def load_ani(path):
    df = pd.read_csv(path, sep="\t", header=None, names=ANI_COLS)
    df["query_id"] = df["query"].apply(clean_id)
    df["ref_id"] = df["ref"].apply(clean_id)
    return df


def best_hits(df, all_query_ids):
    """One row per query: its single best (highest-ANI) ref hit. Queries
    with zero reported pairs get ani=NaN, so they're visible, not dropped."""
    idx = df.groupby("query_id")["ani"].idxmax()
    best = df.loc[idx].set_index("query_id")
    out = pd.DataFrame(index=all_query_ids)
    out = out.join(best[["ref_id", "ani", "frags_matched", "frags_total"]])
    out["coverage"] = out["frags_matched"] / out["frags_total"]
    return out.reset_index().rename(columns={"index": "query_id"})


def report_distribution(label, best_df):
    n_total = len(best_df)
    hit = best_df.dropna(subset=["ani"])
    n_hit = len(hit)
    print(f"\n--- {label} ---")
    print(f"  N queries: {n_total}, with >=1 reported pair (ANI estimable): {n_hit}, "
          f"with ZERO reported pairs (too divergent for minFraction=0.2): {n_total - n_hit}")
    if n_hit == 0:
        print("  No ANI values to summarise.")
        return
    print(f"  best-hit ANI: min={hit.ani.min():.2f}  median={hit.ani.median():.2f}  "
          f"max={hit.ani.max():.2f}  mean={hit.ani.mean():.2f}")
    print(f"  >95% ANI (overlap threshold): {(hit.ani > 95).sum()}/{n_hit}")
    # text histogram, 5-point bins from 75 to 100
    bins = list(range(70, 101, 5))
    counts, edges = np.histogram(hit.ani, bins=bins)
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * c
        print(f"    [{lo:>3}-{hi:>3}) {c:>3} {bar}")


def build_exclusions(best_df, side_label, klebphacol_col="query_id", boeckaerts_col="ref_id"):
    hit = best_df.dropna(subset=["ani"])
    over = hit[hit.ani > 95]
    rows = []
    for _, r in over.iterrows():
        rows.append(dict(klebphacol_id=r[klebphacol_col], boeckaerts_id=r[boeckaerts_col],
                          ani=r.ani, coverage=r.coverage, side=side_label))
    return rows


def main():
    kp_phage_ids = [clean_id(p) for p in
                     open("/tmp/klebphacol_phage_list.txt").read().splitlines() if p]
    kp_host_ids = [clean_id(p) for p in
                    open("/tmp/klebphacol_host_list.txt").read().splitlines() if p]
    assert len(kp_phage_ids) == 52, f"expected 52 phage query IDs, got {len(kp_phage_ids)}"
    assert len(kp_host_ids) == 74, f"expected 74 host query IDs, got {len(kp_host_ids)}"

    print("=" * 66)
    print("STAGE 2 CHECKPOINT: fastANI overlap")
    print("=" * 66)

    ani_phage = load_ani("results/klebphacol/ani_phages_raw.txt")
    phage_best = best_hits(ani_phage, kp_phage_ids)
    report_distribution("PHAGES: KlebPhaCol (n=52) best hit among Boeckaerts (n=105)",
                         phage_best)

    ani_host = load_ani("results/klebphacol/ani_hosts_raw.txt")
    host_best = best_hits(ani_host, kp_host_ids)

    resolved = pd.read_csv("results/klebphacol/host_accessions_resolved.csv")
    name_based_strains = set(resolved.strain)
    host_best["resolution"] = np.where(host_best.query_id.isin(name_based_strains),
                                        "name-based", "table-resolved")

    table_best = host_best[host_best.resolution == "table-resolved"]
    name_best = host_best[host_best.resolution == "name-based"]
    report_distribution(f"HOSTS (table-resolved, n={len(table_best)}): "
                         f"KlebPhaCol best hit among Boeckaerts (n=200)", table_best)
    report_distribution(f"HOSTS (name-based, n={len(name_best)}): "
                         f"KlebPhaCol best hit among Boeckaerts (n=200)", name_best)

    print("\n" + "=" * 66)
    print("EXCLUSION LIST (best-hit ANI > 95% -> exclude BOECKAERTS entry from training)")
    print("=" * 66)
    excl_rows = []
    excl_rows += build_exclusions(phage_best, "phage")
    excl_rows += build_exclusions(host_best, "host")
    # tag host exclusions with resolution provenance so a name-based-only
    # overlap is visible rather than silently merged with table-resolved ones
    excl_df = pd.DataFrame(excl_rows)
    if len(excl_df):
        res_map = host_best.set_index("query_id")["resolution"].to_dict()
        excl_df["host_resolution"] = excl_df.apply(
            lambda r: res_map.get(r.klebphacol_id, "") if r.side == "host" else "", axis=1)
    out_path = "results/klebphacol/overlap_exclusions.csv"
    excl_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(excl_df)} rows)")
    if len(excl_df):
        print(excl_df.to_string(index=False))
        if "host_resolution" in excl_df.columns:
            name_based_overlaps = excl_df[(excl_df.side == "host") &
                                           (excl_df.host_resolution == "name-based")]
            if len(name_based_overlaps):
                print(f"\n*** {len(name_based_overlaps)} overlap(s) involve a "
                      f"NAME-BASED-resolved KlebPhaCol host -- verify these "
                      f"before trusting them, since the KlebPhaCol genome "
                      f"itself was chosen by name search, not from the "
                      f"source table. ***")
    else:
        print("No pairs exceeded 95% ANI -- no exclusions.")

    n_boeckaerts_phages_excluded = excl_df[excl_df.side == "phage"].boeckaerts_id.nunique() if len(excl_df) else 0
    n_boeckaerts_hosts_excluded = excl_df[excl_df.side == "host"].boeckaerts_id.nunique() if len(excl_df) else 0
    print(f"\nDistinct Boeckaerts phages to exclude from training: {n_boeckaerts_phages_excluded}/105")
    print(f"Distinct Boeckaerts hosts to exclude from training: {n_boeckaerts_hosts_excluded}/200")
    print("\n(KlebPhaCol/test side: untouched, per instructions.)")


if __name__ == "__main__":
    main()
