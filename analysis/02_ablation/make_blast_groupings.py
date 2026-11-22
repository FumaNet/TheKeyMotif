"""Write phage grouping CSVs for the BLASTp-grouped phage-held-out sweep.

Consumes results/metric_calibration_pairs.csv (all 37,401 training-RBP pairs,
BLASTp pident + query coverage + normalised Levenshtein) and emits one
phage_ID,group_id file per threshold into results/groupings/, for
2d_phage_holdout.py --groups-file.

Adjacency rule, identical to cluster_phages_by_identity(): single-linkage
union-find over phages; two phages adjacent when ANY cross-phage RBP pair meets
the threshold; same-phage RBP pairs skipped; all phages in RBPbase.csv seeded so
a phage with no qualifying pair survives as its own singleton group.

BLASTp files additionally require query coverage >= COV_FLOOR, because BLASTp
pident is a local statistic: a high pident over a short HSP is not the claim the
grouping is meant to encode. The nLev file uses the sweep's own metric at 1.00
and exists to verify this path reproduces the published 99 groups.

No training, no GPU. Run from the repo root:
    python analysis/02_ablation/make_blast_groupings.py
"""
import os

import pandas as pd

PAIRS = "results/metric_calibration_pairs.csv"
RBP = "data/RBPbase.csv"
OUTDIR = "results/groupings"
COV_FLOOR = 50.0

BLAST_THRESHOLDS = [100, 95, 90, 80, 70, 60, 50]
DEGENERATE = [40, 30]          # reported, not written: no usable folds
EXPECTED = {"blast90": 34, "blast80": 28, "blast70": 24,
            "blast60": 21, "blast50": 17, "nlev100": 99}


def union_find(all_phages, edges):
    """Single-linkage over phages. Returns {phage_ID: group_label}."""
    idx = {p: k for k, p in enumerate(all_phages)}
    parent = list(range(len(all_phages)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pa, pb in edges:
        ra, rb = find(idx[pa]), find(idx[pb])
        if ra != rb:
            parent[rb] = ra

    roots = {p: find(idx[p]) for p in all_phages}
    labels = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {p: labels[r] for p, r in roots.items()}


def stats(groups):
    s = pd.Series(groups).value_counts()
    return len(s), int(s.max()), int((s == 1).sum())


def write(name, groups, all_phages):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"groups_{name}.csv")
    pd.DataFrame({"phage_ID": all_phages,
                  "group_id": [groups[p] for p in all_phages]}).to_csv(
                      path, index=False)
    n, largest, singles = stats(groups)
    exp = EXPECTED.get(name)
    ok = "" if exp is None else ("  MATCHES Table B" if n == exp
                                 else f"  *** MISMATCH, expected {exp} ***")
    print(f"  {path:<42} {n:>3} groups  largest {largest:>3}  "
          f"singletons {singles:>3}{ok}")
    return n == exp if exp is not None else True


def main():
    pairs = pd.read_csv(PAIRS)
    all_phages = sorted(pd.read_csv(RBP)["phage_ID"].unique())
    print(f"{len(pairs):,} RBP pairs, {len(all_phages)} phages seeded from {RBP}")

    cross = pairs[pairs.phage_a != pairs.phage_b]
    hit = cross[cross.blast_pident.notna()]
    covered = hit[hit.blast_qcovs >= COV_FLOOR]
    print(f"cross-phage pairs {len(cross):,}  with BLASTp hit {len(hit):,}  "
          f"and qcovs >= {COV_FLOOR:.0f}% {len(covered):,}\n")

    print(f"BLASTp pident, qcovs >= {COV_FLOOR:.0f}% (Table B rule):")
    allok = True
    for t in BLAST_THRESHOLDS:
        e = covered[covered.blast_pident >= t]
        g = union_find(all_phages, zip(e.phage_a, e.phage_b))
        allok &= write(f"blast{t}", g, all_phages)

    print("\nNormalised Levenshtein 1.00 (verification run, sweep's own metric):")
    e = cross[cross.nlev >= 1.00]
    g = union_find(all_phages, zip(e.phage_a, e.phage_b))
    allok &= write("nlev100", g, all_phages)

    print(f"\nDegenerate thresholds -- NOT written, reported for citation "
          f"(same qcovs >= {COV_FLOOR:.0f}% rule):")
    for t in DEGENERATE:
        e = covered[covered.blast_pident >= t]
        n, largest, singles = stats(union_find(all_phages,
                                               zip(e.phage_a, e.phage_b)))
        print(f"  BLASTp {t}%: {n:>3} groups  largest {largest:>3} of "
              f"{len(all_phages)}  singletons {singles:>3}   "
              f"-> {100*largest/len(all_phages):.0f}% of phages in one fold")

    print("\nALL COUNTS MATCH Table B" if allok else
          "\n*** SOME COUNTS DIVERGED -- do not run the sweep until resolved ***")


if __name__ == "__main__":
    main()
