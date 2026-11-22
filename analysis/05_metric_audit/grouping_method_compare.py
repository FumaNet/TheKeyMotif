"""Test 1 from GROUPING_INVESTIGATION.md.

Read-only comparison. Does NOT modify or import 2d_phage_holdout.py or
preflight_grouping.py -- it rebuilds the same phage-level adjacency those
scripts use, then clusters it two different ways:

  (a) union-find single linkage -- what cluster_phages_by_identity() does now.
  (b) neighbourhood-assignment, ported verbatim (same iteration order, same
      overwrite behaviour) from the loci-grouping loop in
      src/finding_groups.py -- the hypothesis is that THIS is what actually
      produced CLAUDE.md's established group counts (99, 89, 80, 69).

No GPU, no training. Run from the repo root: python grouping_method_compare.py
"""

import numpy as np
import pandas as pd

try:
    from rapidfuzz.distance import Levenshtein
    identity = Levenshtein.normalized_similarity
except ImportError:
    raise SystemExit(
        "rapidfuzz is required for this test (the brief says fail loudly, "
        "not fall back to difflib -- a different fallback metric would "
        "contaminate the comparison). pip install rapidfuzz.")

RBP_FILE = "data/RBPbase.csv"
COL_PHAGE = "phage_ID"
COL_SEQ = "protein_sequence"
TEST_THRESHOLDS = [1.00, 0.95, 0.90, 0.80]
HISTORICAL = {1.00: 99, 0.95: 89, 0.90: 80, 0.80: 69}


def build_phage_identity_matrix():
    """Same adjacency the sweep uses: phages i,j linked if ANY pair of their
    RBPs reaches a given identity. Stored as a continuous max-identity matrix
    so every threshold can be tested by a simple >= comparison, exactly like
    finding_groups.py thresholds all_loci_score_matrix.txt."""
    df = pd.read_csv(RBP_FILE)
    phages = df[COL_PHAGE].tolist()
    seqs = df[COL_SEQ].tolist()
    uniq = sorted(set(phages))
    idx = {p: k for k, p in enumerate(uniq)}
    n = len(uniq)

    mat = np.zeros((n, n))
    np.fill_diagonal(mat, 1.0)

    n_rbp = len(seqs)
    for i in range(n_rbp):
        for j in range(i + 1, n_rbp):
            pi, pj = idx[phages[i]], idx[phages[j]]
            if pi == pj:
                continue
            v = identity(seqs[i], seqs[j])
            if v > mat[pi, pj]:
                mat[pi, pj] = mat[pj, pi] = v

    return uniq, mat


def cluster_union_find(mat, threshold):
    """(a) True single linkage -- what cluster_phages_by_identity() does."""
    n = mat.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in zip(*np.where(np.triu(mat >= threshold, k=1))):
        union(i, j)

    roots = [find(k) for k in range(n)]
    return roots


def cluster_neighbourhood_assignment(mat, threshold):
    """(b) Ported verbatim from src/finding_groups.py's loci-grouping loop.

    Original (host/loci side), for reference:

        group_i = 0
        new_groups = [np.nan] * len(groups_loci)
        for i in range(matrix.shape[0]):
            cluster = np.where(matrix[i,:] >= threshold)[0]
            oldgroups_i = [k for k, x in enumerate(groups_loci) if x in cluster]
            if i in groups_loci and np.isnan(new_groups[groups_loci.index(i)]):
                for ogi in oldgroups_i:
                    new_groups[ogi] = group_i
                group_i += 1
        groups_loci = new_groups

    There, `groups_loci` has one entry per interaction ROW (many rows share
    the same loci index) and `i in groups_loci` / `.index(i)` tests/finds the
    first row referencing entity i. Here every phage is its own single "row"
    (there is exactly one group-slot per phage, not per interaction row), so
    `groups_loci` collapses to `list(range(n))` -- every entity trivially "is
    in" that list once, at position i, and `.index(i) == i`. Same iteration
    order, same overwrite behaviour, same stopping condition; the only
    simplification is the node:row ratio, not the algorithm.
    """
    n = mat.shape[0]
    group_i = 0
    new_groups = [np.nan] * n
    for i in range(n):
        cluster = np.where(mat[i, :] >= threshold)[0]
        if np.isnan(new_groups[i]):
            for j in cluster:
                new_groups[j] = group_i
            group_i += 1
    return new_groups


def leakage_count(mat, threshold, groups):
    """Phage pairs adjacent at `threshold` that ended up in different groups.
    Zero by construction under union-find; may not be under (b)."""
    n = mat.shape[0]
    leaks = 0
    for i, j in zip(*np.where(np.triu(mat >= threshold, k=1))):
        if groups[i] != groups[j]:
            leaks += 1
    return leaks


def main():
    uniq, mat = build_phage_identity_matrix()
    n = len(uniq)
    print(f"{n} phages, continuous max-RBP-identity matrix built "
          f"({n * (n - 1) // 2} phage pairs)\n")

    print(f"{'thresh':>8}{'union-find (a)':>16}{'neighbourhood (b)':>20}"
          f"{'CLAUDE.md':>12}{'(b) matches hist.':>20}{'(b) leakage':>14}")
    for t in TEST_THRESHOLDS:
        groups_a = cluster_union_find(mat, t)
        n_a = len(set(groups_a))

        groups_b = cluster_neighbourhood_assignment(mat, t)
        n_b = len(set(groups_b))
        leak_b = leakage_count(mat, t, groups_b)

        hist = HISTORICAL[t]
        match = "YES" if n_b == hist else "no"
        print(f"{t:>8.2f}{n_a:>16}{n_b:>20}{hist:>12}{match:>20}{leak_b:>14}")

    print()
    all_match = all(
        len(set(cluster_neighbourhood_assignment(mat, t))) == HISTORICAL[t]
        for t in TEST_THRESHOLDS
    )
    if all_match:
        print("DECISION: (b) reproduces CLAUDE.md's historical group counts "
              "at every tested threshold. Hypothesis CONFIRMED: the "
              "historical phage grouping used neighbourhood-assignment, not "
              "union-find, and permitted leakage above the stated threshold "
              "(see the leakage column). Stop here per GROUPING_INVESTIGATION.md.")
    else:
        print("DECISION: (b) does NOT reproduce CLAUDE.md's historical group "
              "counts at every tested threshold (see table above). Hypothesis "
              "NOT confirmed as stated. Report this and stop -- do not guess "
              "at further variants.")


if __name__ == "__main__":
    main()
