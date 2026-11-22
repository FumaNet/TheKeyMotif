"""Test 3 (user-directed follow-up to GROUPING_INVESTIGATION.md).

Hypothesis: every historical phage-identity-sweep run predates today's
`pip install rapidfuzz` in the keymotif env, so CLAUDE.md's established group
counts (99/89/80/69 at 100/95/90/80%) may have been produced by
cluster_phages_by_identity()'s SILENT difflib fallback, not rapidfuzz's
Levenshtein.normalized_similarity.

This replicates that fallback exactly as written in
analysis/02_ablation/2d_phage_holdout.py:

    def sim(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio()

-- isjunk=None, autojunk left at its default (True). autojunk matters here:
every RBP sequence is well over the 200-element threshold where
SequenceMatcher starts treating elements appearing in >1% of one sequence as
junk, which measurably changes ratio() on sequences this long. Not overridden
in the real code, so not overridden here.

Read-only. Does NOT modify 2d_phage_holdout.py. No GPU, no training.
Reuses cluster_union_find / cluster_neighbourhood_assignment / leakage_count
from grouping_method_compare.py (Test 1) rather than redefining them.
"""

import difflib
import itertools
import time

import numpy as np
import pandas as pd

from grouping_method_compare import (
    cluster_union_find,
    cluster_neighbourhood_assignment,
    leakage_count,
)

RBP_FILE = "data/RBPbase.csv"
COL_PHAGE = "phage_ID"
COL_SEQ = "protein_sequence"
TEST_THRESHOLDS = [1.00, 0.95, 0.90, 0.80]
HISTORICAL = {1.00: 99, 0.95: 89, 0.90: 80, 0.80: 69}


def difflib_identity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def build_phage_identity_matrix_difflib():
    df = pd.read_csv(RBP_FILE)
    phages = df[COL_PHAGE].tolist()
    seqs = df[COL_SEQ].tolist()
    uniq = sorted(set(phages))
    idx = {p: k for k, p in enumerate(uniq)}
    n = len(uniq)
    n_rbp = len(seqs)
    n_pairs = n_rbp * (n_rbp - 1) // 2

    sample = list(itertools.islice(itertools.combinations(range(n_rbp), 2), 200))
    t0 = time.time()
    for i, j in sample:
        difflib_identity(seqs[i], seqs[j])
    per_pair = (time.time() - t0) / len(sample)
    est_min = per_pair * n_pairs / 60
    print(f"{n_rbp} RBPs -> {n_pairs:,} pairs. Calibrated on {len(sample)} pairs: "
          f"{per_pair*1000:.2f} ms/pair -> estimated {est_min:.1f} min total. "
          f"Starting the full matrix now (not hung, just slow).", flush=True)

    mat = np.zeros((n, n))
    np.fill_diagonal(mat, 1.0)
    t0 = time.time()
    done = 0
    for i in range(n_rbp):
        for j in range(i + 1, n_rbp):
            pi, pj = idx[phages[i]], idx[phages[j]]
            done += 1
            if pi != pj:
                v = difflib_identity(seqs[i], seqs[j])
                if v > mat[pi, pj]:
                    mat[pi, pj] = mat[pj, pi] = v
            if done % 5000 == 0:
                el = time.time() - t0
                remaining = el / done * (n_pairs - done)
                print(f"  {done:,}/{n_pairs:,} pairs, {el:.0f}s elapsed, "
                      f"~{remaining:.0f}s remaining", flush=True)
    print(f"matrix done in {time.time()-t0:.0f}s\n", flush=True)
    return uniq, mat


def main():
    uniq, mat = build_phage_identity_matrix_difflib()
    n = len(uniq)
    print(f"{n} phages, difflib-based continuous max-identity matrix built\n")

    print(f"{'thresh':>8}{'union-find':>14}{'CLAUDE.md':>12}{'match':>10}")
    uf_counts = {}
    for t in TEST_THRESHOLDS:
        groups_a = cluster_union_find(mat, t)
        n_a = len(set(groups_a))
        uf_counts[t] = n_a
        hist = HISTORICAL[t]
        print(f"{t:>8.2f}{n_a:>14}{hist:>12}{'YES' if n_a == hist else 'no':>10}")

    if all(uf_counts[t] == HISTORICAL[t] for t in TEST_THRESHOLDS):
        print("\nDECISION: difflib fallback + union-find reproduces CLAUDE.md's "
              "historical group counts exactly at every tested threshold. "
              "CONFIRMED. Report and stop.")
        return

    print("\nunion-find on the difflib matrix did not match at every threshold. "
          "Trying neighbourhood-assignment on the SAME difflib matrix, as "
          "instructed, before stopping.\n")
    print(f"{'thresh':>8}{'neighbourhood':>16}{'CLAUDE.md':>12}{'match':>10}{'leakage':>10}")
    nb_counts = {}
    for t in TEST_THRESHOLDS:
        groups_b = cluster_neighbourhood_assignment(mat, t)
        n_b = len(set(groups_b))
        nb_counts[t] = n_b
        leak_b = leakage_count(mat, t, groups_b)
        hist = HISTORICAL[t]
        print(f"{t:>8.2f}{n_b:>16}{hist:>12}{'YES' if n_b == hist else 'no':>10}"
              f"{leak_b:>10}")

    if all(nb_counts[t] == HISTORICAL[t] for t in TEST_THRESHOLDS):
        print("\nDECISION: difflib fallback + neighbourhood-assignment reproduces "
              "CLAUDE.md's historical group counts exactly at every tested "
              "threshold. CONFIRMED. Report and stop.")
    else:
        print("\nDECISION: neither combination (difflib+union-find, "
              "difflib+neighbourhood-assignment) reproduces CLAUDE.md's historical "
              "counts at every threshold. Reporting both tables above. Stopping -- "
              "not guessing at further variants.")


if __name__ == "__main__":
    main()
