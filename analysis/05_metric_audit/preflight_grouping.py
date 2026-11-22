"""Pre-flight for the extended phage-held-out sweep.

Answers one question without training anything: at each candidate RBP-identity
threshold, how many phage folds survive, and how big is the largest one?

A threshold is usable only if there are enough groups to hold one out and still
train on the rest. When the largest group approaches 105 phages, every phage has
chained into a single fold and the sweep cannot run at that threshold.

Run from the repo root:  python preflight_grouping.py
"""

import itertools
import numpy as np
import pandas as pd

try:
    from rapidfuzz.distance import Levenshtein
    identity = Levenshtein.normalized_similarity
except ImportError:
    import difflib
    print("  rapidfuzz not installed; falling back to difflib (much slower, "
          "and NOT the same metric 2d_phage_holdout.py uses as a fallback "
          "approximation -- pip install rapidfuzz for numbers that match "
          "the real sweep)")

    def identity(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio()

# --- CONFIG: match to your repo -------------------------------------------
RBP_FILE = "data/RBPbase.csv"      # columns: phage_ID, protein_sequence
COL_PHAGE = "phage_ID"
COL_SEQ = "protein_sequence"
THRESHOLDS = [1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
# RBPbase.csv currently has no sequence outside 214-1477 aa, so this filter
# is a no-op today; it is NOT applied by 2d_phage_holdout.py's own clustering
# (cluster_phages_by_identity reads RBPbase.csv unfiltered). Kept only so the
# RBP count matches what you'd see if the file ever gains out-of-range rows.
MIN_LEN, MAX_LEN = 200, 1500
# --------------------------------------------------------------------------


def main():
    df = pd.read_csv(RBP_FILE)
    df = df[df[COL_SEQ].str.len().between(MIN_LEN, MAX_LEN)]
    phages = df[COL_PHAGE].tolist()
    seqs = df[COL_SEQ].tolist()
    n_rbp = len(seqs)
    n_phage = len(set(phages))
    print(f"{n_rbp} RBPs across {n_phage} phages\n")

    # pairwise identity, once
    print("computing pairwise identities ...")
    pid = np.ones((n_rbp, n_rbp))
    for i, j in itertools.combinations(range(n_rbp), 2):
        v = identity(seqs[i], seqs[j])
        pid[i, j] = pid[j, i] = v

    off = pid[np.triu_indices(n_rbp, k=1)]
    print(f"identity distribution: min {off.min():.3f}  median {np.median(off):.3f}  "
          f"90th pct {np.percentile(off, 90):.3f}  max {off.max():.3f}\n")

    # single-linkage merge of PHAGES via union-find
    uniq = sorted(set(phages))
    idx = {p: k for k, p in enumerate(uniq)}

    print(f"{'thresh':>8}{'groups':>9}{'largest':>9}{'singletons':>12}  status")
    for t in THRESHOLDS:
        parent = list(range(len(uniq)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j in zip(*np.where(np.triu(pid >= t, k=1))):
            if phages[i] != phages[j]:
                union(idx[phages[i]], idx[phages[j]])

        roots = [find(k) for k in range(len(uniq))]
        sizes = pd.Series(roots).value_counts()
        n_groups, largest = len(sizes), sizes.max()
        singles = int((sizes == 1).sum())

        if n_groups < 3:
            status = "UNUSABLE - everything chained together"
        elif largest > 0.5 * n_phage:
            status = "degenerate - one fold holds most phages"
        elif n_groups < 10:
            status = "marginal - few folds, wide CIs"
        else:
            status = "ok"
        print(f"{t:>8.2f}{n_groups:>9}{largest:>9}{singles:>12}  {status}")


if __name__ == "__main__":
    main()
