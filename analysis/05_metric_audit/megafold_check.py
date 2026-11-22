"""User-directed follow-up after the dedup + rapidfuzz sweep rerun
(id100/95/90/80/70/60/50/40, all now on the same normalised-Levenshtein
metric, option (a) confirmed).

For id70: reconstruct which pooled (accession, phage_ID) rows in
results/2d_phage_holdout_id70.pkl belong to which LeaveOneGroupOut fold
(the pickle stores only concatenated labels/scores/auc, not a fold id per
row), validate that reconstruction against the pickle by exact label
equality (same check used earlier in this investigation for the dedup
control), then split ROC-AUC / PR-AUC into "largest fold alone" vs "every
other fold pooled", plus the largest fold's share of all pooled
predictions.

Then repeat the same reconstruction+split for id60/id50/id40 and report
whether the largest-fold ROC/PR-AUC value is IDENTICAL across all four --
expected if and only if that specific merged group's phage membership is
unchanged across 70/60/50/40% (lowering the threshold only adds merges
elsewhere; if none of them touch this group, its held-out rows -- and
therefore its train/test split and, under a fixed seed, its model output --
are bit-for-bit the same run to run).

Read-only. No GPU, no training. Mirrors 2d_phage_holdout.py's own grouping,
dedup and pooling logic exactly; does not import or modify it.
"""

import pickle
import sys

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import auc, precision_recall_curve, roc_curve

sys.path.insert(0, "src")
import keymotif_data as kd

RBP_FILE = "data/RBPbase.csv"
THRESHOLDS = [0.70, 0.60, 0.50, 0.40]


def cluster_phages_by_identity(threshold, pairs_phage_ids):
    """Verbatim copy of 2d_phage_holdout.py's clustering (rapidfuzz branch
    only -- it's installed, so that's the branch the real run took)."""
    sim = Levenshtein.normalized_similarity
    rbp = pd.read_csv(RBP_FILE)
    low = {c.lower(): c for c in rbp.columns}
    ph_col = next(low[c] for c in ["phage_id", "phage"] if c in low)
    sq_col = next(low[c] for c in
                  ["protein_sequence", "protein_seq", "sequence"] if c in low)
    rbp = rbp.rename(columns={ph_col: "phage_ID", sq_col: "protein_sequence"})
    rbp = rbp.dropna(subset=["protein_sequence"]).reset_index(drop=True)

    seqs = rbp["protein_sequence"].tolist()
    phs = rbp["phage_ID"].tolist()
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for ph in set(phs):
        find(ph)

    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            if phs[i] == phs[j]:
                continue
            if sim(seqs[i], seqs[j]) >= threshold:
                union(phs[i], phs[j])

    groups = {ph: find(ph) for ph in set(phs)}
    labels = {r: k for k, r in enumerate(sorted(set(groups.values())))}
    return {ph: labels[r] for ph, r in groups.items()}


def build_pairs(threshold):
    pairs, _, _ = kd.load(verbose=False)
    pairs = pairs.copy()
    pg = cluster_phages_by_identity(threshold, pairs["phage_ID"])
    pairs["phage_group"] = pairs["phage_ID"].map(pg)
    pairs = pairs.dropna(subset=["phage_group"])
    before = len(pairs)
    pairs = pairs.drop_duplicates(
        subset=["accession", "phage_ID", "protein_ID", "virus_idx"]
    ).reset_index(drop=True)
    return pairs


def reconstruct_fold_ids(pairs):
    """Same groupby/pool, same LOGO iteration order (ascending unique group
    id) as the real run. Returns (label_array, fold_id_array, group_members)
    in the exact concatenation order the pickle was written in."""
    groups_sorted = sorted(pairs["phage_group"].unique())
    labels_out, fold_ids_out = [], []
    group_members = {}  # fold_id -> set of original phage_ID in this group
    for fold_id, g in enumerate(groups_sorted):
        sub_te = pairs[pairs["phage_group"] == g]
        group_members[fold_id] = set(sub_te["phage_ID"].unique())
        mx = sub_te.groupby(["accession", "phage_ID"]).agg(
            {"label": "first"}).reset_index()
        labels_out.append(mx["label"].astype(int).values)
        fold_ids_out.append(np.full(len(mx), fold_id))
    return np.concatenate(labels_out), np.concatenate(fold_ids_out), group_members


def analyse(threshold, tag):
    with open(f"results/2d_phage_holdout_id{tag}.pkl", "rb") as f:
        label_pkl, score_pkl, stored_roc = pickle.load(f)

    pairs = build_pairs(threshold)
    label_rec, fold_ids, group_members = reconstruct_fold_ids(pairs)

    n_groups = len(group_members)
    print(f"\n=== id{tag} (threshold {threshold:.0%}) ===")
    print(f"reconstructed groups/folds: {n_groups}  "
          f"pooled rows reconstructed: {len(label_rec)}  "
          f"pooled rows in pickle: {len(label_pkl)}")

    if len(label_rec) != len(label_pkl):
        print("MISMATCH IN ROW COUNT -- cannot validate further. Stopping for this id.")
        return None

    identical = np.array_equal(label_rec, label_pkl)
    n_diff = int((label_rec != label_pkl).sum())
    print(f"label-equality validation: {'IDENTICAL' if identical else f'{n_diff} differ'} "
          f"across all {len(label_pkl)} positions")
    if not identical:
        print("Reconstruction does not match the pickle's row order/labels -- "
              "stopping for this id, not reporting a split on an unvalidated mapping.")
        return None

    # largest fold by pooled-row count
    sizes = np.bincount(fold_ids, minlength=n_groups)
    mega_fold = int(np.argmax(sizes))
    mega_n = int(sizes[mega_fold])
    mega_members = group_members[mega_fold]

    mega_mask = fold_ids == mega_fold
    rest_mask = ~mega_mask

    def roc_pr(mask):
        y, s = label_pkl[mask], score_pkl[mask]
        if len(set(y.tolist())) < 2:
            return float("nan"), float("nan")
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        return auc(fpr, tpr), auc(rec, prec)

    roc_mega, pr_mega = roc_pr(mega_mask)
    roc_rest, pr_rest = roc_pr(rest_mask)

    print(f"largest fold: fold_id={mega_fold}, {len(mega_members)} merged phages, "
          f"{mega_n} pooled predictions "
          f"({100*mega_n/len(label_pkl):.1f}% of all {len(label_pkl)} predictions)")
    print(f"  largest fold alone   : ROC-AUC {roc_mega:.4f}  PR-AUC {pr_mega:.4f}  "
          f"(n={mega_n}, pos={int(label_pkl[mega_mask].sum())})")
    print(f"  all other folds pooled: ROC-AUC {roc_rest:.4f}  PR-AUC {pr_rest:.4f}  "
          f"(n={int(rest_mask.sum())}, pos={int(label_pkl[rest_mask].sum())})")

    return {
        "roc_mega": roc_mega, "pr_mega": pr_mega,
        "roc_rest": roc_rest, "pr_rest": pr_rest,
        "mega_n": mega_n, "share": mega_n / len(label_pkl),
        "mega_members": mega_members,
    }


def main():
    tags = {0.70: "70", 0.60: "60", 0.50: "50", 0.40: "40"}
    results = {}
    for t in THRESHOLDS:
        results[t] = analyse(t, tags[t])

    print("\n=== cross-threshold check ===")
    base = results[0.70]
    if base is None:
        print("id70 failed validation; cannot compare.")
        return
    base_members = base["mega_members"]
    print(f"id70 mega-fold members ({len(base_members)}): {sorted(base_members)}")

    all_identical = True
    for t in [0.60, 0.50, 0.40]:
        r = results[t]
        tag = tags[t]
        if r is None:
            print(f"id{tag}: failed validation, skipped.")
            all_identical = False
            continue
        same_members = r["mega_members"] == base_members
        same_roc = np.isclose(r["roc_mega"], base["roc_mega"], atol=0, rtol=0) or \
            r["roc_mega"] == base["roc_mega"]
        same_pr = r["pr_mega"] == base["pr_mega"]
        print(f"id{tag}: mega-fold members {'SAME' if same_members else 'DIFFERENT'} "
              f"as id70 ({len(r['mega_members'])} members); "
              f"ROC {'identical' if same_roc else 'differs'} "
              f"({r['roc_mega']:.4f} vs {base['roc_mega']:.4f}); "
              f"PR-AUC {'identical' if same_pr else 'differs'} "
              f"({r['pr_mega']:.4f} vs {base['pr_mega']:.4f})")
        if not (same_members and same_roc and same_pr):
            all_identical = False

    print()
    if all_identical:
        print("CONFIRMED: the largest merged group is unchanged across "
              "70/60/50/40%, and its held-out ROC-AUC/PR-AUC are bit-identical "
              "at every threshold -- consistent with a fixed random_state and "
              "identical train/test rows for that fold regardless of how the "
              "OTHER groups are further split.")
    else:
        print("NOT fully confirmed -- see per-threshold lines above for where "
              "membership or the AUC values diverge.")


if __name__ == "__main__":
    main()
