"""
2d_phage_holdout.py — hold out PHAGES instead of bacteria.

THE QUESTION
------------
Reviewer #1's third alternative: does PHL-RBP+S learn transferable RBP
features, or does it memorise phage-serotype associations?

The published LOGO protocol splits BACTERIA. Every phage therefore appears in
every training fold. A model can score a held-out (host, phage) pair correctly
by recalling "this phage infected K64 hosts in training" and noting that the
held-out host is also K64 -- without any transferable understanding of the RBP
sequence. Nothing in the bacterial split can detect that.

This script runs the SAME model (RBP embedding + one-hot serotype, max
aggregation) under a PHAGE split. Every test pair involves a phage the model has
never seen. Interpretation:

    AUC holds near 0.817   -> genuine RBP feature learning; the model
                              generalises to unseen phages
    AUC collapses to ~0.5  -> the bacterial-split number rests substantially on
                              phage-identity recall, and the interpretability
                              claims need heavy qualification

Either outcome is publishable and neither is available from the existing
experiments. This is the decisive test.

COST
----
One split, not eight. The bacterial similarity thresholds are a property of the
bacterial grouping and are irrelevant here, so this is 105 folds (or fewer with
clustering) -- roughly one threshold's worth of script 3, not eight.

PHAGE REDUNDANCY
----------------
Plain leave-one-phage-out has the same weakness the bacterial thresholds were
introduced to fix: near-identical phages may sit on both sides of the split. Use
--cluster-rbp to merge phages sharing an identical RBP sequence (union-find)
before splitting, which is the phage-side analogue of the 100% bacterial
threshold. Report both; the gap between them is informative.

    python 2d_phage_holdout.py --dry-run
    python 2d_phage_holdout.py
    python 2d_phage_holdout.py --cluster-rbp

Writes Results/2d_phage_holdout[_clustered].pkl as a single
(labels, scores, auc) tuple -- NOT the 8-threshold list format, since there is
only one split. verify_reproduction.py will skip it; read it directly.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
from xgboost import XGBClassifier

import keymotif_data as kd

DEVICE = os.environ.get("KM_DEVICE", "cuda")


def cluster_phages_by_rbp(rbp_csv="Data/RBPbase.csv"):
    """
    Union-find: phages sharing an identical RBP sequence land in one group.

    The phage-side analogue of the 100% bacterial similarity threshold. Coarse
    (exact match only, no alignment), but it removes the most blatant leakage
    -- two phages carrying literally the same adsorption protein.
    """
    if not os.path.exists(rbp_csv):
        raise SystemExit(
            f"--cluster-rbp needs {rbp_csv} (RBP sequences), which is in the\n"
            "PhageHostLearn Zenodo record alongside the embedding files.\n"
            "Without it, run plain leave-one-phage-out (omit --cluster-rbp)\n"
            "and note the lack of phage-redundancy control as a caveat.")
    rbp = pd.read_csv(rbp_csv)

    # Column names differ between the Zenodo RBPbase.csv and the repo CSVs
    # (protein_sequence vs protein_seq vs sequence). Detect, do not assume.
    def pick(cands, what):
        low = {c.lower(): c for c in rbp.columns}
        for c in cands:
            if c in low:
                return low[c]
        raise SystemExit(
            f"{rbp_csv}: could not find a {what} column.\n"
            f"  looked for: {cands}\n"
            f"  file has:   {list(rbp.columns)[:12]}\n"
            "Fix the file, or run without --cluster-rbp.")

    ph_col = pick(["phage_id", "phage", "phageid"], "phage id")
    sq_col = pick(["protein_sequence", "protein_seq", "sequence", "seq",
                   "aa_sequence"], "protein sequence")
    rbp = rbp.rename(columns={ph_col: "phage_ID", sq_col: "protein_sequence"})
    rbp = rbp.dropna(subset=["protein_sequence"])
    print(f"  clustering on '{sq_col}' from {rbp_csv} ({len(rbp):,} RBPs)")

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

    for ph in rbp["phage_ID"].unique():
        find(ph)
    for _, g in rbp.groupby("protein_sequence"):
        phs = g["phage_ID"].unique()
        for other in phs[1:]:
            union(phs[0], other)

    groups = {ph: find(ph) for ph in rbp["phage_ID"].unique()}
    labels = {r: i for i, r in enumerate(sorted(set(groups.values())))}
    return {ph: labels[r] for ph, r in groups.items()}


def cluster_phages_by_identity(threshold, rbp_csv="Data/RBPbase.csv"):
    """
    Merge phages whose RBPs exceed a sequence-identity threshold.

    WHY THIS EXISTS
    ---------------
    The bacterial side of this study sweeps K-locus similarity from 100% down
    to 75%, so held-out bacteria become progressively less similar to anything
    in training. The phage side, with exact-match clustering only, has no
    equivalent: two phages whose RBPs differ by a single residue land in
    different folds, which is precisely the leakage the bacterial sweep exists
    to control.

    This restores the symmetry. Two phages are merged (union-find) if ANY pair
    of their RBPs reaches `threshold` normalised Levenshtein similarity.
    threshold=1.0 reproduces exact-match clustering.

    ~37k pairwise comparisons for 274 RBPs: about a second with rapidfuzz.
    """
    if not os.path.exists(rbp_csv):
        raise SystemExit(f"--cluster-identity needs {rbp_csv} (RBP sequences).")
    try:
        from rapidfuzz.distance import Levenshtein
        sim = Levenshtein.normalized_similarity
    except ImportError:
        import difflib
        print("  rapidfuzz not installed; falling back to difflib (much slower)")
        print("  pip install rapidfuzz")

        def sim(a, b):
            return difflib.SequenceMatcher(None, a, b).ratio()

    rbp = pd.read_csv(rbp_csv)
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

    n_merge = 0
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            if phs[i] == phs[j]:
                continue
            if sim(seqs[i], seqs[j]) >= threshold:
                if find(phs[i]) != find(phs[j]):
                    n_merge += 1
                union(phs[i], phs[j])

    groups = {ph: find(ph) for ph in set(phs)}
    labels = {r: k for k, r in enumerate(sorted(set(groups.values())))}
    print(f"  identity clustering at {threshold:.0%}: {len(set(phs))} phages -> "
          f"{len(labels)} groups ({n_merge} merges, {len(rbp):,} RBPs compared)")
    return {ph: labels[r] for ph, r in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-rbp", action="store_true",
                    help="Merge phages sharing an identical RBP before splitting.")
    ap.add_argument("--cluster-identity", type=float, metavar="THRESH",
                    help="Merge phages whose RBPs reach THRESH sequence "
                         "identity (e.g. 0.95). The phage-side analogue of the "
                         "bacterial similarity sweep. Overrides --cluster-rbp.")
    ap.add_argument("--no-serotype", action="store_true",
                    help="Drop the serotype one-hot: RBP embedding only. "
                         "Completes the 2x2 with 2c_rbp_only.py.")
    ap.add_argument("--dedup", action="store_true",
                    help="Deduplicate to unique (accession, phage, RBP) rows "
                         "(25k instead of 487k). Much faster; use with "
                         "--no-serotype to match 2c's row weighting.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.cluster_identity is not None:
        tag = f"_id{int(args.cluster_identity*100)}"
    elif args.cluster_rbp:
        tag = "_clustered"
    else:
        tag = ""
    tag += "_noserotype" if args.no_serotype else ""
    OUT = f"Results/2d_phage_holdout{tag}.pkl"

    pairs, host_emb, virus_emb = kd.load()

    pairs = pairs.copy()
    if args.cluster_identity is not None:
        pg = cluster_phages_by_identity(args.cluster_identity)
        how = f"{args.cluster_identity:.0%} RBP identity"
    elif args.cluster_rbp:
        pg = cluster_phages_by_rbp()
        how = "identical RBP"
    else:
        pg = None

    if pg is None:
        pairs["phage_group"] = pairs["phage_ID"]
        print(f"Leave-one-phage-out: {pairs['phage_ID'].nunique()} phages, "
              "no redundancy control (see --cluster-rbp / --cluster-identity)")
    else:
        pairs["phage_group"] = pairs["phage_ID"].map(pg)
        n_missing = int(pairs["phage_group"].isna().sum())
        if n_missing:
            print(f"  warning: {n_missing:,} rows whose phage is absent from "
                  "RBPbase.csv; dropping them")
            pairs = pairs.dropna(subset=["phage_group"])
        print(f"Phage clustering: {pairs['phage_ID'].nunique()} phages -> "
              f"{pairs['phage_group'].nunique()} groups (merged by {how})")

    if args.dedup:
        before = len(pairs)
        pairs = pairs.drop_duplicates(
            subset=["accession", "phage_ID", "protein_ID", "virus_idx"]
        ).reset_index(drop=True)
        print(f"Deduplicated: {before:,} -> {len(pairs):,} rows "
              "(matches 2c_rbp_only.py weighting)")

    sero_encoded = sero_cols = None
    if not args.no_serotype:
        df_sero = pd.read_csv("Data/kaptive_results.tsv", sep="\t")
        df_sero = df_sero[["Assembly", "Best match type", "Match confidence"]]
        one_hot = pd.get_dummies(df_sero["Best match type"], prefix="sero_")
        sero_encoded = pd.concat([df_sero[["Assembly"]], one_hot], axis=1)
        sero_cols = [c for c in sero_encoded.columns if c != "Assembly"]

    n_folds = pairs["phage_group"].nunique()
    print(f"Rows: {len(pairs):,}   folds: {n_folds}")
    if args.no_serotype:
        print("Features: RBP embedding ONLY — no host information at all.")
        print("  This asks whether transferable signal is capsule MATCHING or")
        print("  merely phage promiscuity predicted from RBP sequence.")
    else:
        print(f"Features: RBP embedding + {len(sero_cols)} serotype columns "
              "(identical to PHL-RBP+S)")
    print(f"Output: {OUT}")

    if args.dry_run:
        sizes = pairs.groupby("phage_group").size()
        print(f"\nfold sizes: min {sizes.min():,}  median "
              f"{int(sizes.median()):,}  max {sizes.max():,}")
        pos = pairs[pairs["label"] == 1].groupby("phage_group").size()
        print(f"folds with zero positives in test: "
              f"{n_folds - len(pos)} of {n_folds}")
        print("\nDry run. Re-run without --dry-run to execute.")
        return

    logo = LeaveOneGroupOut()
    scores_max, label_max = [], []
    pbar = tqdm(total=n_folds, desc="phage-holdout CV")

    for tr, te in logo.split(pairs, pairs["label"], pairs["phage_group"]):
        if args.no_serotype:
            sub_tr, sub_te = pairs.iloc[tr], pairs.iloc[te]
            X_train = kd.make_X(sub_tr, host_emb, virus_emb, mode="virus")
            X_test = kd.make_X(sub_te, host_emb, virus_emb, mode="virus")
        else:
            sub_tr, S_tr = kd.attach_serotype(pairs.iloc[tr], sero_encoded, sero_cols)
            sub_te, S_te = kd.attach_serotype(pairs.iloc[te], sero_encoded, sero_cols)
            X_train = kd.make_X(sub_tr, host_emb, virus_emb, mode="virus", sero=S_tr)
            X_test = kd.make_X(sub_te, host_emb, virus_emb, mode="virus", sero=S_te)
        y_train = sub_tr["label"].astype(int).values
        y_test = sub_te["label"].astype(int).values

        if len(set(y_train)) < 2:
            pbar.update(1)
            continue

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        imbalance = n_pos / n_neg if n_neg else 1

        xgb = XGBClassifier(
            scale_pos_weight=1 / imbalance,
            learning_rate=0.3, n_estimators=250, max_depth=7,
            eval_metric="logloss", tree_method="hist", device=DEVICE,
        )
        xgb.fit(X_train, y_train)
        score = xgb.predict_proba(X_test)[:, 1]

        df_preds = pd.DataFrame({
            "accession": sub_te["accession"].values,
            "phage_ID": sub_te["phage_ID"].values,
            "true_label": y_test, "score": score,
        })
        mx = df_preds.groupby(["accession", "phage_ID"]).agg(
            {"score": "max", "true_label": "first"}).reset_index()

        scores_max.append(mx["score"].values)
        label_max.append(mx["true_label"].values)
        pbar.update(1)

    pbar.close()

    scores_max = np.concatenate(scores_max)
    label_max = np.concatenate(label_max)
    fpr, tpr, _ = roc_curve(label_max, scores_max)
    prec, rec, _ = precision_recall_curve(label_max, scores_max)
    rauclr = round(auc(fpr, tpr), 3)
    pr = auc(rec, prec)

    os.makedirs("Results", exist_ok=True)
    with open(OUT, "wb") as fh:
        pickle.dump((label_max, scores_max, rauclr), fh)

    print(f"\n{'='*62}")
    print(f"PHAGE-HELD-OUT AUC: {rauclr}    PR-AUC: {pr:.3f}")
    print(f"{'='*62}")
    print("  THE 2x2:                bacterial split   phage split")
    print(f"    RBP + serotype              0.817          "
          f"{'%.3f' % rauclr if not args.no_serotype else '0.749'}")
    print(f"    RBP only                    0.701          "
          f"{'%.3f' % rauclr if args.no_serotype else '    ?'}")
    print("    serotype only               0.589              n/a")
    print(f"  dataset positive rate:      0.033")
    print()
    if args.no_serotype:
        print("  Compare against the phage-split RBP+serotype run (0.749).")
        print("  A small gap means the transferable signal is largely phage")
        print("  promiscuity predicted from sequence, not capsule matching.")
        print("  A large gap means genuine matching that generalises.")
        print()
    if rauclr < 0.60:
        print("  -> Collapse. The bacterial-split number rests substantially on")
        print("     phage-identity recall. Report this; it reframes the paper.")
    elif rauclr < 0.72:
        print("  -> Partial. Some transferable RBP signal, but a large share of")
        print("     the bacterial-split performance does not survive unseen phages.")
    else:
        print("  -> Holds up. The model generalises to unseen phages, which")
        print("     answers Reviewer #1's memorisation concern directly.")
    print(f"\n  Wrote {OUT}")


if __name__ == "__main__":
    main()
