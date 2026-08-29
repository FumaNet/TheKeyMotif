#!/usr/bin/env python3
"""
test_equivalence.py — prove the compact data layer gives identical results.

Generates a synthetic dataset with exactly the schema of the Zenodo inputs,
then runs (a) the ORIGINAL wide-frame logic lifted verbatim from
0_original_replica.py and 3_max_max_sero.py, and (b) the keymotif_data
equivalent, and asserts the scores match.

Run from the repo root:  python test_equivalence.py
"""

import os
import pickle
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier

DIM = 24          # stand-in for 1280; the code is dimension-agnostic
N_ACC = 24
N_PHAGE = 12
SEED = 0
DEVICE = os.environ.get("KM_DEVICE", "cpu")


# ---------------------------------------------------------------------------
# synthetic data with the real schema
# ---------------------------------------------------------------------------

def make_synthetic(root):
    rng = np.random.default_rng(SEED)
    data = os.path.join(root, "Data")
    os.makedirs(data, exist_ok=True)
    os.makedirs(os.path.join(root, "grouping"), exist_ok=True)

    accs = [f"ACC{i:03d}" for i in range(N_ACC)]
    phages = [f"PH{i:02d}" for i in range(N_PHAGE)]

    # loci: variable number of proteins per accession
    rows, idxs, embs = [], [], []
    for a in accs:
        for p in range(rng.integers(2, 6)):
            rows.append(a)
            idxs.append(p)
            embs.append(rng.normal(size=DIM))
    loci = pd.concat([pd.DataFrame({"accession": rows, "protein_index": idxs}),
                      pd.DataFrame(np.array(embs))], axis=1)
    loci.to_csv(os.path.join(data, "esm2_embeddings_loci_per_protein.csv"), index=False)

    # rbp: variable number of RBPs per phage
    prows, pids, pembs = [], [], []
    for ph in phages:
        for r in range(rng.integers(1, 5)):
            prows.append(ph)
            pids.append(f"{ph}_gp{r}")
            pembs.append(rng.normal(size=DIM))
    rbp = pd.concat([pd.DataFrame({"phage_ID": prows, "protein_ID": pids}),
                     pd.DataFrame(np.array(pembs))], axis=1)
    rbp.to_csv(os.path.join(data, "esm2_embeddings_rbp.csv"), index=False)

    # interaction matrix, ~12% positive so folds aren't degenerate
    mat = (rng.random((N_ACC, N_PHAGE)) < 0.12).astype(float)
    inter = pd.DataFrame(mat, columns=phages)
    inter.insert(0, "Unnamed: 0", accs)
    inter.to_csv(os.path.join(data, "phage_host_interactions.csv"), index=False)

    # grouping: pairs of accessions per group
    grouping = {a: i // 2 for i, a in enumerate(accs)}
    with open(os.path.join(root, "grouping", "grouping_1.pkl"), "wb") as fh:
        pickle.dump(grouping, fh)

    # kaptive-style serotypes
    sero = pd.DataFrame({
        "Assembly": accs,
        "Best match type": [f"K{rng.integers(1, 6)}" for _ in accs],
        "Match confidence": ["Typeable"] * N_ACC,
    })
    sero.to_csv(os.path.join(data, "kaptive_results.tsv"), sep="\t", index=False)
    return grouping


def clf():
    return XGBClassifier(learning_rate=0.3, n_estimators=40, max_depth=4,
                         eval_metric="logloss", tree_method="hist",
                         device=DEVICE, random_state=SEED)


# ---------------------------------------------------------------------------
# ORIGINAL logic (lifted from the repo scripts)
# ---------------------------------------------------------------------------

def original_wide_frame():
    loci = pd.read_csv("Data/esm2_embeddings_loci_per_protein.csv")
    rbp = pd.read_csv("Data/esm2_embeddings_rbp.csv")
    inter = pd.read_csv("Data/phage_host_interactions.csv")

    melted = inter.melt(id_vars=["Unnamed: 0"], var_name="phage_ID",
                        value_name="label").rename(columns={"Unnamed: 0": "accession"})
    melted = melted.dropna(subset=["label"])

    merged = melted.merge(loci, on="accession", how="inner")
    merged = merged.merge(rbp, on="phage_ID", how="inner")

    hcols = [c for c in merged.columns
             if c not in ["accession", "phage_ID", "protein_index", "protein_ID", "label"]
             and "_x" in c]
    vcols = [c for c in merged.columns
             if c not in ["accession", "phage_ID", "protein_index", "protein_ID", "label"]
             and "_y" in c]

    h = merged[hcols].astype(np.float32)
    v = merged[vcols].astype(np.float32)
    h.columns = [f"host_{i}" for i in range(h.shape[1])]
    v.columns = [f"virus_{i}" for i in range(v.shape[1])]

    return pd.concat([merged[["accession", "phage_ID", "protein_index", "protein_ID"]],
                      h, v, merged[["label"]]], axis=1)


def original_script0(final_df, grouping):
    final_df = final_df.copy()
    final_df["group_loci"] = final_df["accession"].map(grouping)
    logo = LeaveOneGroupOut()
    scores, labels = [], []

    for tr, te in logo.split(final_df, final_df["label"], final_df["group_loci"]):
        train_df, test_df = final_df.iloc[tr].copy(), final_df.iloc[te].copy()
        for d in (train_df, test_df):
            pass
        train_df = train_df.groupby(["accession", "phage_ID"]).agg({
            **{c: "mean" for c in train_df.columns if c.startswith(("host_", "virus_"))},
            "label": "first", "protein_ID": "first"}).reset_index()
        test_df = test_df.groupby(["accession", "phage_ID"]).agg({
            **{c: "mean" for c in test_df.columns if c.startswith(("host_", "virus_"))},
            "label": "first", "protein_ID": "first"}).reset_index()

        fc = [c for c in train_df.columns if c.startswith(("host_", "virus_"))]
        X_tr, y_tr = train_df[fc].values, train_df["label"].astype(int).values
        X_te, y_te = test_df[fc].values, test_df["label"].astype(int).values
        if len(set(y_tr)) < 2:
            continue
        m = clf()
        m.fit(X_tr, y_tr)
        s = m.predict_proba(X_te)[:, 1]
        dfp = pd.DataFrame({"accession": test_df["accession"],
                            "phage_ID": test_df["phage_ID"],
                            "true_label": y_te, "score": s})
        mx = dfp.groupby(["accession", "phage_ID"]).agg(
            {"score": "max", "true_label": "first"}).reset_index()
        scores.append(mx["score"].values)
        labels.append(mx["true_label"].values)

    return np.concatenate(labels), np.concatenate(scores)


def original_script3(final_df, grouping):
    df_sero = pd.read_csv("Data/kaptive_results.tsv", sep="\t")[
        ["Assembly", "Best match type", "Match confidence"]]
    final_df = final_df.copy()
    final_df["group_loci"] = final_df["accession"].map(grouping)
    logo = LeaveOneGroupOut()
    scores, labels = [], []

    for tr, te in logo.split(final_df, final_df["label"], final_df["group_loci"]):
        train_df, test_df = final_df.iloc[tr].copy(), final_df.iloc[te].copy()
        hc = [c for c in train_df.columns if c.startswith("host_")]
        train_df = train_df.drop(columns=hc)
        test_df = test_df.drop(columns=hc)

        oh = pd.get_dummies(df_sero["Best match type"], prefix="sero_")
        enc = pd.concat([df_sero[["Assembly"]], oh], axis=1)
        train_df = train_df.merge(enc, how="left", left_on="accession",
                                  right_on="Assembly").drop(columns=["Assembly"])
        test_df = test_df.merge(enc, how="left", left_on="accession",
                                right_on="Assembly").drop(columns=["Assembly"])
        train_df.fillna(0, inplace=True)
        test_df.fillna(0, inplace=True)
        train_df.drop_duplicates(inplace=True)
        test_df.drop_duplicates(inplace=True)

        fc = [c for c in train_df.columns if c.startswith(("sero_", "virus_"))]
        X_tr, y_tr = train_df[fc].values, train_df["label"].astype(int).values
        X_te, y_te = test_df[fc].values, test_df["label"].astype(int).values
        if len(set(y_tr)) < 2:
            continue
        m = clf()
        m.fit(X_tr, y_tr)
        s = m.predict_proba(X_te)[:, 1]
        dfp = pd.DataFrame({"accession": test_df["accession"],
                            "phage_ID": test_df["phage_ID"],
                            "true_label": y_te, "score": s})
        mx = dfp.groupby(["accession", "phage_ID"]).agg(
            {"score": "max", "true_label": "first"}).reset_index()
        scores.append(mx["score"].values)
        labels.append(mx["true_label"].values)

    return np.concatenate(labels), np.concatenate(scores)


# ---------------------------------------------------------------------------
# COMPACT logic
# ---------------------------------------------------------------------------

def compact_script0(kd, pairs, host_emb, virus_emb, grouping):
    pairs = pairs.copy()
    pairs["group_loci"] = pairs["accession"].map(grouping)
    logo = LeaveOneGroupOut()
    scores, labels = [], []

    for tr, te in logo.split(pairs, pairs["label"], pairs["group_loci"]):
        meta_tr, X_tr = kd.collapse_averaged_exact(pairs.iloc[tr], host_emb, virus_emb)
        meta_te, X_te = kd.collapse_averaged_exact(pairs.iloc[te], host_emb, virus_emb)
        y_tr = meta_tr["label"].astype(int).values
        y_te = meta_te["label"].astype(int).values
        if len(set(y_tr)) < 2:
            continue
        m = clf()
        m.fit(X_tr, y_tr)
        s = m.predict_proba(X_te)[:, 1]
        dfp = pd.DataFrame({"accession": meta_te["accession"],
                            "phage_ID": meta_te["phage_ID"],
                            "true_label": y_te, "score": s})
        mx = dfp.groupby(["accession", "phage_ID"]).agg(
            {"score": "max", "true_label": "first"}).reset_index()
        scores.append(mx["score"].values)
        labels.append(mx["true_label"].values)

    return np.concatenate(labels), np.concatenate(scores)


def compact_script3(kd, pairs, host_emb, virus_emb, grouping):
    df_sero = pd.read_csv("Data/kaptive_results.tsv", sep="\t")[
        ["Assembly", "Best match type", "Match confidence"]]
    oh = pd.get_dummies(df_sero["Best match type"], prefix="sero_")
    enc = pd.concat([df_sero[["Assembly"]], oh], axis=1)
    sero_cols = [c for c in enc.columns if c != "Assembly"]

    pairs = pairs.copy()
    pairs["group_loci"] = pairs["accession"].map(grouping)
    logo = LeaveOneGroupOut()
    scores, labels = [], []

    for tr, te in logo.split(pairs, pairs["label"], pairs["group_loci"]):
        out = []
        for idx in (tr, te):
            # NO dedup: the original drop_duplicates() is a no-op because the
            # host protein_index column survives. See attach_serotype docstring.
            sub, S = kd.attach_serotype(pairs.iloc[idx], enc, sero_cols)
            X = kd.make_X(sub, host_emb, virus_emb, mode="virus", sero=S)
            out.append((sub, X))

        (sub_tr, X_tr), (sub_te, X_te) = out
        y_tr = sub_tr["label"].astype(int).values
        y_te = sub_te["label"].astype(int).values
        if len(set(y_tr)) < 2:
            continue
        m = clf()
        m.fit(X_tr, y_tr)
        s = m.predict_proba(X_te)[:, 1]
        dfp = pd.DataFrame({"accession": sub_te["accession"],
                            "phage_ID": sub_te["phage_ID"],
                            "true_label": y_te, "score": s})
        mx = dfp.groupby(["accession", "phage_ID"]).agg(
            {"score": "max", "true_label": "first"}).reset_index()
        scores.append(mx["score"].values)
        labels.append(mx["true_label"].values)

    return np.concatenate(labels), np.concatenate(scores)


def report(name, a, b):
    la, sa = a
    lb, sb = b
    auc_a = auc(*roc_curve(la, sa)[:2])
    auc_b = auc(*roc_curve(lb, sb)[:2])
    same_labels = np.array_equal(np.sort(la), np.sort(lb))
    max_dev = np.max(np.abs(np.sort(sa) - np.sort(sb))) if len(sa) == len(sb) else float("nan")
    ok = same_labels and abs(auc_a - auc_b) < 1e-9
    print(f"\n{name}")
    print(f"  rows            original {len(la):>6}   compact {len(lb):>6}")
    print(f"  label multiset  {'identical' if same_labels else 'DIFFERENT'}")
    print(f"  ROC-AUC         original {auc_a:.10f}   compact {auc_b:.10f}")
    print(f"  |delta AUC|     {abs(auc_a - auc_b):.2e}")
    print(f"  max |score dev| {max_dev:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    root = tempfile.mkdtemp(prefix="km_equiv_")
    here = os.getcwd()
    sys.path.insert(0, here)
    import keymotif_data as kd

    try:
        grouping = make_synthetic(root)
        os.chdir(root)

        print("=" * 66)
        print(f"EQUIVALENCE TEST  (dim={DIM}, {N_ACC} accessions, {N_PHAGE} phages)")
        print("=" * 66)

        wide = original_wide_frame()
        print(f"original wide frame : {wide.shape[0]:,} rows x {wide.shape[1]:,} cols")

        pairs, host_emb, virus_emb = kd.build_cache(verbose=False)
        print(f"compact pair index  : {len(pairs):,} rows, "
              f"host {host_emb.shape}, virus {virus_emb.shape}")
        assert len(pairs) == len(wide), "row count mismatch!"

        ok0 = report("script 0 (PHL-AVG, in-fold averaging)",
                     original_script0(wide, grouping),
                     compact_script0(kd, pairs, host_emb, virus_emb, grouping))

        print(f"\nsanity: original drop_duplicates() removes "
              f"{len(wide) - len(wide.drop(columns=[c for c in wide.columns if c.startswith('host_')]).drop_duplicates()):,} "
              f"of {len(wide):,} rows")

        ok3 = report("script 3 (PHL-RBP+S, dedup + serotype one-hot)",
                     original_script3(wide, grouping),
                     compact_script3(kd, pairs, host_emb, virus_emb, grouping))

        print("\n" + "=" * 66)
        print("ALL EQUIVALENT" if (ok0 and ok3) else "MISMATCH — do not use")
        print("=" * 66)
        return 0 if (ok0 and ok3) else 1
    finally:
        os.chdir(here)
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
