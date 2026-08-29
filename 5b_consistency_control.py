"""
5b_consistency_control.py — is PHL-M's advantage the MOTIF, or just a
CONSISTENT training set?

THE PROBLEM WITH PHL-RANDOM
---------------------------
PHL-M is trained with only motif-bearing RBPs as positives, and compared against
a control that picks a RANDOM RBP per interacting phage instead. PHL-M wins
(ROC 0.658 vs 0.545).

But your own thesis (S5.3) names the confound: picking a random protein destroys
the consistency of the training set, so there is no pattern left to learn. The
comparison then mostly shows that coherent training sets beat incoherent ones --
which is close to tautological -- rather than that these motifs are functional.

THE MISSING CONTROL
-------------------
A selection rule that produces an equally COHERENT training set while having
nothing to do with MEME. If such a rule matches PHL-M's 0.658, the motif adds
nothing beyond consistency. If PHL-M still wins clearly, the motif carries real
information.

Three rules are implemented, all deterministic and all motif-blind:

  longest      the longest RBP of each interacting phage. Adsorption proteins
               are often the largest tail proteins, so this is a strong,
               biologically motivated non-motif rule.
  central      the RBP with highest mean sequence similarity to all other RBPs
               of phages infecting the same serotype. Explicitly mimics what
               MEME rewards (conservation within a serotype group) WITHOUT
               running MEME -- the sharpest of the three.
  first        the first RBP by protein_ID. Coherent but arbitrary; a floor for
               "consistency alone".

Everything else -- LOGO protocol, negative undersampling, hyperparameters,
max-aggregation, evaluation against original interaction labels -- is identical
to 5_motif_focus.py, so the numbers are directly comparable.

INTERPRETING THE RESULT
-----------------------
    PHL-M >> all three     motif carries information beyond consistency
    PHL-M ~= longest       the "motif" may be tracking protein length
    PHL-M ~= central       the motif is a conservation artifact, not a
                           functional element -- the most likely outcome given
                           that FIMO found no serotype-exclusive motif
    all ~= 0.545           nothing works; consistency is not the story either

Note this control is INDEPENDENT of the leakage question. Run it alongside
fold_internal_motifs.py: together they separate three explanations -- leakage,
mere consistency, and genuine motif signal.

    python 5b_consistency_control.py --rule longest
    python 5b_consistency_control.py --rule central
    python 5b_consistency_control.py --rule first
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
THRESHOLDS = [1.0, 0.995, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75]
TSTR = ["100", "99.5", "99", "95", "90", "85", "80", "75"]
GROUPING_FILES = [
    "grouping/grouping_1.pkl", "grouping/grouping_995.pkl",
    "grouping/grouping_990.pkl", "grouping/grouping_950.pkl",
    "grouping/grouping_900.pkl", "grouping/grouping_850.pkl",
    "grouping/grouping_800.pkl", "grouping/grouping_750.pkl",
]


def load_rbp_sequences(path="Data/RBPbase.csv"):
    if not os.path.exists(path):
        raise SystemExit(f"Need {path} (RBP sequences) for --rule longest/central.")
    rbp = pd.read_csv(path)
    low = {c.lower(): c for c in rbp.columns}
    ph = next(low[c] for c in ["phage_id", "phage"] if c in low)
    pid = next(low[c] for c in ["protein_id", "protein"] if c in low)
    sq = next(low[c] for c in
              ["protein_sequence", "protein_seq", "sequence"] if c in low)
    rbp = rbp.rename(columns={ph: "phage_ID", pid: "protein_ID",
                              sq: "protein_sequence"})
    return rbp.dropna(subset=["protein_sequence"])


def select_longest(pairs, rbp):
    """Longest RBP per phage. Deterministic, motif-blind, biologically motivated."""
    rbp = rbp.copy()
    rbp["L"] = rbp["protein_sequence"].str.len()
    pick = rbp.sort_values(["phage_ID", "L", "protein_ID"],
                           ascending=[True, False, True]) \
              .drop_duplicates("phage_ID")[["phage_ID", "protein_ID"]]
    return dict(zip(pick["phage_ID"], pick["protein_ID"]))


def select_first(pairs, rbp):
    """First RBP by protein_ID. Coherent but arbitrary — the consistency floor."""
    pick = rbp.sort_values(["phage_ID", "protein_ID"]).drop_duplicates("phage_ID")
    return dict(zip(pick["phage_ID"], pick["protein_ID"]))


def select_central(pairs, rbp, sero_map):
    """
    Per serotype, the RBP most similar on average to the other RBPs of phages
    infecting that serotype. This is what MEME rewards -- within-group
    conservation -- reached without running MEME. The sharpest control.
    """
    try:
        from rapidfuzz.distance import Levenshtein
        sim = Levenshtein.normalized_similarity
    except ImportError:
        import difflib

        def sim(a, b):
            return difflib.SequenceMatcher(None, a, b).ratio()

    pos = pairs[pairs["label"] == 1].copy()
    pos["serotype"] = pos["accession"].map(sero_map)
    out = {}
    for sero, g in pos.groupby("serotype"):
        phages = sorted(g["phage_ID"].unique())
        sub = rbp[rbp["phage_ID"].isin(phages)]
        if len(sub) < 2:
            continue
        seqs = sub["protein_sequence"].tolist()
        ids = list(zip(sub["phage_ID"], sub["protein_ID"]))
        best = {}
        for i, (ph, pid) in enumerate(ids):
            others = [s for j, s in enumerate(seqs) if ids[j][0] != ph]
            if not others:
                continue
            score = float(np.mean([sim(seqs[i], o) for o in others]))
            if ph not in best or score > best[ph][1]:
                best[ph] = (pid, score)
        for ph, (pid, _) in best.items():
            out[(sero, ph)] = pid
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["longest", "central", "first"],
                    required=True)
    ap.add_argument("--thresholds", default=None,
                    help="Comma-separated subset, e.g. 100,90")
    args = ap.parse_args()

    OUT = f"Results/5b_AUCs_consistency_{args.rule}.pkl"
    CKPT = f"Results/5b_checkpoint_{args.rule}.pkl"

    pairs, host_emb, virus_emb = kd.load()
    rbp = load_rbp_sequences()

    df_sero = pd.read_csv("Data/kaptive_results.tsv", sep="\t")
    df_sero = df_sero[["Assembly", "Best match type", "Match confidence"]]
    sero_map = dict(zip(df_sero["Assembly"].astype(str),
                        df_sero["Best match type"]))
    one_hot = pd.get_dummies(df_sero["Best match type"], prefix="sero_")
    sero_encoded = pd.concat([df_sero[["Assembly"]], one_hot], axis=1)
    sero_cols = [c for c in sero_encoded.columns if c != "Assembly"]

    print(f"Selection rule: {args.rule}")
    if args.rule == "central":
        chosen = select_central(pairs, rbp, sero_map)
        print(f"  chose {len(chosen)} (serotype, phage) representatives")
    else:
        fn = select_longest if args.rule == "longest" else select_first
        chosen = fn(pairs, rbp)
        print(f"  chose {len(chosen)} phage representatives")

    def is_chosen(row):
        if args.rule == "central":
            return chosen.get((sero_map.get(row["accession"]),
                               row["phage_ID"])) == row["protein_ID"]
        return chosen.get(row["phage_ID"]) == row["protein_ID"]

    os.makedirs("Results", exist_ok=True)
    done = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}
    wanted = ({t.strip() for t in args.thresholds.split(",")}
              if args.thresholds else set(TSTR))

    for i, _t in enumerate(THRESHOLDS):
        if TSTR[i] in done or TSTR[i] not in wanted:
            continue
        fold_pairs = kd.attach_groups(pairs, GROUPING_FILES[i])

        # Relabelling, matching 5_motif_focus.py: positives are ONLY the
        # selected RBP of interacting phages; negatives are undersampled to one
        # RBP per non-interacting pair.
        sel = fold_pairs.apply(is_chosen, axis=1)
        pos = fold_pairs[(fold_pairs["label"] == 1) & sel]
        neg = (fold_pairs[fold_pairs["label"] == 0]
               .drop_duplicates(subset=["accession", "phage_ID"]))
        train_pool = pd.concat([pos, neg], ignore_index=True)

        logo = LeaveOneGroupOut()
        scores_max, label_max = [], []
        n_groups = fold_pairs["group_loci"].nunique()
        pbar = tqdm(total=n_groups, desc=f"{args.rule} @ {TSTR[i]}%")

        for g in sorted(fold_pairs["group_loci"].unique()):
            tr = train_pool[train_pool["group_loci"] != g]
            te = fold_pairs[fold_pairs["group_loci"] == g]
            if tr["label"].nunique() < 2 or len(te) == 0:
                pbar.update(1)
                continue

            sub_tr, S_tr = kd.attach_serotype(tr, sero_encoded, sero_cols)
            sub_te, S_te = kd.attach_serotype(te, sero_encoded, sero_cols)
            X_tr = kd.make_X(sub_tr, host_emb, virus_emb, mode="virus", sero=S_tr)
            X_te = kd.make_X(sub_te, host_emb, virus_emb, mode="virus", sero=S_te)
            y_tr = sub_tr["label"].astype(int).values
            y_te = sub_te["label"].astype(int).values

            n_pos = int((y_tr == 1).sum())
            n_neg = int((y_tr == 0).sum())
            imb = n_pos / n_neg if n_neg else 1

            xgb = XGBClassifier(scale_pos_weight=1 / imb, learning_rate=0.3,
                                n_estimators=250, max_depth=7,
                                eval_metric="logloss", tree_method="hist",
                                device=DEVICE, random_state=0)
            xgb.fit(X_tr, y_tr)
            sc = xgb.predict_proba(X_te)[:, 1]

            dfp = pd.DataFrame({"accession": sub_te["accession"].values,
                                "phage_ID": sub_te["phage_ID"].values,
                                "true_label": y_te, "score": sc})
            mx = dfp.groupby(["accession", "phage_ID"]).agg(
                {"score": "max", "true_label": "first"}).reset_index()
            scores_max.append(mx["score"].values)
            label_max.append(mx["true_label"].values)
            pbar.update(1)
        pbar.close()

        if not scores_max:
            continue
        sm = np.concatenate(scores_max)
        lm = np.concatenate(label_max)
        if len(set(lm)) < 2:
            continue
        fpr, tpr, _ = roc_curve(lm, sm)
        prec, rec, _ = precision_recall_curve(lm, sm)
        a = round(auc(fpr, tpr), 3)
        print(f"  {args.rule} @ {TSTR[i]}%: ROC {a}   PR {auc(rec, prec):.3f}")
        done[TSTR[i]] = (lm, sm, a)
        with open(CKPT, "wb") as fh:
            pickle.dump(done, fh)

    missing = [t for t in TSTR if t not in done]
    if missing:
        print(f"\nCheckpoint holds {len(done)}/8. Missing: {missing}")
        return
    with open(OUT, "wb") as fh:
        pickle.dump([done[t] for t in TSTR], fh)
    print(f"\nWrote {OUT} — {[done[t][2] for t in TSTR]}")
    print("\nCompare:")
    print("  PHL-M (motif)   [0.658, 0.652, 0.656, 0.651, 0.647, 0.636, 0.638, 0.635]")
    print("  PHL-Random      0.545 (best across thresholds)")
    print("If this rule matches PHL-M, the advantage was consistency, not motif.")


if __name__ == "__main__":
    main()
