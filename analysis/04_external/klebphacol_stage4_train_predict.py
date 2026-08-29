#!/usr/bin/env python3
"""
klebphacol_stage4_train_predict.py — Stage 4: train PHL-RBP+S on Boeckaerts
(minus the 7 RBP-overlap-excluded phages), no CV, and predict on KlebPhaCol,
held out entirely.

MODEL: identical architecture/hyperparameters to 3_max_max_sero_compact.py
(RBP embedding [1280] + one-hot capsular serotype, XGBClassifier(
learning_rate=0.3, n_estimators=250, max_depth=7, scale_pos_weight=1/imbalance,
eval_metric="logloss", tree_method="hist")), but trained ONCE on the full
(minus exclusions) Boeckaerts pair table instead of LOGO-CV -- KlebPhaCol IS
the held-out set here, so no internal fold-holdout is needed. Deliberately
keeps the drop_duplicates() no-op behaviour documented in SETUP_LOCAL.md
section 4 (host protein_index survives -> hosts with larger K-loci are
upweighted) -- this is what "PHL-RBP+S" means as an established artifact in
this project; changing it would silently make Stage 4 a different model.

HOST SEROTYPE CROSSWALK: the training one-hot vocabulary is built from
Kaptive's "Best match type" column (a mix of resolved K-antigen names like
"K13", "unknown (KLxx)" fallbacks, and "Capsule null"), but Table S1 only
gives KlebPhaCol hosts the KL LOCUS number ("KL68"). These are different
namespaces -- string-matching "KL68" against training column names would
spuriously fail almost everywhere. Built via kaptive_results.tsv's own
locus->type mapping (its "Best match locus" column, in the same "KLxx" form
Table S1 uses) instead: 82/87 training loci map to exactly one type value;
5 (KL15, KL22, KL39, KL57, KL62) are ambiguous within training itself
(mostly a majority "Kxx" call plus a minority "Capsule null" or, for KL22,
a minority "K37" -- different assemblies of the same locus, different
Kaptive confidence) and resolved to their MAJORITY type, documented
per-locus. A KlebPhaCol host whose locus never appears in training at all
gets an all-zero serotype vector -- a silent degradation the guard below
counts explicitly rather than lets pass unnoticed.

STRATIFICATION: every (host, phage) pair inherits its phage's Stage 3
stratum (novel/related/near-identical -- the riskiest stratum among that
phage's own RBPs, since max-aggregation lets one near-identical RBP carry
the whole prediction regardless of the phage's other RBPs).
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import keymotif_data as kd

OUT_DIR = "results/klebphacol"
MODEL_PATH = os.path.join(OUT_DIR, "stage4_model.pkl")
RBP_TAGGED = os.path.join(OUT_DIR, "stage3_rbps_tagged.csv")
RBP_EMB = os.path.join(OUT_DIR, "stage3_rbp_embeddings.npy")
EXCLUSIONS = os.path.join(OUT_DIR, "overlap_exclusions.csv")
N_BOOT = 2000
RNG_SEED = 42


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def build_training_serotype_vocab():
    df_sero = pd.read_csv("data/kaptive_results.tsv", sep="\t")
    df_sero = df_sero[["Assembly", "Best match type", "Match confidence"]]
    one_hot = pd.get_dummies(df_sero["Best match type"], prefix="sero_")
    sero_encoded = pd.concat([df_sero[["Assembly"]], one_hot], axis=1)
    sero_cols = [c for c in sero_encoded.columns if c != "Assembly"]
    return sero_encoded, sero_cols


def train_model():
    pairs, host_emb, virus_emb = kd.load()

    excl = pd.read_csv(EXCLUSIONS)
    excluded_phages = sorted(excl.boeckaerts_phage_to_exclude.unique())
    print(f"Excluding {len(excluded_phages)} RBP-overlap phages from training: {excluded_phages}")

    n_before = pairs.phage_ID.nunique()
    train_pairs = pairs[~pairs.phage_ID.isin(excluded_phages)].reset_index(drop=True)
    n_after = train_pairs.phage_ID.nunique()

    # --- GUARD 1: assert the exclusion actually took, don't assume ---
    assert not (set(train_pairs.phage_ID.unique()) & set(excluded_phages)), \
        "excluded phages still present in training pairs after filtering"
    assert n_after == 105 - len(excluded_phages), \
        f"expected {105 - len(excluded_phages)} remaining Boeckaerts phages, got {n_after}"
    print(f"GUARD 1 PASSED: {n_before} -> {n_after} distinct Boeckaerts phages "
          f"in training ({len(excluded_phages)} excluded, none present).")

    sero_encoded, sero_cols = build_training_serotype_vocab()
    print(f"Serotype vocabulary: {len(sero_cols)} columns "
          f"(from Kaptive 'Best match type' on {sero_encoded.Assembly.nunique()} training hosts)")

    sub, S = kd.attach_serotype(train_pairs, sero_encoded, sero_cols)
    X = kd.make_X(sub, host_emb, virus_emb, mode="virus", sero=S)
    y = sub["label"].astype(int).values

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    imbalance = n_pos / n_neg
    print(f"Training rows: {len(y):,}  positives: {n_pos:,}  negatives: {n_neg:,}  "
          f"(base rate {100*n_pos/len(y):.2f}%)")

    model = XGBClassifier(
        scale_pos_weight=1 / imbalance,
        learning_rate=0.3,
        n_estimators=250,
        max_depth=7,
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        random_state=0,
    )
    print("Training (no CV, single model, ~480k rows x "
          f"{X.shape[1]} features)...")
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": model, "sero_cols": sero_cols}, fh)
    print(f"Wrote {MODEL_PATH}")
    return model, sero_cols


# ---------------------------------------------------------------------------
# KlebPhaCol host serotype crosswalk
# ---------------------------------------------------------------------------

def build_locus_crosswalk():
    """kaptive_results.tsv's own (Best match locus -> Best match type)
    mapping, resolving the 5 ambiguous loci to their majority type."""
    df = pd.read_csv("data/kaptive_results.tsv", sep="\t")
    counts = df.groupby(["Best match locus", "Best match type"]).size()
    crosswalk = {}
    ambiguous_report = []
    for locus, sub in counts.groupby(level=0):
        sub = sub.sort_values(ascending=False)
        crosswalk[locus] = sub.index[0][1]
        if len(sub) > 1:
            ambiguous_report.append((locus, dict(sub.droplevel(0))))
    return crosswalk, ambiguous_report


def load_klebphacol_hosts():
    """Strain names here MUST match Stage 1's canonical S4-style names (used
    throughout interactions_*.csv and the genome files), not Table S1's raw
    "Isolate name" -- 3 of those carry a "(aka ...)" alias suffix and a
    literal space (e.g. "164413U/2 (aka 164413U12)") that would silently
    fail to join against interactions_LB_strict.csv's "164413U12", turning
    those 3 hosts' pairs into unscored, dropped rows rather than an error.
    Uses data/klebphacol/strain_aliases.csv's s4_name (the same canonical
    name Stage 1 committed and asserted bijective against Table S1)."""
    import pyxlsb
    with pyxlsb.open_workbook("data/klebphacol/Supplementary_Tables_R2.xlsb") as wb:
        with wb.get_sheet("Table S1") as sheet:
            rows = list(sheet.rows())
    header = [c.v for c in rows[2]]
    idx_kl = header.index("KL best match (Kaptive)")
    idx_name = header.index("Isolate name")
    out = []
    for r in rows[3:]:
        v = [c.v for c in r]
        if v[idx_name] is None:
            break
        name = v[idx_name]
        if isinstance(name, float) and name.is_integer():
            name = str(int(name))
        out.append((name, v[idx_kl]))
    raw = pd.DataFrame(out, columns=["s1_name", "kl_locus"])

    aliases = pd.read_csv("data/klebphacol/strain_aliases.csv")
    merged = raw.merge(aliases[["s4_name", "s1_name"]], on="s1_name", how="left")
    assert merged.s4_name.notna().all(), \
        f"{merged.s1_name[merged.s4_name.isna()].tolist()} not found in strain_aliases.csv"
    assert merged.s4_name.nunique() == len(merged) == 74, \
        f"expected 74 unique canonical strain names, got {merged.s4_name.nunique()}/{len(merged)}"
    return merged.rename(columns={"s4_name": "strain"})[["strain", "kl_locus"]]


def build_host_sero_matrix(sero_cols):
    crosswalk, ambiguous = build_locus_crosswalk()
    hosts = load_klebphacol_hosts()

    print("\n" + "=" * 66)
    print("GUARD 2: KlebPhaCol host KL types absent from training vocabulary")
    print("=" * 66)
    print("Ambiguous loci in training itself, resolved to majority type:")
    for locus, dist in ambiguous:
        print(f"  {locus}: {dist} -> using {max(dist, key=dist.get)!r}")

    col_lookup = {c: c for c in sero_cols}  # sero_ prefix applied below

    def resolve(locus):
        type_str = crosswalk.get(locus)
        if type_str is None:
            return None
        col = f"sero__{type_str}"
        return col if col in col_lookup else None

    hosts["training_type"] = hosts.kl_locus.map(crosswalk)
    hosts["sero_col"] = hosts.kl_locus.apply(resolve)
    n_absent = hosts.sero_col.isna().sum()
    print(f"\n{n_absent}/{len(hosts)} KlebPhaCol hosts have a KL locus absent from "
          f"the training vocabulary (all-zero serotype vector):")
    if n_absent:
        print(hosts[hosts.sero_col.isna()][["strain", "kl_locus"]].to_string(index=False))

    S = np.zeros((len(hosts), len(sero_cols)), dtype=np.float32)
    col_idx = {c: i for i, c in enumerate(sero_cols)}
    for i, col in enumerate(hosts.sero_col):
        if col is not None and col in col_idx:
            S[i, col_idx[col]] = 1.0
    hosts["sero_vec_idx"] = range(len(hosts))
    return hosts, S, n_absent


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------

def predict_klebphacol(model, sero_cols, hosts, host_S, rbp_df, rbp_emb,
                        rbp_gene_idx, top_n_per_phage=None):
    """For every (phage, RBP) x host combination, score with the model, then
    max-aggregate per (phage, host). top_n_per_phage, if given, restricts
    each phage to its top-N RBPs by scores_RBPDetect before aggregating
    (the sensitivity check)."""
    use_rbps = rbp_df
    if top_n_per_phage is not None:
        use_rbps = (rbp_df.sort_values("scores_RBPDetect", ascending=False)
                    .groupby("phage_ID").head(top_n_per_phage))

    rbp_vecs = rbp_emb[use_rbps.gene_ID.map(rbp_gene_idx).values]
    n_rbp = len(use_rbps)
    n_host = len(hosts)

    # build the full (n_rbp * n_host) x (1280 + n_sero) matrix directly;
    # 193 x 74 = 14,282 rows max -- trivial size, no need for kd.make_X's
    # index-based path (that's for the multi-hundred-thousand-row Boeckaerts
    # training set)
    rbp_block = np.repeat(rbp_vecs, n_host, axis=0)
    sero_block = np.tile(host_S, (n_rbp, 1))
    X = np.hstack([rbp_block, sero_block]).astype(np.float32)
    scores = model.predict_proba(X)[:, 1]

    phage_rep = np.repeat(use_rbps.phage_ID.values, n_host)
    host_rep = np.tile(hosts.strain.values, n_rbp)
    pred = pd.DataFrame({"phage": phage_rep, "strain": host_rep, "score": scores})
    agg = pred.groupby(["phage", "strain"])["score"].max().reset_index()
    return agg


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, scores):
    if len(set(labels)) < 2:
        return dict(roc_auc=float("nan"), pr_auc=float("nan"),
                    n=len(labels), n_pos=int(sum(labels)))
    return dict(roc_auc=roc_auc_score(labels, scores),
                pr_auc=average_precision_score(labels, scores),
                n=len(labels), n_pos=int(sum(labels)))


def bootstrap_ci(labels, scores, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n = len(labels)
    roc_vals, pr_vals = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb, sb = labels[idx], scores[idx]
        if len(set(yb)) < 2:
            continue
        roc_vals.append(roc_auc_score(yb, sb))
        pr_vals.append(average_precision_score(yb, sb))
    def ci(vals):
        return (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    return dict(roc_auc_ci=ci(roc_vals), pr_auc_ci=ci(pr_vals),
                n_valid_boots=len(roc_vals))


def _score_scope(m, label):
    """m: a merged (label, score, stratum) frame already restricted to one
    host scope. Returns the overall + per-stratum metrics dict for it."""
    results = {}
    results["overall"] = compute_metrics(m.label.values, m.score.values)
    results["overall"]["base_rate"] = m.label.mean()
    for s in ("novel", "related", "near-identical"):
        sub = m[m.stratum == s]
        if len(sub):
            met = compute_metrics(sub.label.values, sub.score.values)
            met["base_rate"] = sub.label.mean()
            results[s] = met
    return results


def evaluate(pred, inter_path, stratum_map, unseen_kl_hosts=()):
    """Returns ({"incl_unseen_kl": {...}, "excl_unseen_kl": {...}}, merged).

    incl_unseen_kl: every pair, including the hosts whose KL type has no
      training one-hot column (all-zero serotype vector) -- the fair
      deployment-realistic number, since a real query set will contain
      capsule types the training vocabulary never saw.
    excl_unseen_kl: same pairs minus those hosts -- the fair measure of the
      model as designed/trained, isolating genuine RBP-side generalisation
      from a host-side vocabulary gap that has nothing to do with RBP
      novelty. Both are legitimate; conflating them understates the model
      (the incl number is dragged down by a problem the excl number shows
      isn't there for known capsule types) and overstates deployment
      readiness (the excl number hides how often real queries will hit an
      unseen KL type).
    """
    inter = pd.read_csv(inter_path)
    merged = inter.merge(pred, left_on=["phage", "strain"], right_on=["phage", "strain"],
                          how="left")
    n_missing = merged.score.isna().sum()
    merged["stratum"] = merged.phage.map(stratum_map)
    m = merged.dropna(subset=["score"])

    out = {"incl_unseen_kl": _score_scope(m, "incl_unseen_kl")}
    out["incl_unseen_kl"]["overall"]["n_missing_predictions"] = int(n_missing)
    m_excl = m[~m.strain.isin(unseen_kl_hosts)]
    out["excl_unseen_kl"] = _score_scope(m_excl, "excl_unseen_kl")
    out["excl_unseen_kl"]["overall"]["n_missing_predictions"] = int(n_missing)
    return out, merged
