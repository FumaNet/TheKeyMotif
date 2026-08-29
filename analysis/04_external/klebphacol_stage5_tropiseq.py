#!/usr/bin/env python3
"""
klebphacol_stage5_tropiseq.py — Stage 5: TropiSEQ comparator.

METHOD (replicating scripts/Run_TropiSEQ.ipynb from the DpoTropiSearch repo
exactly): BLASTp each KlebPhaCol RBP (the SAME 193-RBP query set Stage 3
built and PHL-RBP+S was scored on -- same candidate proteins for both
models, see the input-difference flag below) against TropiSEQ's released
depolymerase-cluster database (Zenodo 10.5281/zenodo.14065540,
TropiSEQ_model.zip). Best hit (top bitscore row per query, BLAST's default
sort) is used if bitscore > 75 AND qlen/alignment_length > 0.8 -- both
exactly TropiSEQ's own published criteria, not a re-derivation. A query
passing that bar is mapped hit-sequence -> CD-HIT cluster ->
predicted-KL-type/score dict (prediction_based.labeling.0604.json, the
released version; notebook-era file names differ slightly but the schema
and lookup logic are identical).

FAIRNESS RULE: TropiSEQ's own training corpus (74,302 screened prophages)
can easily contain a sequence matching a KlebPhaCol depolymerase by chance
prophage redundancy, which would let TropiSEQ "predict" a case it had
effectively already memorised. TropiSEQ's paper discards such cases before
benchmarking; the exact numeric threshold they used isn't in the
notebooks available here, so this reproduces it with this project's own
established convention for "the same sequence, already memorised" (Stage
2b used 95% identity for exactly this purpose, on the PHL-RBP+S side) --
best-hit BLAST %identity >= 95% (already at >80% coverage, since a
prediction-hit is a prerequisite for the fairness check to even apply) ->
dropped from the benchmark, count reported.

STRATA: reuses Stage 3b's corrected (post-PHL-RBP+S-exclusion) phage-level
stratum labels, so both models are scored under the SAME novel/related/
near-identical grouping -- not independently derived from TropiSEQ's own
notion of similarity, which would defeat the like-for-like comparison.

HOST SCOPE: TropiSEQ predicts KL-locus labels directly (no serotype
one-hot vocabulary, so it has no analogue of PHL-RBP+S's Guard 2). To keep
the comparison on identical pairs, the primary comparison table restricts
BOTH models to the same excl-unseen-KL host subset PHL-RBP+S's Stage 4 used
(pairs are filtered by KlebPhaCol host, so "unseen-KL" here just means
"the same 16 hosts Stage 4 flagged," applied for comparability, not because
TropiSEQ has a vocabulary gap of its own).

INPUT-DIFFERENCE FLAG (not a footnote): PHL-RBP+S scores whole RBPdetect-
called RBPs (ESM-2 embeddings of the full protein); TropiSEQ scores
DepoScope/BLAST-defined depolymerase DOMAINS via direct sequence identity
to a reference database. Different feature representations of what is
sometimes the same protein and sometimes not (RBPdetect and DepoScope
don't always call the same region, or the same protein, as the
RBP/depolymerase). Any performance gap between the two models is
confounded with this input difference and cannot be attributed to
architecture alone.
"""
import os
import json
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

OUT_DIR = "results/klebphacol"
TS_DIR = "data/tropiseq"
TS_MODEL = f"{TS_DIR}/TropiSEQ_model"
BLAST_DB = f"{TS_MODEL}/depolymerase_clusters_database/TropiSeq_0.85.db"
QUERY_FASTA = f"{TS_DIR}/klebphacol_rbps_query.fasta"
BLAST_OUT = f"{TS_DIR}/blast_out.tsv"
BITSCORE_THRESH = 75
COVERAGE_THRESH = 0.8
FAIRNESS_IDENTITY_THRESH = 95.0

BLAST_COLS = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
              "qstart", "qend", "sstart", "send", "evalue", "bitscore"]


def run_blast(rbp_df):
    with open(QUERY_FASTA, "w") as f:
        for _, r in rbp_df.iterrows():
            f.write(f">{r.gene_ID}\n{r.protein_sequence}\n")
    subprocess.run(["conda", "run", "-n", "genomics", "blastp",
                     "-query", QUERY_FASTA, "-db", BLAST_DB,
                     "-out", BLAST_OUT, "-outfmt", "6", "-evalue", "1e-10"],
                    check=True)
    return pd.read_csv(BLAST_OUT, sep="\t", names=BLAST_COLS)


def predict_tropiseq(rbp_df):
    blast = run_blast(rbp_df)
    qlen = dict(zip(rbp_df.gene_ID, rbp_df.protein_sequence.str.len()))

    dico_cluster = json.load(open(f"{TS_MODEL}/dico_cluster.cdhit__0.85.json"))
    dico_cluster_r = {ref: cluster for cluster, refs in dico_cluster.items() for ref in refs}
    dico_pred_raw = json.load(open(f"{TS_MODEL}/prediction_based.labeling.0604.json"))
    dico_pred = {f"Dpo_cdhit_{c.split('_')[1]}": hits for c, hits in dico_pred_raw.items()}

    rows = []
    n_no_hit = 0
    for gene_id, sub in blast.groupby("qseqid", sort=False):
        best = sub.iloc[0]  # BLAST outfmt 6 is sorted by bitscore desc within each query
        coverage = qlen[gene_id] / best.length
        if best.bitscore <= BITSCORE_THRESH or coverage <= COVERAGE_THRESH:
            n_no_hit += 1
            continue
        cluster = dico_cluster_r.get(best.sseqid)
        kl_scores = dico_pred.get(cluster, {}) if cluster else {}
        rows.append(dict(gene_ID=gene_id, sseqid=best.sseqid, pident=best.pident,
                          bitscore=best.bitscore, coverage=coverage, cluster=cluster,
                          kl_scores=kl_scores))
    pred = pd.DataFrame(rows)
    n_all = rbp_df.gene_ID.nunique()
    print(f"BLAST predictions: {len(pred)}/{n_all} RBPs pass bitscore>{BITSCORE_THRESH} "
          f"and coverage>{COVERAGE_THRESH} ({n_no_hit} no-hit/below-threshold)")
    return pred


def apply_fairness_rule(pred):
    fair_excluded = pred[pred.pident >= FAIRNESS_IDENTITY_THRESH]
    kept = pred[pred.pident < FAIRNESS_IDENTITY_THRESH].copy()
    print(f"\nFAIRNESS RULE (best-hit identity >= {FAIRNESS_IDENTITY_THRESH}% -> "
          f"'already in TropiSEQ's own training data'):")
    print(f"  {len(fair_excluded)}/{len(pred)} predicted RBPs dropped")
    if len(fair_excluded):
        print(fair_excluded[["gene_ID", "sseqid", "pident", "cluster"]].to_string(index=False))
    return kept


def build_pair_scores(pred_kept, rbp_df):
    rbp_to_phage = dict(zip(rbp_df.gene_ID, rbp_df.phage_ID))
    pred_kept = pred_kept.copy()
    pred_kept["phage_ID"] = pred_kept.gene_ID.map(rbp_to_phage)

    import pyxlsb
    with pyxlsb.open_workbook("data/klebphacol/Supplementary_Tables_R2.xlsb") as wb:
        with wb.get_sheet("Table S1") as sheet:
            rows = list(sheet.rows())
    header = [c.v for c in rows[2]]
    idx_kl = header.index("KL best match (Kaptive)")
    idx_name = header.index("Isolate name")
    host_kl = {}
    for r in rows[3:]:
        v = [c.v for c in r]
        if v[idx_name] is None:
            break
        name = v[idx_name]
        if isinstance(name, float) and name.is_integer():
            name = str(int(name))
        host_kl[name] = v[idx_kl]
    aliases = pd.read_csv("data/klebphacol/strain_aliases.csv")
    s1_to_s4 = dict(zip(aliases.s1_name, aliases.s4_name))
    host_kl = {s1_to_s4.get(k, k): v for k, v in host_kl.items()}

    scored_phages = pred_kept.phage_ID.unique()
    scores = []
    for phage in scored_phages:
        phage_rbps = pred_kept[pred_kept.phage_ID == phage]
        for strain, kl in host_kl.items():
            best = 0.0
            for _, r in phage_rbps.iterrows():
                best = max(best, r.kl_scores.get(kl, 0.0))
            scores.append((phage, strain, best))
    return pd.DataFrame(scores, columns=["phage", "strain", "score"])


def compute_metrics(labels, scores):
    if len(set(labels)) < 2 or len(labels) == 0:
        return dict(roc_auc=float("nan"), pr_auc=float("nan"),
                    n=len(labels), n_pos=int(sum(labels)) if len(labels) else 0,
                    base_rate=float("nan"), lift=float("nan"))
    base_rate = np.mean(labels)
    return dict(roc_auc=roc_auc_score(labels, scores),
                pr_auc=average_precision_score(labels, scores),
                n=len(labels), n_pos=int(sum(labels)), base_rate=base_rate, lift=1.0)


def evaluate_scoped(pred_scores, inter_path, phage_stratum, unseen_kl_hosts):
    """Lift is always relative to THIS scope's own overall base rate (i.e.
    overall's lift is trivially 1.0x) -- comparing a stratum's positive
    concentration against a base rate computed on a different (e.g.
    unfiltered) pool would mix scopes and make the lift numbers not mean
    what they say."""
    inter = pd.read_csv(inter_path)
    merged = inter.merge(pred_scores, on=["phage", "strain"], how="left")
    merged["score"] = merged["score"].fillna(0.0)  # unscored (phage never predicted) -> lowest possible score
    merged["stratum"] = merged.phage.map(phage_stratum)

    out = {}
    for scope, mask in (("incl_unseen_kl", pd.Series(True, index=merged.index)),
                         ("excl_unseen_kl", ~merged.strain.isin(unseen_kl_hosts))):
        m = merged[mask]
        scope_base_rate = m.label.mean() if len(m) else float("nan")
        res = {}
        res["overall"] = compute_metrics(m.label.values, m.score.values)
        if res["overall"]["n"]:
            res["overall"]["lift"] = 1.0
        for s in ("novel", "related", "near-identical"):
            sub = m[m.stratum == s]
            if len(sub):
                met = compute_metrics(sub.label.values, sub.score.values)
                met["lift"] = met["base_rate"] / scope_base_rate if scope_base_rate else float("nan")
                res[s] = met
        out[scope] = res
    return out, merged


def main():
    rbp_df = pd.read_csv(f"{OUT_DIR}/stage3_rbps_tagged.csv")
    phage_stratum = (rbp_df.assign(risk=rbp_df.stratum.map(
        {"novel": 0, "related": 1, "near-identical": 2}))
        .groupby("phage_ID")["risk"].max()
        .map({0: "novel", 1: "related", 2: "near-identical"}))

    print("=" * 66)
    print("STAGE 5: TropiSEQ BLASTp prediction")
    print("=" * 66)
    pred = predict_tropiseq(rbp_df)
    pred_kept = apply_fairness_rule(pred)
    pred_scores = build_pair_scores(pred_kept, rbp_df)
    print(f"\nScored {pred_scores.phage.nunique()} phages x "
          f"{pred_scores.strain.nunique()} hosts = {len(pred_scores)} pairs "
          f"(phages with no surviving prediction score 0 everywhere)")

    # host scope: reuse Stage 4's crosswalk to get the same 16 unseen-KL hosts
    import klebphacol_stage4_train_predict as s4
    sero_encoded, sero_cols = s4.build_training_serotype_vocab()
    hosts, host_S, n_absent = s4.build_host_sero_matrix(sero_cols)
    unseen_kl_hosts = set(hosts[hosts.sero_col.isna()].strain)

    interaction_files = {
        "LB strict": f"{OUT_DIR}/interactions_LB_strict.csv",
        "LB permissive": f"{OUT_DIR}/interactions_LB_permissive.csv",
        "TSB strict": f"{OUT_DIR}/interactions_TSB_strict.csv",
    }

    print("\n" + "=" * 66)
    print("TropiSEQ RESULTS (same strata, same host-scope split as PHL-RBP+S)")
    print("=" * 66)
    for label, path in interaction_files.items():
        res, merged = evaluate_scoped(pred_scores, path, phage_stratum, unseen_kl_hosts)
        print(f"\n  [{label}]")
        for scope, title in (("incl_unseen_kl", "including 16 unseen-KL hosts"),
                              ("excl_unseen_kl", "excluding 16 unseen-KL hosts")):
            print(f"    -- {title} --")
            print(f"    {'stratum':<16}{'n':>6}{'n_pos':>7}{'base_rate':>11}"
                  f"{'lift':>8}{'ROC-AUC':>10}{'PR-AUC':>10}")
            for s in ("overall", "novel", "related", "near-identical"):
                if s not in res[scope]:
                    continue
                m = res[scope][s]
                print(f"    {s:<16}{m['n']:>6}{m['n_pos']:>7}{100*m['base_rate']:>10.1f}%"
                      f"{m['lift']:>7.2f}x{m['roc_auc']:>10.3f}{m['pr_auc']:>10.3f}")


if __name__ == "__main__":
    main()
