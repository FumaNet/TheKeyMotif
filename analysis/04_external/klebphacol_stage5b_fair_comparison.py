#!/usr/bin/env python3
"""
klebphacol_stage5b_fair_comparison.py — three fairer re-cuts of the Stage 5
comparison. Stage 5's all-pairs number scores TropiSEQ's abstentions (score
0 on every unscored phage) as if they were confident wrong answers, which
overstates how much worse TropiSEQ looks. Not reported as a headline number
here; if shown at all it is explicitly labelled as penalising abstention.

1. COVERED-SUBSET: both models, restricted to the 23 phages TropiSEQ (on
   the 193-RBP query set) actually made a call for. The like-for-like
   number -- same pairs scored by both models on their own terms.

2. DEPOLYMERASE-DOMAIN: restricted to the 69 RBPs with scores_DepoScope
   >=0.5 -- the input TropiSEQ is actually designed for (DepoScope-defined
   depolymerase domains), as opposed to Stage 3's 193 RBPDetect-defined
   RBPs (PHL-RBP+S's own pipeline's input, and a materially different,
   larger, RBPdetect-called set -- only 53/69 overlap). If TropiSEQ's
   coverage is much higher here than the 24/187 (12.8%) seen on the
   RBPDetect set, the low coverage figure is a query-set artifact of this
   benchmark's choice to score PHL-RBP+S's own RBP definition, not
   evidence TropiSEQ itself rarely fires.

3. COVERAGE AS A METRIC: reported as a first-class result, not a caveat.
   PHL-RBP+S has no abstention mechanism -- it always emits a score,
   confident or not, by construction. That is a limitation of PHL-RBP+S
   (it cannot say "I don't know"), not a capability TropiSEQ lacks.
"""
import os
import json
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import keymotif_data as kd
import klebphacol_stage4_train_predict as s4
import klebphacol_stage5_tropiseq as s5

OUT_DIR = "Results/klebphacol"
INTERACTION_FILES = {
    "LB strict": f"{OUT_DIR}/interactions_LB_strict.csv",
    "LB permissive": f"{OUT_DIR}/interactions_LB_permissive.csv",
    "TSB strict": f"{OUT_DIR}/interactions_TSB_strict.csv",
}


def compute_metrics(labels, scores):
    if len(set(labels)) < 2 or len(labels) == 0:
        return dict(roc_auc=float("nan"), pr_auc=float("nan"),
                    n=len(labels), n_pos=int(sum(labels)) if len(labels) else 0,
                    base_rate=float("nan"))
    return dict(roc_auc=roc_auc_score(labels, scores),
                pr_auc=average_precision_score(labels, scores),
                n=len(labels), n_pos=int(sum(labels)), base_rate=np.mean(labels))


def eval_pairs(pred_scores, inter_path, phages_subset, phage_stratum=None, stratified=False):
    inter = pd.read_csv(inter_path)
    inter = inter[inter.phage.isin(phages_subset)]
    merged = inter.merge(pred_scores, on=["phage", "strain"], how="left")
    merged["score"] = merged["score"].fillna(0.0)
    if not stratified:
        m = compute_metrics(merged.label.values, merged.score.values)
        m["lift"] = m["base_rate"] / merged.label.mean() if merged.label.mean() else float("nan")
        return {"overall": m}
    merged["stratum"] = merged.phage.map(phage_stratum)
    scope_base = merged.label.mean()
    out = {}
    m = compute_metrics(merged.label.values, merged.score.values)
    m["lift"] = 1.0
    out["overall"] = m
    for s in ("novel", "related", "near-identical"):
        sub = merged[merged.stratum == s]
        if len(sub):
            met = compute_metrics(sub.label.values, sub.score.values)
            met["lift"] = met["base_rate"] / scope_base if scope_base else float("nan")
            out[s] = met
    return out


def print_table(title, results_by_model):
    print(f"\n  [{title}]")
    print(f"    {'model':<12}{'stratum':<16}{'n':>6}{'n_pos':>7}{'base_rate':>11}"
          f"{'lift':>7}{'ROC-AUC':>10}{'PR-AUC':>10}")
    for model, res in results_by_model.items():
        for s, m in res.items():
            print(f"    {model:<12}{s:<16}{m['n']:>6}{m['n_pos']:>7}"
                  f"{100*m['base_rate']:>10.1f}%{m['lift']:>6.2f}x"
                  f"{m['roc_auc']:>10.3f}{m['pr_auc']:>10.3f}")


def get_phlrbps_pred(rbp_df):
    import pickle
    with open(s4.MODEL_PATH, "rb") as fh:
        saved = pickle.load(fh)
    model, sero_cols = saved["model"], saved["sero_cols"]
    sero_encoded, sero_cols = s4.build_training_serotype_vocab()
    hosts, host_S, n_absent = s4.build_host_sero_matrix(sero_cols)
    rbp_emb = embed_rbps(rbp_df)
    rbp_gene_idx = {g: i for i, g in enumerate(rbp_df.gene_ID)}
    pred = s4.predict_klebphacol(model, sero_cols, hosts, host_S, rbp_df, rbp_emb, rbp_gene_idx)
    return pred.rename(columns={"strain": "strain"})[["phage", "strain", "score"]]


def embed_rbps(rbp_df, cache_gene_ids=None, cache_emb=None):
    """Reuse Stage 3's cached embeddings for any gene_ID already embedded;
    embed only the rest fresh."""
    if cache_gene_ids is None:
        existing = pd.read_csv(f"{OUT_DIR}/stage3_rbps_tagged.csv")
        cache_gene_ids = list(existing.gene_ID)
        cache_emb = np.load(f"{OUT_DIR}/stage3_rbp_embeddings.npy")
    cache_idx = {g: i for i, g in enumerate(cache_gene_ids)}

    out = np.zeros((len(rbp_df), cache_emb.shape[1]), dtype=np.float32)
    missing = []
    for i, (gene_id, seq) in enumerate(zip(rbp_df.gene_ID, rbp_df.protein_sequence)):
        if gene_id in cache_idx:
            out[i] = cache_emb[cache_idx[gene_id]]
        else:
            missing.append((i, gene_id, seq))

    if missing:
        print(f"Embedding {len(missing)} new sequences with ESM-2 650M "
              f"(return_contacts=False)...")
        import torch, esm
        from tqdm import tqdm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        for i, gene_id, seq in tqdm(missing):
            data = [(gene_id, seq)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            rep = results["representations"][33]
            out[i] = rep[0, 1:len(seq) + 1].mean(0).cpu().numpy()
    return out


def tropiseq_predict_on(rbp_df, fasta_path, blast_out_path):
    with open(fasta_path, "w") as f:
        for _, r in rbp_df.iterrows():
            f.write(f">{r.gene_ID}\n{r.protein_sequence}\n")
    subprocess.run(["conda", "run", "-n", "genomics", "blastp",
                     "-query", fasta_path, "-db", s5.BLAST_DB,
                     "-out", blast_out_path, "-outfmt", "6", "-evalue", "1e-10"],
                    check=True)
    blast = pd.read_csv(blast_out_path, sep="\t", names=s5.BLAST_COLS)
    qlen = dict(zip(rbp_df.gene_ID, rbp_df.protein_sequence.str.len()))
    dico_cluster = json.load(open(f"{s5.TS_MODEL}/dico_cluster.cdhit__0.85.json"))
    dico_cluster_r = {ref: cluster for cluster, refs in dico_cluster.items() for ref in refs}
    dico_pred_raw = json.load(open(f"{s5.TS_MODEL}/prediction_based.labeling.0604.json"))
    dico_pred = {f"Dpo_cdhit_{c.split('_')[1]}": hits for c, hits in dico_pred_raw.items()}

    rows = []
    for gene_id, sub in blast.groupby("qseqid", sort=False):
        best = sub.iloc[0]
        coverage = qlen[gene_id] / best.length
        if best.bitscore <= s5.BITSCORE_THRESH or coverage <= s5.COVERAGE_THRESH:
            continue
        cluster = dico_cluster_r.get(best.sseqid)
        kl_scores = dico_pred.get(cluster, {}) if cluster else {}
        rows.append(dict(gene_ID=gene_id, sseqid=best.sseqid, pident=best.pident,
                          bitscore=best.bitscore, coverage=coverage, cluster=cluster,
                          kl_scores=kl_scores))
    pred = pd.DataFrame(rows)
    n_total = rbp_df.gene_ID.nunique()
    print(f"  TropiSEQ coverage: {len(pred)}/{n_total} RBPs pass "
          f"bitscore>{s5.BITSCORE_THRESH} + coverage>{s5.COVERAGE_THRESH} "
          f"({100*len(pred)/n_total:.1f}%)")
    kept = pred[pred.pident < s5.FAIRNESS_IDENTITY_THRESH]
    n_dropped = len(pred) - len(kept)
    if n_dropped:
        print(f"  Fairness rule: {n_dropped}/{len(pred)} dropped (>=95% identity)")
    return kept


def main():
    rbp_full = pd.read_csv(f"{OUT_DIR}/stage3_rbps_tagged.csv")
    phage_stratum = (rbp_full.assign(risk=rbp_full.stratum.map(
        {"novel": 0, "related": 1, "near-identical": 2}))
        .groupby("phage_ID")["risk"].max()
        .map({0: "novel", 1: "related", 2: "near-identical"}))

    print("=" * 66)
    print("PART 3: COVERAGE AS ITS OWN METRIC")
    print("=" * 66)
    ts_kept_193 = tropiseq_predict_on(rbp_full, s5.QUERY_FASTA, s5.BLAST_OUT)
    covered_phages = set(rbp_full.set_index("gene_ID").loc[ts_kept_193.gene_ID, "phage_ID"])
    print(f"\n  TropiSEQ: {len(ts_kept_193)}/{rbp_full.gene_ID.nunique()} RBPs scored, "
          f"{len(covered_phages)}/52 phages get >=1 scored RBP")
    print(f"  PHL-RBP+S: 193/193 RBPs scored, 52/52 phages get >=1 scored RBP "
          f"(100% by construction -- it has no abstention mechanism; that is a "
          f"limitation of PHL-RBP+S, not an advantage over TropiSEQ)")

    pred_phl = get_phlrbps_pred(rbp_full)
    rbp_to_phage = dict(zip(rbp_full.gene_ID, rbp_full.phage_ID))
    ts_kept_193 = ts_kept_193.copy()
    ts_kept_193["phage_ID"] = ts_kept_193.gene_ID.map(rbp_to_phage)
    pred_ts_193 = s5.build_pair_scores(ts_kept_193, rbp_full)

    print("\n" + "=" * 66)
    print("PART 1: COVERED-SUBSET COMPARISON (the like-for-like number)")
    print("=" * 66)
    print(f"Restricted to the {len(covered_phages)} phages TropiSEQ actually scored: "
          f"{sorted(covered_phages)}")
    for label, path in INTERACTION_FILES.items():
        print(f"\n--- {label} ---")
        res_phl = eval_pairs(pred_phl, path, covered_phages, phage_stratum, stratified=True)
        res_ts = eval_pairs(pred_ts_193, path, covered_phages, phage_stratum, stratified=True)
        print_table(label, {"PHL-RBP+S": res_phl, "TropiSEQ": res_ts})

    print("\n" + "=" * 66)
    print("PART 2: DEPOLYMERASE-DOMAIN COMPARISON (DepoScope>=0.5, TropiSEQ's own input)")
    print("=" * 66)
    depo = pd.read_csv(f"{OUT_DIR}/stage5b_depo_rbps.csv")
    print(f"{len(depo)} DepoScope>=0.5 RBPs across {depo.phage_ID.nunique()} phages")
    ts_kept_depo = tropiseq_predict_on(depo, f"{s5.TS_DIR}/klebphacol_depo_query.fasta",
                                        f"{s5.TS_DIR}/blast_out_depo.tsv")
    covered_phages_depo = set(depo.set_index("gene_ID").loc[ts_kept_depo.gene_ID, "phage_ID"]) \
        if len(ts_kept_depo) else set()
    print(f"  -> {len(covered_phages_depo)}/{depo.phage_ID.nunique()} DepoScope-input phages get a TropiSEQ call")
    if len(ts_kept_depo):
        cov_rate_all = 24 / rbp_full.gene_ID.nunique()
        cov_rate_depo = len(ts_kept_depo) / depo.gene_ID.nunique()
        print(f"  Coverage on RBPdetect (193-RBP) set: {100*cov_rate_all:.1f}%  vs  "
              f"DepoScope (69-RBP) set: {100*cov_rate_depo:.1f}%")

    pred_phl_depo = get_phlrbps_pred(depo)
    ts_kept_depo = ts_kept_depo.copy()
    if len(ts_kept_depo):
        rbp_to_phage_depo = dict(zip(depo.gene_ID, depo.phage_ID))
        ts_kept_depo["phage_ID"] = ts_kept_depo.gene_ID.map(rbp_to_phage_depo)
        pred_ts_depo = s5.build_pair_scores(ts_kept_depo, depo)
    else:
        pred_ts_depo = pd.DataFrame(columns=["phage", "strain", "score"])

    depo_phages = set(depo.phage_ID.unique())
    for label, path in INTERACTION_FILES.items():
        print(f"\n--- {label} ---")
        res_phl = eval_pairs(pred_phl_depo, path, depo_phages, stratified=False)
        res_ts = eval_pairs(pred_ts_depo, path, depo_phages, stratified=False)
        print_table(label, {"PHL-RBP+S": res_phl, "TropiSEQ": res_ts})


if __name__ == "__main__":
    main()
