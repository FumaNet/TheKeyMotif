#!/usr/bin/env python3
"""
klebphacol_stage3_features.py — Stage 3: build the 193-RBP KlebPhaCol
feature set (RBPDetect>=0.5, 200<len<1500), tag each RBP with its Stage 2b
overlap stratum, embed with ESM-2 650M, and report stratum sizes at both
the RBP level and the (host, phage) pair level.

RBP-calling: RBPDetect>=0.5 only (not the DepoScope-OR variant). DepoScope
is Concha-Eloko's depolymerase-specific tool (TropiGAT/DpoTropiSearch), not
part of this pipeline; RBPdetect is both this project's own RBP caller and
the tool that built Boeckaerts' 274-RBP training set (RBPbase.csv's
xgb_score column). RBPbase.csv's minimum xgb_score is 0.520 -- Boeckaerts
used ~0.5, not a stricter cutoff, so 0.5 stands unmodified here.

Overlap stratum (Stage 2b, rapidfuzz Levenshtein normalized_similarity vs
all 274 Boeckaerts RBPs), assigned per RBP by its own best-hit identity:
  novel          < 80%
  related        80-95%
  near-identical  >= 95%
A PHAGE's stratum is the highest-risk stratum among its own RBPs (one
near-identical RBP is enough to carry a max-aggregated prediction,
regardless of how novel its other RBPs are) -- so a (host, phage) pair's
stratum depends only on which phage it is, never on the host. Pair-level
counts are therefore the interaction table's per-phage row counts, grouped
by that phage's stratum.
"""
import os
import pyxlsb
import pandas as pd
import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process

XLSB_PATH = "data/klebphacol/Supplementary_Tables_R2.xlsb"
RBPBASE_PATH = "data/RBPbase.csv"
OUT_DIR = "results/klebphacol"
MIN_LEN, MAX_LEN = 200, 1500
SCORE_THRESH = 0.5


def load_table_s3():
    with pyxlsb.open_workbook(XLSB_PATH) as wb:
        with wb.get_sheet("Table S3") as sheet:
            rows = list(sheet.rows())
    header = [c.v for c in rows[1]]
    n_named = sum(1 for h in header if h is not None)  # trailing None columns are sheet padding
    header = header[:n_named]
    data = []
    for r in rows[2:]:
        vals = [c.v for c in r]
        if vals[0] is None:
            break
        data.append(vals[:n_named])
    return pd.DataFrame(data, columns=header)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_table_s3()
    n_total_proteins = len(df)
    n_phages = df.phage_ID.nunique()

    rbp_mask = df.scores_RBPDetect >= SCORE_THRESH
    n_before_len = rbp_mask.sum()
    df["seqlen"] = df.protein_sequence.str.len()
    len_mask = (df.seqlen > MIN_LEN) & (df.seqlen < MAX_LEN)
    rbps = df[rbp_mask & len_mask].copy().reset_index(drop=True)

    print("=" * 66)
    print("STAGE 3: RBP set construction")
    print("=" * 66)
    print(f"Table S3 total scored proteins: {n_total_proteins} across {n_phages} phages")
    print(f"scores_RBPDetect >= {SCORE_THRESH}: {n_before_len}")
    print(f"+ {MIN_LEN}<len<{MAX_LEN}: {len(rbps)} ({n_before_len - len(rbps)} dropped by length)")
    per_phage_counts = rbps.groupby("phage_ID").size()
    n_phages_repr = rbps.phage_ID.nunique()
    print(f"Phages represented: {n_phages_repr}/52")
    if n_phages_repr < 52:
        missing = sorted(set(df.phage_ID.unique()) - set(rbps.phage_ID.unique()))
        print(f"  ZERO-RBP phages after filtering: {missing}")
    print(f"RBPs/phage: {len(rbps)}/52 = {len(rbps)/52:.3f} "
          f"(Boeckaerts baseline 274/105 = {274/105:.3f}, ratio {len(rbps)/52/(274/105):.2f}x)")

    # --- Stage 2b overlap stratum, recomputed for exactly this 193-RBP set ---
    bk = pd.read_csv(RBPBASE_PATH)
    print(f"\nComputing {len(rbps)} x {len(bk)} rapidfuzz identities for stratum tagging...")
    sim = process.cdist(rbps.protein_sequence.tolist(), bk.protein_sequence.tolist(),
                         scorer=Levenshtein.normalized_similarity) * 100
    best_idx = sim.argmax(axis=1)
    rbps["best_identity"] = sim.max(axis=1)
    rbps["best_boeckaerts_phage"] = bk.phage_ID.values[best_idx]
    rbps["best_boeckaerts_protein"] = bk.protein_ID.values[best_idx]

    def stratum(id_):
        if id_ >= 95:
            return "near-identical"
        if id_ >= 80:
            return "related"
        return "novel"
    rbps["stratum"] = rbps.best_identity.apply(stratum)

    print("\n--- RBP-level stratum sizes (n=%d) ---" % len(rbps))
    counts = rbps.stratum.value_counts()
    for s in ("novel", "related", "near-identical"):
        n = counts.get(s, 0)
        print(f"  {s:<15} {n:>4}  ({100*n/len(rbps):.1f}%)")

    # --- phage-level stratum: highest-risk stratum among a phage's own RBPs ---
    risk_rank = {"novel": 0, "related": 1, "near-identical": 2}
    rbps["risk"] = rbps.stratum.map(risk_rank)
    phage_stratum = rbps.groupby("phage_ID")["risk"].max().map(
        {v: k for k, v in risk_rank.items()})
    print(f"\n--- Phage-level stratum (n={len(phage_stratum)} phages with >=1 RBP) ---")
    print("(a phage's stratum = the riskiest stratum among its own RBPs)")
    pc = phage_stratum.value_counts()
    for s in ("novel", "related", "near-identical"):
        n = pc.get(s, 0)
        print(f"  {s:<15} {n:>4} phages")

    # --- pair-level stratum: every (host, phage) row in the primary interaction
    # table inherits its phage's stratum -- host identity plays no role ---
    inter = pd.read_csv(os.path.join(OUT_DIR, "interactions_LB_strict.csv"))
    inter["stratum"] = inter.phage.map(phage_stratum)
    n_no_rbp = inter.stratum.isna().sum()
    print(f"\n--- Pair-level stratum, interactions_LB_strict.csv (n={len(inter)} pairs) ---")
    if n_no_rbp:
        no_rbp_phages = sorted(inter[inter.stratum.isna()].phage.unique())
        print(f"  {n_no_rbp} pairs involve a phage with ZERO qualifying RBPs "
              f"(unscoreable, excluded from stratum counts): {no_rbp_phages}")
    pair_counts = inter.stratum.value_counts()
    n_scoreable = inter.stratum.notna().sum()
    for s in ("novel", "related", "near-identical"):
        n = pair_counts.get(s, 0)
        print(f"  {s:<15} {n:>5}  ({100*n/n_scoreable:.1f}% of scoreable pairs)")
    pos = inter[inter.label == 1]
    print(f"\n  Same breakdown, POSITIVES only (n={len(pos)}):")
    pos_counts = pos.stratum.value_counts()
    n_pos_scoreable = pos.stratum.notna().sum()
    for s in ("novel", "related", "near-identical"):
        n = pos_counts.get(s, 0)
        print(f"    {s:<15} {n:>4}  ({100*n/n_pos_scoreable:.1f}% of scoreable positives)")

    rbps.drop(columns=["risk"]).to_csv(
        os.path.join(OUT_DIR, "stage3_rbps_tagged.csv"), index=False)
    print(f"\nWrote {OUT_DIR}/stage3_rbps_tagged.csv ({len(rbps)} rows)")

    # --- embed ---
    print("\n" + "=" * 66)
    print("ESM-2 650M embedding (return_contacts=False)")
    print("=" * 66)
    import torch
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = model.to(device)

    embeddings = np.zeros((len(rbps), 1280), dtype=np.float32)
    from tqdm import tqdm
    for i, seq in enumerate(tqdm(rbps.protein_sequence.tolist(), desc="embedding")):
        data = [(f"rbp_{i}", seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        try:
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            rep = results["representations"][33]
            emb = rep[0, 1:len(seq) + 1].mean(0).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            with torch.no_grad():
                results = model.cpu()(batch_tokens.cpu(), repr_layers=[33], return_contacts=False)
            rep = results["representations"][33]
            emb = rep[0, 1:len(seq) + 1].mean(0).numpy()
            model = model.to(device)
        embeddings[i] = emb

    np.save(os.path.join(OUT_DIR, "stage3_rbp_embeddings.npy"), embeddings)
    print(f"Wrote {OUT_DIR}/stage3_rbp_embeddings.npy {embeddings.shape}")
    print("\nSTAGE 3 CHECKPOINT COMPLETE.")


if __name__ == "__main__":
    main()
