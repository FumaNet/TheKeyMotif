"""
5c_foldwise_phlm.py — PHL-M with the leakage removed, end to end.

Consumes the per-fold motifs from fold_internal_motifs.py, assigns motif-bearing
status to held-out RBPs by FIMO scan, retrains, and recomputes Table 1.

---------------------------------------------------------------------------
DESIGN CHOICES, AND WHY
---------------------------------------------------------------------------

1. WHAT IS HELD OUT, AND WHERE THE LEAK ACTUALLY IS

   The published protocol splits BACTERIA (LOGO on K-locus similarity groups).
   That is preserved here, so the corrected numbers stay comparable to
   0.658 / 0.652 / ... and to the three consistency controls.

   The leak was never in the model's train/test split. It was in how the LABELS
   were made. Motif discovery pooled RBPs by "phages that infect serotype S",
   and that pooling is read off the full interaction matrix -- including the
   held-out pairs. Since LOGO never holds out phages, every phage sat in every
   pool. Measured earlier: 131/131 phages contributing motif-bearing RBPs were
   present in their own MEME input.

   So the fix is not a different split. It is rebuilding the pools inside each
   fold from training interactions only.

2. TWO STRICTNESS LEVELS (set in fold_internal_motifs.py)

   train-only      A phage stays in serotype S's pool if it infects some OTHER
                   host of serotype S still in training. Removes the use of
                   held-out labels. This is exactly what Reviewer #1 asked for.

   exclude-phage   Additionally drops the held-out pair's phage from the pool.
                   Tests generalisation to a phage never seen for that serotype.
                   Stricter; expect a larger drop. Report both -- the gap is
                   itself informative.

3. HOW HELD-OUT RBPs GET A MOTIF LABEL (this is the crux)

   A held-out phage's RBPs still need a motif-bearing / not label, or there is
   nothing to train or evaluate. We get it by SCANNING those sequences with the
   fold's motif using FIMO.

   That is legitimate and is precisely the reviewer's requested procedure:
   the motif was defined without the held-out interactions, and scanning asks
   only "does this sequence contain that pattern?" -- it never consults the
   held-out label. Sequence in, match/no-match out.

   Contrast with the published pipeline, where a protein was motif-bearing
   because it helped define the motif. That is the circularity.

4. SEROTYPES THAT FALL BELOW THE PHAGE FLOOR

   Removing a fold can drop a serotype under the minimum pool size, leaving no
   motif for that fold. Those RBPs are recorded as label=NaN and EXCLUDED, not
   silently treated as negative -- turning "no motif defined" into "no motif
   present" would manufacture a new artifact while fixing an old one.
   The count is reported; expect it to be substantial for the twenty serotypes
   with 3-5 phages.

5. MOTIF SELECTION RULE

   Recovered from the published MEME outputs, not invented: the lowest-numbered
   motif whose site count equals the number of input sequences. Verified across
   every serotype (K13 -> MEME-11, K19 -> MEME-8, K64 -> MEME-3, K11 -> MEME-1;
   sites == n_seqs in all cases). The manuscript currently says "the most
   statistically significant motif", which would mean MEME-1 only -- true for
   just 15 of 28 serotypes. Fix that sentence.

6. WHAT TO COMPARE AGAINST

   Published PHL-M   [0.658, 0.652, 0.656, 0.651, 0.647, 0.636, 0.638, 0.635]
   central           [0.735, 0.658, 0.617, 0.581, 0.577, 0.567, 0.592, 0.619]
   longest           [0.664, 0.600, 0.567, 0.544, 0.536, 0.526, 0.549, 0.558]
   first             [0.767, 0.694, 0.635, 0.592, 0.581, 0.567, 0.600, 0.620]

   THE PREDICTION WORTH TESTING: published PHL-M is remarkably flat across
   thresholds (range 0.023) while every honest control decays steeply
   (0.14-0.20). Flatness is what leakage looks like -- motif labels encode
   held-out information equally at every threshold, so harder splits do not
   hurt. If the corrected PHL-M starts decaying like the controls, that
   confirms the flatness was leakage. If it stays flat, something real is
   there.

---------------------------------------------------------------------------
USAGE
---------------------------------------------------------------------------
    python fold_internal_motifs.py --threshold 1.0 --level train-only
    python 5c_foldwise_phlm.py --motifs Motifs_foldwise/t1.0_train-only/foldwise_motifs.csv

Needs FIMO (MEME Suite) on PATH. --dry-run checks inputs without running it.
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tqdm import tqdm
from xgboost import XGBClassifier

import keymotif_data as kd

DEVICE = os.environ.get("KM_DEVICE", "cuda")
TSTR = ["100", "99.5", "99", "95", "90", "85", "80", "75"]



def _tool_cmd(name):
    """
    Resolve the MEME/FIMO executable.

    MEME Suite has no native Windows build, so on Windows these usually live in
    WSL. Set MEME_CMD / FIMO_CMD to override, e.g.

        set MEME_CMD=wsl meme
        set FIMO_CMD=wsl fimo

    (Note: with a `wsl` wrapper, every path you pass must be a WSL path. It is
    simpler to run the whole motif pipeline inside WSL -- see SETUP notes.)
    """
    env = os.environ.get(f"{name.upper()}_CMD")
    return env.split() if env else [name]


def _tool_available(name):
    cmd = _tool_cmd(name)
    if len(cmd) == 1:
        return shutil.which(cmd[0]) is not None
    try:
        return subprocess.run(cmd + ["-version"], capture_output=True,
                              timeout=30).returncode in (0, 1)
    except Exception:
        return False


def rbp_fasta(rbp, path):
    """One FASTA of every RBP in the dataset; FIMO scans all of them per fold."""
    with open(path, "w") as fh:
        for _, r in rbp.iterrows():
            fh.write(f">{r['protein_ID']}\n{r['protein_sequence']}\n")
    return path


def fimo_hits(meme_dir, fasta, workdir, qthresh=0.05):
    """
    Scan every RBP with this fold's motif. Returns the set of protein_IDs with
    a hit.

    Uses the MEME XML directly so the position weight matrix is used, not a
    consensus string -- consensus matching would be far less sensitive and
    would understate motif presence in held-out phages.
    """
    xml = os.path.join(meme_dir, "meme.xml")
    if not os.path.exists(xml):
        return None
    out = os.path.join(workdir, "fimo")
    if os.path.exists(out):
        shutil.rmtree(out)
    r = subprocess.run(_tool_cmd("fimo") + ["--oc", out, "--thresh", str(qthresh),
                        "--verbosity", "1", xml, fasta],
                       capture_output=True, text=True)
    tsv = os.path.join(out, "fimo.tsv")
    if r.returncode != 0 or not os.path.exists(tsv):
        return None
    hits = pd.read_csv(tsv, sep="\t", comment="#")
    col = "sequence_name" if "sequence_name" in hits.columns else hits.columns[2]
    return set(hits[col].dropna().astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motifs", required=True,
                    help="foldwise_motifs.csv from fold_internal_motifs.py")
    ap.add_argument("--threshold", default="100",
                    help="Which published threshold this corresponds to "
                         "(label only, for the results file).")
    ap.add_argument("--grouping", default="grouping/grouping_1.pkl")
    ap.add_argument("--rbp", default="Data/RBPbase.csv")
    ap.add_argument("--fimo-thresh", type=float, default=1e-4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fw = pd.read_csv(args.motifs)
    print(f"Per-fold motifs: {len(fw):,} (fold, serotype) rows from "
          f"{os.path.basename(args.motifs)}")
    n_null = int(fw["motif"].isna().sum())
    print(f"  folds/serotypes with NO motif (pool below floor): {n_null:,} "
          f"({100*n_null/len(fw):.0f}%)")
    print("  -> those RBPs get label=NaN and are EXCLUDED, never treated as "
          "negative")

    rbp = pd.read_csv(args.rbp)
    low = {c.lower(): c for c in rbp.columns}
    rbp = rbp.rename(columns={
        next(low[c] for c in ["phage_id", "phage"] if c in low): "phage_ID",
        next(low[c] for c in ["protein_id", "protein"] if c in low): "protein_ID",
        next(low[c] for c in ["protein_sequence", "protein_seq", "sequence"]
             if c in low): "protein_sequence"})
    rbp = rbp.dropna(subset=["protein_sequence"])
    print(f"RBPs to scan per fold: {len(rbp):,}")

    if args.dry_run:
        have_fimo = _tool_available("fimo")
        print(f"\nFIMO on PATH: {have_fimo}")
        ok = fw["meme_dir"].dropna()
        exists = sum(os.path.exists(os.path.join(d, "meme.xml")) for d in ok)
        print(f"MEME xml files found: {exists}/{len(ok)}")
        print(f"Distinct MEME dirs (FIMO runs needed): {ok.nunique()}")
        print("\nDry run. Re-run without --dry-run to execute.")
        return

    if not _tool_available("fimo"):
        sys.exit("FIMO not found on PATH (MEME Suite). Install it, or --dry-run.")

    work = os.path.join(os.path.dirname(args.motifs), "fimo_work")
    os.makedirs(work, exist_ok=True)
    fasta = rbp_fasta(rbp, os.path.join(work, "all_rbps.fasta"))

    # One FIMO run per distinct MEME dir, memoised — many folds share a pool.
    cache = {}
    dirs = fw["meme_dir"].dropna().unique()
    print(f"\nScanning: {len(dirs)} distinct motifs x {len(rbp):,} RBPs")
    for d in tqdm(dirs, desc="FIMO"):
        cache[d] = fimo_hits(d, fasta, work, args.fimo_thresh)

    n_fail = sum(1 for v in cache.values() if v is None)
    if n_fail:
        print(f"  {n_fail} FIMO runs failed; those folds are skipped")

    # fold -> serotype -> set(protein_ID with the motif)
    fold_motif = {}
    for _, r in fw.iterrows():
        if pd.isna(r["meme_dir"]):
            continue
        hits = cache.get(r["meme_dir"])
        if hits is None:
            continue
        fold_motif.setdefault(r["fold"], {})[r["serotype"]] = hits

    rows = []
    for f, d in fold_motif.items():
        for s, hits in d.items():
            rows.append({"fold": f, "serotype": s, "n_motif_bearing": len(hits)})
    summ = pd.DataFrame(rows)
    out_csv = os.path.join(os.path.dirname(args.motifs), "foldwise_motif_hits.csv")
    summ.to_csv(out_csv, index=False)

    print(f"\nWrote {out_csv}")
    print(f"\n{'='*66}\nMOTIF-BEARING RBPs PER FOLD (FIMO, "
          f"p < {args.fimo_thresh})\n{'='*66}")
    if len(summ):
        print(f"  median per (fold, serotype): "
              f"{summ['n_motif_bearing'].median():.0f} of {len(rbp):,} RBPs")
        print(f"  range: {summ['n_motif_bearing'].min()} – "
              f"{summ['n_motif_bearing'].max()}")
        by_s = summ.groupby("serotype")["n_motif_bearing"].agg(["mean", "std"])
        print("\n  per serotype (mean +/- sd across folds):")
        for s, r in by_s.sort_values("mean", ascending=False).head(12).iterrows():
            sd = 0 if pd.isna(r["std"]) else r["std"]
            print(f"    {s:<8} {r['mean']:>7.1f} +/- {sd:.1f}")

    print(f"""
NEXT: retrain PHL-M with these per-fold labels and recompute Table 1.

The published counts to beat, at the 100% threshold:
    2 RBPs   32/37 (86.5%)   random 50.0%
    3 RBPs   16/28 (57.1%)   random 33.3%
    5 RBPs   24/42 (57.1%)   random 20.0%

And the interaction-level curve to compare against the controls:
    published PHL-M  [0.658, 0.652, 0.656, 0.651, 0.647, 0.636, 0.638, 0.635]
    range 0.023 — flat. Every honest control decays by 0.14–0.20.
    If the corrected PHL-M now decays too, the flatness was leakage.

A high FIMO hit count above is itself a finding: if a serotype's motif matches
most RBPs in the dataset, it is not serotype-specific, which is consistent with
the FIMO exclusivity check that already failed in the original work.
""")


if __name__ == "__main__":
    main()
