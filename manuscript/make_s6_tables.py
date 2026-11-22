#!/usr/bin/env python3
"""
Generate the two calibration tables for S6 Appendix from the all-versus-all
BLASTp / normalised-Levenshtein join.

Input : results/metric_calibration_pairs.csv
        columns: rbp_a, rbp_b, phage_a, phage_b, len_a, len_b,
                 blast_pident, blast_qcovs, blast_length, blast_evalue, nlev
        pairs with no BLASTp hit carry null BLAST fields and must be retained.

Output: S6_calibration_forward.tex   BLASTp identity bin -> nLev distribution
        S6_calibration_reverse.tex   nLev threshold      -> BLASTp distribution

    python make_s6_tables.py
"""

import numpy as np
import pandas as pd

INFILE = "results/metric_calibration_pairs.csv"
BINS = [(0, 30), (30, 40), (40, 50), (50, 60),
        (60, 70), (70, 80), (80, 90), (90, 100.01)]
NLEV_THRESHOLDS = [1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40]


def main():
    df = pd.read_csv(INFILE)
    hit = df[df["blast_pident"].notna()].copy()
    hit["cov_longer"] = (hit["blast_length"] /
                         hit[["len_a", "len_b"]].max(axis=1) * 100)

    print(f"{len(df):,} pairs; {len(hit):,} with a BLASTp hit "
          f"({100*len(hit)/len(df):.1f}%)")
    nohit = df[df["blast_pident"].isna()]["nlev"]
    print(f"no-hit nLev: median {nohit.median():.3f}  "
          f"p90 {nohit.quantile(0.9):.3f}  max {nohit.max():.3f}")

    # ---- forward: BLASTp bin -> nLev --------------------------------------
    rows = []
    for lo, hi in BINS:
        s = hit[(hit.blast_pident >= lo) & (hit.blast_pident < hi)]
        if len(s) == 0:
            continue
        label = f"$<$30\\%" if lo == 0 else (
            f"90--100\\%" if hi > 100 else f"{lo}--{hi}\\%")
        rows.append((label, len(s), s.blast_qcovs.median(),
                     s.cov_longer.median(), s.nlev.quantile(0.10),
                     s.nlev.median(), s.nlev.quantile(0.90)))

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{\rev{Normalised Levenshtein identity by BLASTp identity bin, "
        r"over all pairs of the 274 training RBPs with a BLASTp hit. Coverage "
        r"of longer is the alignment length as a fraction of the longer "
        r"sequence. The measure is near-flat at the unrelated-sequence floor "
        r"(0.221) until roughly 80\% BLASTp identity.}}",
        r"\label{tab:S6_forward}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"\textbf{BLASTp} & \textbf{Pairs} & \textbf{Med.\ query cov.} & "
        r"\textbf{Med.\ cov.\ of longer} & \textbf{nLev p10} & "
        r"\textbf{nLev median} & \textbf{nLev p90} \\", r"\midrule"]
    for lab, n, q, c, p10, med, p90 in rows:
        lines.append(f"{lab} & {n} & {q:.0f}\\% & {c:.0f}\\% & "
                     f"{p10:.3f} & {med:.3f} & {p90:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open("manuscript/S6_calibration_forward.tex", "w").write("\n".join(lines) + "\n")

    # ---- reverse: nLev threshold -> BLASTp ---------------------------------
    rows = []
    for t in NLEV_THRESHOLDS:
        s = df[df.nlev >= t]
        sh = s[s.blast_pident.notna()]
        covl = (sh["blast_length"] /
                sh[["len_a", "len_b"]].max(axis=1) * 100)
        rows.append((t, len(s), sh.blast_pident.median(),
                     sh.blast_qcovs.median(), covl.median(),
                     int(s.blast_pident.isna().sum())))

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{\rev{BLASTp identity among the RBP pairs merged at each "
        r"normalised Levenshtein threshold. Even at the loosest setting the "
        r"criterion merges only pairs that BLASTp calls near-identical over "
        r"full-length coverage.}}",
        r"\label{tab:S6_reverse}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"\textbf{nLev $\geq$} & \textbf{Pairs} & \textbf{Med.\ BLASTp \%} & "
        r"\textbf{Med.\ query cov.} & \textbf{Med.\ cov.\ of longer} & "
        r"\textbf{No BLASTp hit} \\", r"\midrule"]
    for t, n, pid, q, c, nh in rows:
        lines.append(f"{t:.2f} & {n} & {pid:.1f} & {q:.0f}\\% & "
                     f"{c:.0f}\\% & {nh} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open("manuscript/S6_calibration_reverse.tex", "w").write("\n".join(lines) + "\n")

    print("\nwrote S6_calibration_forward.tex and S6_calibration_reverse.tex")


if __name__ == "__main__":
    main()
