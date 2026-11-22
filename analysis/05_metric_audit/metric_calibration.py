"""
metric_calibration.py — BLASTp identity vs normalised Levenshtein on the 274 RBPs.

THE QUESTION
------------
Reviewer #4 asks for the sweep to extend to 30-40% identity, on the grounds
that 30-40% sequence similarity still predicts depolymerase specificity. Their
30-40% is BLASTp pident: local, over an aligned region. The sweep's thresholds
are normalised Levenshtein over whole sequences: global, and penalised for
length differences. These are not the same scale, and the conversion between
them has to be measured rather than asserted.

WHAT IT WRITES
--------------
results/metric_calibration_pairs.csv — one row per unordered pair of the 274
training RBPs (C(274,2) = 37,401), carrying the best BLASTp HSP for that pair
where one exists and the whole-sequence metric for every pair. Pairs with no
BLASTp hit keep null BLAST fields: how many there are is itself the result,
since it bounds how much of the identity range is populated at all.

Consumed by make_s6_tables.py (the S6 conversion tables) and by
analysis/02_ablation/make_blast_groupings.py (the BLASTp-grouped sweep).

BEST HSP PER PAIR
-----------------
BLASTp reports each pair twice, once per orientation, and may report several
HSPs per orientation. This keeps one row per unordered pair: highest bitscore,
then highest query coverage, then highest pident, then lexicographic on
(qseqid, sseqid). Every tie-break after bitscore is needed for determinism, not
cosmetic -- 1,600 of the 2,551 hit pairs have more than one row at their maximum
bitscore, and 886 of those differ in query coverage between the tied rows. A
rule that stops at bitscore leaves the coverage of a third of the file to the
sort's internal order. See the note in REPRODUCIBILITY.md.

Needs BLAST+ (the `genomics` conda env) and rapidfuzz. No GPU; takes a minute.

    python analysis/05_metric_audit/metric_calibration.py --check
    python analysis/05_metric_audit/metric_calibration.py -o /tmp/pairs.csv
"""
import argparse
import itertools
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein   # hard import: no difflib fallback

RBP = "data/RBPbase.csv"
OUT = "results/metric_calibration_pairs.csv"
WORK = "results/metric_audit_blast"
EVALUE = "1e-10"
CONDA_ENV = "genomics"
OUTFMT = "6 qseqid sseqid pident length qcovs qcovhsp qlen slen evalue bitscore"
# column names of the BLAST table, minus the leading format code
BLAST_COLS = OUTFMT.split()[1:]
HSP_COLS = BLAST_COLS[2:]

# Validation targets, from the committed CSV this script reconstructs.
EXPECT = {
    "n_pairs": 37401, "n_hit": 2551, "hit_frac": 6.82,
    "nohit_nlev_median": 0.2208, "nohit_nlev_p90": 0.2417,
    "nohit_nlev_max": 0.2739,
    "bin3040_n": 277, "bin3040_nlev_max": 0.344, "bin3040_nlev_median": 0.247,
    "nlev40_n": 921, "nlev40_pident_median": 94.5,
}


def conda_run(env, *cmd):
    """Run a BLAST+ binary inside a conda env, as klebphacol_stage5 does."""
    r = subprocess.run(["conda", "run", "-n", env, *cmd],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            f"`{' '.join(cmd[:1])}` failed in conda env '{env}' "
            f"(exit {r.returncode}).\n"
            f"  command: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r


def run_blast(rbp, workdir, force=False):
    """All-vs-all BLASTp of the RBPs against themselves. Returns the raw table."""
    os.makedirs(workdir, exist_ok=True)
    fasta = os.path.join(workdir, "rbps.fasta")
    db = os.path.join(workdir, "rbps")
    tsv = os.path.join(workdir, "allvsall.tsv")

    if os.path.exists(tsv) and not force:
        print(f"  reusing {tsv} (--force-blast to redo)")
    else:
        with open(fasta, "w") as fh:
            for pid, seq in zip(rbp["protein_ID"], rbp["protein_sequence"]):
                fh.write(f">{pid}\n{seq}\n")
        print(f"  wrote {len(rbp)} sequences to {fasta}")
        conda_run(CONDA_ENV, "makeblastdb", "-in", fasta, "-dbtype", "prot",
                  "-out", db)
        conda_run(CONDA_ENV, "blastp", "-query", fasta, "-db", db,
                  "-out", tsv, "-outfmt", OUTFMT, "-evalue", EVALUE)
        print(f"  blastp -evalue {EVALUE} -> {tsv}")

    b = pd.read_csv(tsv, sep="\t", names=BLAST_COLS)
    n_self = int((b.qseqid == b.sseqid).sum())
    b = b[b.qseqid != b.sseqid].copy()
    print(f"  {len(b):,} HSP rows after dropping {n_self} self-hits")
    return b


TIE_BREAKS = {
    # Highest coverage among the tied orientations. Matches the semantics of
    # make_blast_groupings.py's COV_FLOOR test ("is there an orientation in
    # which this HSP covers at least half the query"), so it is the default.
    "qcovs": (["bitscore", "qcovs", "pident", "qseqid", "sseqid"],
              [False, False, False, True, True]),
    # Orientation chosen by name alone: no coverage bias either way.
    "lex": (["bitscore", "qseqid", "sseqid"], [False, True, True]),
}


def best_hsp_per_pair(b, tie_break="qcovs"):
    """
    One row per unordered pair. Fully ordered tie-break -- see the module
    docstring on why bitscore alone is not enough.
    """
    b = b.copy()
    b["rbp_a"] = np.minimum(b.qseqid, b.sseqid)
    b["rbp_b"] = np.maximum(b.qseqid, b.sseqid)
    tied = (b.groupby(["rbp_a", "rbp_b"]).bitscore.transform("max") == b.bitscore)
    n_tied = int((b[tied].groupby(["rbp_a", "rbp_b"]).size() > 1).sum())
    cols, asc = TIE_BREAKS[tie_break]
    best = (b.sort_values(cols, ascending=asc, kind="stable")
             .drop_duplicates(["rbp_a", "rbp_b"]))
    print(f"  {len(best):,} unordered pairs with a hit "
          f"({n_tied:,} had >1 row at their best bitscore)")
    return best.set_index(["rbp_a", "rbp_b"])


def build(rbp, best):
    """Every unordered pair, in RBPbase order, with both metrics."""
    ids = rbp["protein_ID"].tolist()
    seq = dict(zip(rbp["protein_ID"], rbp["protein_sequence"]))
    phage = dict(zip(rbp["protein_ID"], rbp["phage_ID"]))

    keys = [tuple(sorted(p)) for p in itertools.combinations(ids, 2)]
    a = [k[0] for k in keys]
    b = [k[1] for k in keys]

    df = pd.DataFrame({
        "rbp_a": a, "rbp_b": b,
        "phage_a": [phage[x] for x in a], "phage_b": [phage[x] for x in b],
        "len_a": [len(seq[x]) for x in a], "len_b": [len(seq[x]) for x in b],
    })

    # Left join: a pair with no HSP keeps NaN BLAST fields rather than vanishing.
    hsp = best.reindex(pd.MultiIndex.from_tuples(keys))
    for col in ["pident", "qcovs", "qcovhsp", "length", "evalue", "bitscore"]:
        df["blast_" + col] = hsp[col].values

    # Coverage of the LONGER sequence: the local-vs-global quantity the brief
    # asks for, and the one make_s6_tables.py reports as cov_longer.
    longer = np.maximum(df.len_a, df.len_b)
    df["blast_scovs_derived"] = (df.blast_length / longer * 100).round(1)

    df["nlev"] = [Levenshtein.normalized_similarity(seq[x], seq[y])
                  for x, y in zip(a, b)]

    return df[["rbp_a", "rbp_b", "phage_a", "phage_b", "len_a", "len_b",
               "blast_pident", "blast_qcovs", "blast_qcovhsp", "blast_length",
               "blast_scovs_derived", "blast_evalue", "blast_bitscore", "nlev"]]


def check(df):
    """Compare against the committed CSV's established numbers."""
    hit = df[df.blast_pident.notna()]
    nohit = df[df.blast_pident.isna()]
    b3040 = hit[(hit.blast_pident >= 30) & (hit.blast_pident < 40)]
    nl40 = df[df.nlev >= 0.40]

    got = {
        "n_pairs": len(df), "n_hit": len(hit),
        "hit_frac": round(100.0 * len(hit) / len(df), 2),
        "nohit_nlev_median": round(nohit.nlev.median(), 4),
        "nohit_nlev_p90": round(nohit.nlev.quantile(0.90), 4),
        "nohit_nlev_max": round(nohit.nlev.max(), 4),
        "bin3040_n": len(b3040),
        "bin3040_nlev_max": round(b3040.nlev.max(), 3),
        "bin3040_nlev_median": round(b3040.nlev.median(), 3),
        "nlev40_n": len(nl40),
        "nlev40_pident_median": round(nl40.blast_pident.median(), 1),
    }
    print("\nValidation against the committed CSV")
    bad = []
    for k, want in EXPECT.items():
        ok = got[k] == want
        print(f"  {'OK  ' if ok else 'DIFF'} {k:22s} got {got[k]:<10} expect {want}")
        if not ok:
            bad.append(k)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("-o", "--out", default=OUT,
                    help=f"output CSV (default {OUT})")
    ap.add_argument("--workdir", default=WORK,
                    help="where the FASTA, BLAST db and raw TSV go")
    ap.add_argument("--force-blast", action="store_true",
                    help="re-run BLASTp even if the raw TSV is present")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out if it exists")
    ap.add_argument("--tie-break", choices=sorted(TIE_BREAKS), default="qcovs",
                    help="how to choose between reciprocal HSPs tied on "
                         "bitscore (default qcovs). Affects blast_qcovs only, "
                         "and through it the median-coverage column of S6 "
                         "forward; see REPRODUCIBILITY.md.")
    ap.add_argument("--check", action="store_true",
                    help="validate and report; do not write")
    args = ap.parse_args()

    if not os.path.exists(RBP):
        raise SystemExit(f"need {RBP} (run from the repo root)")
    if os.path.exists(args.out) and not (args.force or args.check):
        raise SystemExit(
            f"{args.out} exists. Re-run with --check to validate against it "
            "without writing, or --force to overwrite it.")

    rbp = pd.read_csv(RBP)
    print(f"{len(rbp)} RBPs, {rbp.phage_ID.nunique()} phages")

    raw = run_blast(rbp, args.workdir, force=args.force_blast)
    df = build(rbp, best_hsp_per_pair(raw, args.tie_break))

    bad = check(df)

    if args.check:
        print("\n--check: nothing written.")
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nWrote {args.out} ({len(df):,} rows)")

    if bad:
        print(f"\n{len(bad)} target(s) differ: {', '.join(bad)}. "
              "Reported, not silently accepted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
