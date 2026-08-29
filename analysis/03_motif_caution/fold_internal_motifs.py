#!/usr/bin/env python3
"""
fold_internal_motifs.py — leakage-free motif discovery for TheKeyMotif.

WHAT THIS FIXES
---------------
In the published pipeline, MEME was run once per serotype over the RBPs of ALL
phages infecting that serotype. That pooling is read off the full interaction
matrix, including the pairs LOGO later holds out. Since LOGO splits bacteria
and not phages, the held-out pair's own phage sits in the MEME input, so the
"motif-bearing" label is not independent of the held-out interaction.

This script rebuilds the serotype pools inside each fold, using only training
interactions, and re-derives motif membership per fold.

WHY IT IS AFFORDABLE
--------------------
Naively this is (n_folds x n_serotypes) MEME runs -- 185 x 28 at the 100%
threshold. But the pool for serotype S changes only when the held-out group
contains a host OF SEROTYPE S whose phages contribute uniquely. Every other
serotype keeps the global pool. Memoising on frozenset(pool) collapses the work
to roughly (number of distinct pools), typically a few hundred runs total.

Run --dry-run first: it reports the exact number of MEME invocations needed
before you commit any compute.

TWO STRICTNESS LEVELS
---------------------
  --level train-only  (default; what Reviewer #1 asked for)
      Pools are built from training-fold interactions only. A phage stays in
      the pool if it infects some OTHER host of that serotype still in training.
      Removes the held-out-label leak.

  --level exclude-phage  (stricter, and closer to the biological claim)
      Additionally drops the held-out pair's phage from the pool entirely.
      Tests whether the motif generalises to a phage never seen for that
      serotype. Expect a larger drop. Report both.

MOTIF SELECTION RULE
--------------------
Recovered from your own stored MEME outputs: the retained motif is the
LOWEST-NUMBERED motif whose site count equals the number of input sequences
(K13 -> MEME-11, K19 -> MEME-8, K64 -> MEME-3, K11 -> MEME-1; sites == n_seqs
in every case). That is "most significant motif present in every phage in the
pool", not "most significant motif" as the manuscript currently states.

Note what that rule implies: motif-bearing status is near-guaranteed for pool
members by construction. That is precisely why the fold-internal version is the
only meaningful test.

USAGE
-----
    python fold_internal_motifs.py --dry-run
    python fold_internal_motifs.py --threshold 1.0 --level train-only
    python fold_internal_motifs.py --threshold 1.0 --level exclude-phage

Requires MEME Suite on PATH (meme, fimo). Tested against MEME 5.5.7, the
version recorded in your stored .meme files.
"""

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from collections import defaultdict

import pandas as pd

DATA = "Data"
OUTDIR = "Motifs_foldwise"
MIN_COVERAGE = 0.8
MIN_PHAGES = 6          # pool size floor; the published run used 6 for figures, 3 for training
MINW, MAXW = 21, 99   # override with --minw/--maxw     # as in the published MEME call
NMOTIFS = 15          # override with --nmotifs            # must exceed 11: K13's retained motif was MEME-11
MOD = "zoops"         # override with --mod           # zero-or-one occurrence per sequence

GROUPING = {
    1.0: "grouping/grouping_1.pkl", 0.995: "grouping/grouping_995.pkl",
    0.99: "grouping/grouping_990.pkl", 0.95: "grouping/grouping_950.pkl",
    0.9: "grouping/grouping_900.pkl", 0.85: "grouping/grouping_850.pkl",
    0.8: "grouping/grouping_800.pkl", 0.75: "grouping/grouping_750.pkl",
}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_inputs():
    """Interactions, serotypes, and RBP sequences."""
    inter = pd.read_csv(f"{DATA}/phage_host_interactions.csv")
    inter = inter.melt(id_vars=["Unnamed: 0"], var_name="phage_ID",
                       value_name="label").rename(columns={"Unnamed: 0": "accession"})
    inter = inter.dropna(subset=["label"])
    inter["accession"] = inter["accession"].astype(str)

    sero = pd.read_csv(f"{DATA}/kaptive_results.tsv", sep="\t")
    keep = [c for c in ["Assembly", "Best match type", "Match confidence"]
            if c in sero.columns]
    sero = sero[keep].rename(columns={"Assembly": "accession",
                                      "Best match type": "serotype",
                                      "Match confidence": "confidence"})
    sero["accession"] = sero["accession"].astype(str)

    # Kaptive emits "Capsule null" / "unknown (KL...)" for hosts whose capsule
    # locus could not be typed. Those are not biological serotypes -- pooling
    # RBPs by "phages infecting untypeable hosts" groups unrelated phages and
    # invents a motif for a non-existent capsule class. The published analysis
    # excluded them (the filters exist, commented out, in
    # 2_max_max_original_sero.py), and the repo's own analysis file contains
    # only real K-types. Match that.
    n0 = len(sero)
    bad = (sero["serotype"].isna()
           | sero["serotype"].astype(str).str.contains(
               "null|unknown|none", case=False, na=True))
    if "confidence" in sero.columns:
        bad |= sero["confidence"].astype(str).str.strip().ne("Typeable")
    dropped = sero[bad]
    if len(dropped):
        labels = dropped["serotype"].astype(str).value_counts().head(4)
        print(f"  excluding {len(dropped)} of {n0} hosts with untypeable "
              f"capsules: {dict(labels)}")
    sero = sero[~bad]

    # RBPbase.csv from the PHL Zenodo record: one row per RBP
    rbp = pd.read_csv(f"{DATA}/RBPbase.csv")
    low = {c.lower(): c for c in rbp.columns}

    def pick(cands, what):
        for c in cands:
            if c in low:
                return low[c]
        raise SystemExit(f"RBPbase.csv: no {what} column. Has: "
                         f"{list(rbp.columns)[:10]}")

    rbp = rbp.rename(columns={
        pick(["phage_id", "phage"], "phage id"): "phage_ID",
        pick(["protein_id", "protein"], "protein id"): "protein_ID",
        pick(["protein_sequence", "protein_seq", "sequence"],
             "protein sequence"): "protein_sequence"})
    rbp = rbp.dropna(subset=["protein_sequence"])

    return inter, sero, rbp


def concat_rbps(rbp):
    """
    Concatenate each phage's RBPs into one sequence, as the published pipeline
    did before submitting to MEME, and keep the offsets so a motif hit can be
    mapped back to the RBP it fell in.
    """
    recs, offsets = {}, {}
    for ph, g in rbp.groupby("phage_ID"):
        seq, off, pos = "", [], 0
        for _, r in g.iterrows():
            s = str(r["protein_sequence"])
            off.append((r["protein_ID"], pos, pos + len(s)))
            seq += s
            pos += len(s)
        recs[ph] = seq
        offsets[ph] = off
    return recs, offsets


# ---------------------------------------------------------------------------
# pool construction
# ---------------------------------------------------------------------------

def serotype_pools(inter, sero, keep_mask, exclude_phages=frozenset()):
    """
    serotype -> frozenset of phages having a POSITIVE interaction, within the
    rows selected by keep_mask, to a host of that serotype.
    """
    df = inter[keep_mask & (inter["label"] == 1)].merge(sero, on="accession", how="left")
    pools = defaultdict(set)
    for s, g in df.groupby("serotype"):
        pool = set(g["phage_ID"]) - set(exclude_phages)
        if len(pool) >= MIN_PHAGES:
            pools[s] = frozenset(pool)
    return dict(pools)


def plan_folds(inter, sero, groups, level, kfold=None):
    """
    Yield (fold_id, held_out, {serotype: pool}) per fold.

    The split is always on the BACTERIAL side, exactly as the published LOGO
    protocol is, so the corrected numbers stay comparable. What changes is only
    how the serotype pools are built: from training interactions instead of all
    of them.

    With kfold=K the bacterial groups are dealt round-robin into K folds. That
    holds out MORE per fold, so pools are SMALLER and more serotypes fall below
    the phage floor. Cheaper, stricter, not comparable to published numbers.
    """
    inter = inter.copy()
    inter["group_loci"] = inter["accession"].map(groups)
    inter = inter.dropna(subset=["group_loci"])

    uniq = sorted(inter["group_loci"].unique())
    if kfold:
        # round-robin keeps fold sizes even without shuffling (deterministic)
        assign = {g: i % kfold for i, g in enumerate(uniq)}
        inter["fold"] = inter["group_loci"].map(assign)
        fold_ids = list(range(kfold))
    else:
        inter["fold"] = inter["group_loci"]
        fold_ids = uniq

    for fold_id, g in enumerate(fold_ids):
        test = inter["fold"] == g
        train = ~test

        excl = frozenset()
        if level == "exclude-phage":
            # phages involved in any held-out POSITIVE pair
            excl = frozenset(inter[test & (inter["label"] == 1)]["phage_ID"])

        yield fold_id, g, serotype_pools(inter, sero, train, excl)


# ---------------------------------------------------------------------------
# MEME
# ---------------------------------------------------------------------------


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


def pool_key(serotype, pool):
    h = hashlib.md5("|".join(sorted(pool)).encode()).hexdigest()[:10]
    return f"{serotype}_{len(pool)}_{h}"


def run_meme(serotype, pool, seqs, workdir):
    """Run MEME on a pool and return the retained motif, or None."""
    key = pool_key(serotype, pool)
    out = os.path.join(workdir, key)
    kept = os.path.join(out, "kept_motif.txt")
    if os.path.exists(kept):
        return json.load(open(kept))

    os.makedirs(out, exist_ok=True)
    fa = os.path.join(out, "input.fasta")
    usable = [ph for ph in sorted(pool) if ph in seqs]
    if len(usable) < MIN_PHAGES:
        return None
    with open(fa, "w") as fh:
        for ph in usable:
            fh.write(f">{ph}\n{seqs[ph]}\n")

    cmd = _tool_cmd("meme") + [fa, "-protein", "-oc", out, "-mod", MOD,
           "-nmotifs", str(NMOTIFS), "-minw", str(MINW), "-maxw", str(MAXW)]
    r = subprocess.run(cmd, capture_output=True, text=True)

    # MEME returns non-zero when its optional HTML converter (meme_xml_to_html)
    # fails, even though motif discovery itself succeeded and meme.txt/meme.xml
    # were written. Judge by the outputs, not the exit code.
    txt = os.path.join(out, "meme.txt")
    xml = os.path.join(out, "meme.xml")
    if not (os.path.exists(txt) or os.path.exists(xml)):
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        print(f"    MEME produced no output for {key}: "
              f"{tail[-1][:110] if tail else 'no message'}")
        return None
    if r.returncode != 0:
        print(f"    [{key}] MEME exit {r.returncode} but motifs written "
              "(HTML converter only) — continuing")

    motif = select_motif(os.path.join(out, "meme.txt"), len(usable),
                         min_coverage=MIN_COVERAGE)
    if motif is not None:
        motif["meme_dir"] = out          # FIMO needs the .xml from this dir
    with open(kept, "w") as fh:
        json.dump(motif, fh)
    return motif


def select_motif(meme_txt, n_seqs, min_coverage=0.8, verbose=True):
    """
    Retain the motif with the WIDEST coverage of the input pool; ties broken by
    lowest MEME rank (i.e. best E-value ordering).

    WHY NOT "sites == n_seqs"
    -------------------------
    Every motif retained in the published run happened to cover all its input
    sequences, so an exact-equality rule reproduced those choices. But exact
    equality is brittle: coverage shifts with MEME version and parameters, and
    on the fold-internal pools it rejected everything. In the K64 pool (10
    sequences) the best motif reached 9 -- and that motif,
    GADASFYFEEYVGTEHRAIEYMDGFGRT, shares its core with the published
    YVEEYVGTEHRAIIYMDGFGREDAWSFR. The published element was found; the rule
    threw it away.

    Max-coverage reduces to the published behaviour whenever a fully covering
    motif exists (it is then the maximum), and degrades sensibly when none
    does. `min_coverage` guards against retaining a motif present in only a
    couple of phages, which would not be a shared element in any useful sense.

    Returns the motif dict with a `coverage` field, so partial coverage is
    visible downstream instead of silently passing as equivalent.
    """
    if not os.path.exists(meme_txt):
        return None
    import re
    txt = open(meme_txt).read()

    found = []
    for m in re.finditer(
            r"MOTIF\s+(\S+)\s+MEME-(\d+)(.{0,400}?)E-value\s*=\s*(\S+)",
            txt, re.S):
        sites = re.search(r"sites\s*=\s*(\d+)", m.group(3))
        if sites:
            found.append({"consensus": m.group(1), "rank": int(m.group(2)),
                          "sites": int(sites.group(1)), "evalue": m.group(4)})
    if not found:
        if verbose:
            print(f"      no motifs parsed from {meme_txt}")
        return None

    best = sorted(found, key=lambda d: (-d["sites"], d["rank"]))[0]
    best["coverage"] = best["sites"] / n_seqs if n_seqs else 0.0

    if best["coverage"] < min_coverage:
        if verbose:
            got = ", ".join(f"MEME-{f['rank']}:{f['sites']}" for f in found[:6])
            print(f"      best coverage only {best['sites']}/{n_seqs} "
                  f"(< {min_coverage:.0%}); discarding. sites: {got}")
        return None
    return best


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    global MIN_PHAGES, MIN_COVERAGE, NMOTIFS, MOD, MINW, MAXW
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=1.0, choices=sorted(GROUPING))
    ap.add_argument("--level", choices=["train-only", "exclude-phage"],
                    default="train-only")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report how many distinct MEME runs are needed, then exit.")
    ap.add_argument("--min-phages", type=int, default=MIN_PHAGES)
    ap.add_argument("--nmotifs", type=int, default=NMOTIFS,
                    help="Motifs MEME reports. The covering motif can sit deep: "
                         "the published K64 motif came back at rank 12 here.")
    ap.add_argument("--mod", default=MOD, choices=["zoops", "anr", "oops"],
                    help="MEME site model. oops forces one site per sequence "
                         "(coverage always 100%%), zoops allows zero or one.")
    ap.add_argument("--minw", type=int, default=MINW)
    ap.add_argument("--maxw", type=int, default=MAXW,
                    help="Large maxw lets long low-coverage motifs outrank the "
                         "short conserved one. Try 30-50.")
    ap.add_argument("--tag", default=None,
                    help="Suffix for the output dir, to keep sweeps separate.")
    ap.add_argument("--min-coverage", type=float, default=0.8, metavar="F",
                    help="Retain a motif only if it covers at least this "
                         "fraction of the pool (default 0.8).")
    ap.add_argument("--kfold", type=int, default=None, metavar="K",
                    help="Group the bacterial LOGO groups into K folds instead "
                         "of holding out one at a time. Cheaper, but STRICTER "
                         "(more held out per fold -> smaller motif pools), and "
                         "not comparable to the published LOGO numbers. Use as "
                         "a robustness check, not as the primary analysis.")
    args = ap.parse_args()

    MIN_PHAGES = args.min_phages
    MIN_COVERAGE = args.min_coverage
    NMOTIFS = args.nmotifs
    MOD = args.mod
    MINW, MAXW = args.minw, args.maxw

    inter, sero, rbp = load_inputs()
    seqs, offsets = concat_rbps(rbp)

    # Some phages appear in the interaction matrix but have no RBP in
    # RBPbase.csv (no adsorption protein was detected for them). They cannot
    # contribute a sequence to MEME, so they must be removed BEFORE pools are
    # built -- otherwise a pool counted as 6 phages may hold only 5 usable
    # ones, and the minimum-pool-size rule is applied to the wrong number.
    have = set(seqs)
    all_ph = set(inter["phage_ID"].unique())
    missing = sorted(all_ph - have)
    if missing:
        print(f"  {len(missing)} of {len(all_ph)} phages have no RBP in "
              f"RBPbase.csv and are excluded from motif pools:")
        print(f"    {', '.join(missing[:8])}"
              f"{' ...' if len(missing) > 8 else ''}")
        inter = inter[inter["phage_ID"].isin(have)]
    print(f"  phages usable for motif discovery: "
          f"{inter['phage_ID'].nunique()} of {len(all_ph)}")
    groups = pickle.load(open(GROUPING[args.threshold], "rb"))

    folds = list(plan_folds(inter, sero, groups, args.level, args.kfold))

    # --- what actually has to be computed ---
    unique, per_fold = {}, []
    for fold_id, g, pools in folds:
        keys = []
        for s, pool in pools.items():
            k = pool_key(s, pool)
            unique[k] = (s, pool)
            keys.append(k)
        per_fold.append((fold_id, g, keys))

    naive = sum(len(p[2]) for p in per_fold)
    print(f"threshold      {args.threshold}")
    print(f"level          {args.level}")
    print(f"min pool size  {MIN_PHAGES}")
    print(f"folds          {len(folds)}")
    print(f"naive runs     {naive:,}  (folds x serotypes)")
    print(f"distinct runs  {len(unique):,}  <- what memoisation actually costs")
    if naive:
        print(f"saving         {100 * (1 - len(unique) / naive):.1f}%")

    # how many serotypes fall below the floor once the fold is removed
    lost = defaultdict(int)
    allser = set(serotype_pools(inter, sero,
                                pd.Series(True, index=inter.index)).keys())
    for fold_id, g, pools in folds:
        for s in allser - set(pools):
            lost[s] += 1
    if lost:
        print("\nserotypes dropping below the pool floor in at least one fold:")
        for s, n in sorted(lost.items(), key=lambda kv: -kv[1]):
            print(f"  {s:<8} loses its motif in {n}/{len(folds)} folds")
        print("  -> RBPs of these serotypes get NO motif label in those folds.")
        print("     Record that explicitly; do not silently treat them as negative.")

    if args.dry_run:
        print("\nDry run only. Re-run without --dry-run to execute MEME.")
        return

    if not _tool_available("meme"):
        sys.exit("MEME not runnable. Set MEME_CMD, or run inside WSL. See --dry-run.")

    os.makedirs(OUTDIR, exist_ok=True)
    suffix = args.tag or (f"m{MOD}_n{NMOTIFS}_w{MINW}-{MAXW}"
                          if (MOD, NMOTIFS, MAXW) != ("zoops", 15, 99) else "")
    work = os.path.join(OUTDIR, f"t{args.threshold}_{args.level}"
                        + (f"_{suffix}" if suffix else ""))
    os.makedirs(work, exist_ok=True)

    print(f"\nRunning {len(unique):,} MEME jobs into {work}/ ...")
    motifs = {}
    for i, (k, (s, pool)) in enumerate(sorted(unique.items()), 1):
        motifs[k] = run_meme(s, pool, seqs, work)
        if i % 10 == 0 or i == len(unique):
            ok = sum(1 for v in motifs.values() if v)
            print(f"  {i}/{len(unique)}  ({ok} motifs found)")

    # --- per-fold motif assignment table ---
    rows = []
    for fold_id, g, keys in per_fold:
        for k in keys:
            s, pool = unique[k]
            m = motifs.get(k)
            rows.append({"fold": fold_id, "held_out_group": g, "serotype": s,
                         "pool_size": len(pool), "pool": "|".join(sorted(pool)),
                         "motif": m["consensus"] if m else None,
                         "meme_rank": m["rank"] if m else None,
                         "evalue": m["evalue"] if m else None,
                         "sites": m.get("sites") if m else None,
                         "coverage": round(m.get("coverage", 0), 3) if m else None,
                         "meme_dir": m.get("meme_dir") if m else None})
    out_csv = os.path.join(work, "foldwise_motifs.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    print("\nNext: scan each fold's held-out RBPs with that fold's motif (fimo),")
    print("assign motif-bearing status from THAT scan only, then retrain PHL-M")
    print("and recompute Table 1. Compare against the published 32/37, 16/28, 24/42.")


if __name__ == "__main__":
    main()