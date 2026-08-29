"""
keymotif_data.py — memory-safe replacement for the data-prep block in
TheKeyMotif's model scripts.

WHY
---
Every script begins by merging host-protein embeddings and phage RBP embeddings
into one wide DataFrame, then caching it as
`data/combined_embeddings_per_protein.csv`. On the real dataset that frame is

    487,400 rows x 2,560 float columns

and — because `.astype(np.float32)` is applied *after* the merge — pandas builds
it in float64. Peak ~10 GB during `merge`, and the CSV would be ~16 GB of text.

That wide frame is entirely redundant. Each row is just a pointer to one of a
few thousand host-protein embeddings and one of a few hundred RBP embeddings:

    unique host proteins  ~ 200 accessions x ~16 proteins  ~   3,000 rows
    unique phage RBPs     ~ 105 phages     x ~3   RBPs     ~     300 rows

So we store those two small float32 matrices once, plus an index table with one
row per pair holding two integer columns. Total well under 100 MB.

The index table is built by running the *same* melt + merges as the original
code, but on key columns only — so row order and row set are identical.

USAGE
-----
    import keymotif_data as kd

    pairs, host_emb, virus_emb = kd.load()          # builds cache on first call

    # per-fold, per-script:
    X = kd.make_X(fold_pairs, host_emb, virus_emb, mode="pair")

EQUIVALENCE
-----------
`test_equivalence.py` runs the original wide-frame logic and this one over the
same synthetic dataset and asserts the resulting AUCs match exactly.
"""

import json
import os

import numpy as np
import pandas as pd

DATA = "data"
CACHE_DIR = os.path.join(DATA, "cache")
HOST_NPY = os.path.join(CACHE_DIR, "host_emb.npy")
VIRUS_NPY = os.path.join(CACHE_DIR, "virus_emb.npy")
PAIRS_PARQUET = os.path.join(CACHE_DIR, "pairs.parquet")
PAIRS_CSV = os.path.join(CACHE_DIR, "pairs.csv")
META_JSON = os.path.join(CACHE_DIR, "meta.json")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _embedding_cols(df, keys):
    """Embedding columns are everything that isn't a key column."""
    return [c for c in df.columns if c not in keys]


def build_cache(data_dir=DATA, verbose=True):
    """
    Build the compact cache from the Zenodo inputs.

    Peak memory is dominated by reading the two embedding CSVs, which are small
    (a few thousand rows each). Nothing 487k-by-2560 is ever materialised.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    def say(*a):
        if verbose:
            print(*a)

    say("Reading embeddings (small — these are the per-protein tables)...")
    loci = pd.read_csv(os.path.join(data_dir, "esm2_embeddings_loci_per_protein.csv"),
                       dtype={"accession": str})
    rbp = pd.read_csv(os.path.join(data_dir, "esm2_embeddings_rbp.csv"))
    inter = pd.read_csv(os.path.join(data_dir, "phage_host_interactions.csv"))

    host_keys = [c for c in ["accession", "protein_index", "protein_ID"] if c in loci.columns]
    virus_keys = [c for c in ["phage_ID", "protein_ID", "protein_index"] if c in rbp.columns]

    host_cols = _embedding_cols(loci, host_keys)
    virus_cols = _embedding_cols(rbp, virus_keys)
    say(f"  host proteins : {len(loci):>7,} x {len(host_cols)} dims")
    say(f"  phage RBPs    : {len(rbp):>7,} x {len(virus_cols)} dims")

    # --- the two compact float32 matrices -------------------------------
    host_emb = np.ascontiguousarray(loci[host_cols].to_numpy(dtype=np.float32))
    virus_emb = np.ascontiguousarray(rbp[virus_cols].to_numpy(dtype=np.float32))

    # --- key-only frames, carrying their row position -------------------
    loci_keys = loci[host_keys].copy()
    loci_keys["host_idx"] = np.arange(len(loci), dtype=np.int32)

    rbp_keys = rbp[virus_keys].copy()
    rbp_keys["virus_idx"] = np.arange(len(rbp), dtype=np.int32)

    # --- same melt + merges as the original, on keys only ---------------
    say("Building pair index (same merges as the original, keys only)...")
    melted = inter.melt(id_vars=["Unnamed: 0"], var_name="phage_ID",
                        value_name="label").rename(columns={"Unnamed: 0": "accession"})
    melted = melted.dropna(subset=["label"])
    melted["accession"] = melted["accession"].astype(str)

    pairs = melted.merge(loci_keys, on="accession", how="inner")
    pairs = pairs.merge(rbp_keys, on="phage_ID", how="inner",
                        suffixes=("_host", "_virus"))

    # Normalise the protein_ID column name (it may collide across the two
    # tables and get suffixed; downstream scripts expect 'protein_ID').
    if "protein_ID" not in pairs.columns:
        for cand in ("protein_ID_virus", "protein_ID_host"):
            if cand in pairs.columns:
                pairs["protein_ID"] = pairs[cand]
                break
    if "protein_ID" not in pairs.columns:
        pairs["protein_ID"] = pairs["virus_idx"].astype(str)
    if "protein_index" not in pairs.columns:
        for cand in ("protein_index_host", "protein_index_virus"):
            if cand in pairs.columns:
                pairs["protein_index"] = pairs[cand]
                break

    keep = ["accession", "phage_ID", "protein_index", "protein_ID",
            "host_idx", "virus_idx", "label"]
    pairs = pairs[[c for c in keep if c in pairs.columns]].reset_index(drop=True)
    pairs["label"] = pairs["label"].astype(np.int8)

    say(f"  pairs         : {len(pairs):>7,} rows")
    say(f"  wide frame avoided: "
        f"{len(pairs) * (host_emb.shape[1] + virus_emb.shape[1]) * 8 / 1e9:.1f} GB "
        f"(float64) -> {pairs.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    np.save(HOST_NPY, host_emb)
    np.save(VIRUS_NPY, virus_emb)
    try:
        pairs.to_parquet(PAIRS_PARQUET, index=False)
        pairs_path = PAIRS_PARQUET
    except Exception:
        pairs.to_csv(PAIRS_CSV, index=False)
        pairs_path = PAIRS_CSV

    with open(META_JSON, "w") as fh:
        json.dump({"n_pairs": int(len(pairs)),
                   "host_dim": int(host_emb.shape[1]),
                   "virus_dim": int(virus_emb.shape[1]),
                   "n_host_proteins": int(host_emb.shape[0]),
                   "n_rbps": int(virus_emb.shape[0]),
                   "pairs_file": os.path.basename(pairs_path)}, fh, indent=2)

    say(f"Cache written to {CACHE_DIR}/")
    return pairs, host_emb, virus_emb


def load(data_dir=DATA, verbose=True):
    """Load the compact cache, building it first if absent."""
    have = (os.path.exists(HOST_NPY) and os.path.exists(VIRUS_NPY)
            and (os.path.exists(PAIRS_PARQUET) or os.path.exists(PAIRS_CSV)))
    if not have:
        return build_cache(data_dir, verbose=verbose)

    host_emb = np.load(HOST_NPY)
    virus_emb = np.load(VIRUS_NPY)
    if os.path.exists(PAIRS_PARQUET):
        pairs = pd.read_parquet(PAIRS_PARQUET)
    else:
        pairs = pd.read_csv(PAIRS_CSV, dtype={"accession": str})
    if verbose:
        print(f"Loaded cache: {len(pairs):,} pairs, "
              f"{host_emb.shape[0]:,} host proteins, {virus_emb.shape[0]:,} RBPs")
    return pairs, host_emb, virus_emb


# ---------------------------------------------------------------------------
# feature assembly
# ---------------------------------------------------------------------------

def make_X(pairs, host_emb, virus_emb, mode="pair", sero=None):
    """
    Assemble the feature matrix for a subset of `pairs`.

    mode="pair"   host protein embedding || RBP embedding      (script 1)
    mode="virus"  RBP embedding only                           (scripts 2,3,4,5,6)
    mode="host"   host protein embedding only

    `sero`, if given, is a (n_rows, n_serotypes) float32 array appended on the
    right -- the one-hot capsular serotype block.

    Writes blocks into ONE preallocated array rather than np.hstack-ing them.
    hstack holds the inputs and the output simultaneously, which on the real
    script-3 fold (487,400 x 1360) means a 5.3 GB peak instead of 2.65 GB.
    Numerically identical -- same values, same dtype, same layout.
    """
    n = len(pairs)
    widths, sources = [], []
    if mode in ("pair", "host"):
        widths.append(host_emb.shape[1]); sources.append(("host", None))
    if mode in ("pair", "virus"):
        widths.append(virus_emb.shape[1]); sources.append(("virus", None))
    if sero is not None:
        sero = np.asarray(sero, dtype=np.float32)
        widths.append(sero.shape[1]); sources.append(("sero", sero))

    X = np.empty((n, sum(widths)), dtype=np.float32, order="C")
    off = 0
    for (kind, arr), w in zip(sources, widths):
        if kind == "host":
            np.take(host_emb, pairs["host_idx"].to_numpy(), axis=0,
                    out=X[:, off:off + w])
        elif kind == "virus":
            np.take(virus_emb, pairs["virus_idx"].to_numpy(), axis=0,
                    out=X[:, off:off + w])
        else:
            X[:, off:off + w] = arr
        off += w
    return X


_COLLAPSE_CACHE = {}
_COLLAPSE_FULL = {"key": None, "frame": None}


def precompute_collapsed(pairs, host_emb, virus_emb, chunk_key="accession",
                         verbose=True):
    """
    Compute script 0's averaged representation ONCE for the whole dataset.

    Each (accession, phage_ID) group lies entirely inside a single LOGO fold,
    because accession maps to exactly one group. Group means are therefore
    fold-independent: computing them globally and slicing per fold gives
    bit-identical values to recomputing them inside every fold, at 1/370th of
    the work (185 folds x 2 calls).

    Chunking by accession bounds peak memory to one accession's rows x all
    RBPs x 2560 float32 (~50 MB), never the full frame.
    """
    hcols = [f"host_{i}" for i in range(host_emb.shape[1])]
    vcols = [f"virus_{i}" for i in range(virus_emb.shape[1])]
    out = []
    groups = list(pairs.groupby(chunk_key, sort=False))
    for n, (_, sub) in enumerate(groups, 1):
        blk = pd.concat(
            [sub[["accession", "phage_ID", "protein_ID"]].reset_index(drop=True),
             pd.DataFrame(host_emb[sub["host_idx"].to_numpy()], columns=hcols),
             pd.DataFrame(virus_emb[sub["virus_idx"].to_numpy()], columns=vcols),
             sub[["label"]].reset_index(drop=True)],
            axis=1)
        out.append(blk.groupby(["accession", "phage_ID"]).agg(
            {**{c: "mean" for c in hcols + vcols},
             "label": "first", "protein_ID": "first"}).reset_index())
        del blk
        if verbose and (n % 25 == 0 or n == len(groups)):
            print(f"  collapsing {n}/{len(groups)} accessions", end="\r")
    if verbose:
        print()
    collapsed = pd.concat(out, ignore_index=True)
    collapsed = collapsed.sort_values(["accession", "phage_ID"]).reset_index(drop=True)

    _COLLAPSE_FULL["key"] = (id(host_emb), id(virus_emb))
    _COLLAPSE_FULL["frame"] = collapsed
    return collapsed


def collapse_averaged_exact(pairs, host_emb, virus_emb, chunk_key="accession"):
    """
    Bit-identical replacement for script 0's in-fold averaging.

    Uses the globally precomputed frame when available (see
    precompute_collapsed); otherwise computes it from `pairs` on first call and
    reuses it thereafter. The factorised shortcut is NOT used: it differs from
    pandas by ~1 float32 ULP (1.19e-7), which is enough to flip a tree split.
    """
    key = (id(host_emb), id(virus_emb))
    if _COLLAPSE_FULL["key"] != key or _COLLAPSE_FULL["frame"] is None:
        raise RuntimeError(
            "collapse_averaged_exact: no precomputed frame.\n"
            "Call kd.precompute_collapsed(fold_pairs, host_emb, virus_emb) ONCE\n"
            "with the FULL fold frame before the LOGO loop. Auto-precomputing\n"
            "from the first (train) subset would silently omit the held-out\n"
            "rows, and the test call would then return nothing.")

    collapsed = _COLLAPSE_FULL["frame"]

    wanted = set(zip(pairs["accession"], pairs["phage_ID"]))
    mask = [k in wanted for k in zip(collapsed["accession"], collapsed["phage_ID"])]
    sel = collapsed[mask].reset_index(drop=True)

    if len(sel) != len(wanted):
        raise RuntimeError(
            f"collapse_averaged_exact: precomputed frame covers {len(sel)} of "
            f"{len(wanted)} requested (accession, phage_ID) pairs. The frame was "
            "built from a subset. Re-run precompute_collapsed on the full frame.")

    fc = [c for c in sel.columns if c.startswith(("host_", "virus_"))]
    meta = sel[["accession", "phage_ID", "label", "protein_ID"]]
    X = sel[fc].values
    return meta, X


def attach_serotype(sub, sero_table, sero_cols):
    """
    Left-merge the one-hot serotype block onto a fold subset and return it as a
    float32 array, matching scripts 2/3/5.

    NOTE on drop_duplicates: those scripts call `df.drop_duplicates()` with no
    subset after dropping the host_* embedding columns. Because the host
    `protein_index` column is still present and differs across the rows that
    share an RBP, that call removes ZERO rows -- it is a no-op, despite the
    in-code comment describing it as redundant with the averaging. This
    function therefore does not deduplicate either, so behaviour matches.
    """
    merged = sub.merge(sero_table, how="left", left_on="accession",
                       right_on="Assembly").drop(columns=["Assembly"])
    block = merged[sero_cols].fillna(0)
    return merged, block.to_numpy(dtype=np.float32)


def attach_groups(pairs, grouping_path, column="group_loci"):
    """
    Map accessions to their LOGO group and fail loudly if the mapping is bad.

    The raw `pairs['accession'].map(grouping_dict)` in the original scripts
    silently produces NaN for any accession missing from the pickle. NaN groups
    then flow into LeaveOneGroupOut and either crash somewhere confusing or,
    worse, quietly lump every unmapped accession into one pseudo-group. A dtype
    mismatch (int keys in the pickle vs str accessions here, or vice versa) can
    make EVERY row NaN, which is silent data loss, not an error.

    Returns a copy of `pairs` with the group column added.
    """
    import pickle

    with open(grouping_path, "rb") as fh:
        groups = pickle.load(fh)

    out = pairs.copy()
    mapped = out["accession"].map(groups)

    if mapped.isna().all():
        sample_key = next(iter(groups))
        raise ValueError(
            f"No accession matched {grouping_path}.\n"
            f"  pairs['accession'] example: {out['accession'].iloc[0]!r} "
            f"({type(out['accession'].iloc[0]).__name__})\n"
            f"  grouping key example:       {sample_key!r} "
            f"({type(sample_key).__name__})\n"
            "Usually a dtype mismatch — try casting the grouping keys to str."
        )

    n_missing = int(mapped.isna().sum())
    if n_missing:
        missing = sorted(out.loc[mapped.isna(), "accession"].unique())[:5]
        print(f"  warning: {n_missing:,} rows across "
              f"{len(set(out.loc[mapped.isna(), 'accession']))} accession(s) "
              f"not in {os.path.basename(grouping_path)}; dropping them. "
              f"e.g. {missing}")
        out = out.loc[mapped.notna()].copy()
        mapped = mapped.loc[mapped.notna()]

    out[column] = mapped.values
    return out


# ---------------------------------------------------------------------------
# out-of-core training for script 1
# ---------------------------------------------------------------------------

class PairBatchIter:
    """
    xgboost.DataIter that streams (host || virus) batches straight from the
    compact arrays, so script 1 never materialises its ~5 GB float32 matrix.

        import xgboost as xgb
        it = PairBatchIter(fold_pairs, host_emb, virus_emb, y)
        dtrain = xgb.QuantileDMatrix(it, max_bin=256)
        booster = xgb.train(params, dtrain, num_boost_round=250)

    Requires xgboost >= 2.0.
    """

    def __init__(self, pairs, host_emb, virus_emb, y, batch_size=50_000):
        import xgboost as xgb
        self._xgb = xgb
        self.pairs = pairs.reset_index(drop=True)
        self.host_emb = host_emb
        self.virus_emb = virus_emb
        self.y = np.asarray(y)
        self.batch_size = batch_size
        self._it = 0
        self._n = len(self.pairs)
        self._cache = None
        # DataIter needs a cache prefix for its temp pages
        self._proxy = xgb.DataIter(cache_prefix=os.path.join(CACHE_DIR, "dmat"))

    # xgboost.DataIter protocol -------------------------------------------
    def reset(self):
        self._it = 0

    def next(self, input_data):
        if self._it >= self._n:
            return 0
        lo, hi = self._it, min(self._it + self.batch_size, self._n)
        sl = self.pairs.iloc[lo:hi]
        X = make_X(sl, self.host_emb, self.virus_emb, mode="pair")
        input_data(data=X, label=self.y[lo:hi])
        self._it = hi
        return 1
