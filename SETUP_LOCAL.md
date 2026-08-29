# TheKeyMotif — local setup & reproduction (Windows, GTX 1650 4 GB)

Written against the current `main` of `github.com/FumaNet/TheKeyMotif`.

**Two independent blockers, in the order you hit them:**

1. **Host RAM during data prep** — what you just hit. Not GPU-related at all.
2. **GPU-only XGBoost flags** — hits later, once prep succeeds.

---

## 0. TL;DR

```powershell
Copy-Item -Recurse Results Results_published   # back up published numbers FIRST
python verify_reproduction.py --preflight
python patch_device.py --device cuda
python test_equivalence.py                     # proves the data layer is exact
python 0_original_replica_compact.py           # drop-in, no hand-editing
python 3_max_max_sero_compact.py
```

---

## 1. The RAM failure, explained

```
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 4.65 GiB
for an array with shape (1281, 487400) and data type float64
```

487,400 rows is the real size of the merged pair table (10,006 interactions ×
~49 host-protein × RBP combinations each). At 2,560 embedding columns:

| representation | size |
|---|---|
| merge intermediate, **float64** | **~10.0 GB** |
| same as float32 | ~5.0 GB |
| `combined_embeddings_per_protein.csv` as text | **~16 GB on disk** |

It builds in float64 because `.astype(np.float32)` runs **after** the merge, not
before. This was written on a big-memory machine; it is not a 4 GB-VRAM problem
and no GPU setting fixes it.

**A dtype tweak is not enough.** float32 still needs ~5 GB resident, plus a
16 GB CSV you'd re-read every script. The wide frame has to go.

---

## 2. The fix: `keymotif_data.py`

That frame is pure redundancy. Each row only points at one of a few thousand
host-protein embeddings and one of a few hundred RBP embeddings:

```
unique host proteins  ~200 accessions × ~16 proteins  ≈ 3,000 rows
unique phage RBPs     ~105 phages     × ~3 RBPs       ≈   300 rows
```

`keymotif_data.py` stores those two small float32 matrices once, plus an index
table with two integer columns per pair. **~3 MB instead of ~10 GB.** Feature
matrices are assembled per fold by fancy-indexing.

Row order and row set are preserved exactly: it runs the *same* melt + merges,
just on key columns only.

```python
import keymotif_data as kd
pairs, host_emb, virus_emb = kd.load()      # builds the cache on first call
```

### This is verified, not asserted

`test_equivalence.py` generates a synthetic dataset with the real schema, runs
the original wide-frame code and the compact code side by side, and compares.

```
script 0 (PHL-AVG)     orig 0.5173742715   compact 0.5173742715   max|dev| 0.0e+00
script 3 (PHL-RBP+S)   orig 0.5306718094   compact 0.5306718094   max|dev| 0.0e+00
ALL EQUIVALENT
```

Bit-identical, at both 24 and 256 embedding dimensions. Run it yourself before
trusting any of this.

> Getting there took two real corrections, both of which are baked in:
> the naive factorised mean differs from pandas by one float32 ULP (1.19e-7),
> which XGBoost amplified into flipped tree splits — hence
> `collapse_averaged_exact()`. And see §4.

---

## 3. Rewiring the scripts

**Use the drop-in files — don't hand-edit.** `0_original_replica_compact.py`
and `3_max_max_sero_compact.py` are complete replacements. Same models, same
LOGO protocol, same output pickles. Drop them in the repo root next to
`keymotif_data.py` and run:

```powershell
python 0_original_replica_compact.py
python 3_max_max_sero_compact.py
```

Both are tested end-to-end (they run to completion and write 8-threshold
pickles in the original format), and their logic is the same code proven
bit-identical by `test_equivalence.py`.

Set `KM_DEVICE=cpu` to fall back off the GPU for one run:

```powershell
$env:KM_DEVICE="cpu"; python 0_original_replica_compact.py; $env:KM_DEVICE="cuda"
```

### Why not hand-edit

The loader swap alone leaves the rest of the script referencing `final_df`,
which no longer exists — you get a `NameError` at the first
`final_df['group_loci'] = ...`. The compact layer returns `pairs` (a row index)
plus two embedding arrays, so every downstream use of `final_df` has to change
too. That's why these are whole files.

### For scripts 1, 2, 4, 5, 6

Same three moves, patterned on the two above:

1. Replace the prep block with `pairs, host_emb, virus_emb = kd.load()`.
2. Replace `final_df['group_loci'] = final_df['accession'].map(...)` with
   `fold_pairs = kd.attach_groups(pairs, GROUPING_FILES[i])`.
3. Build features per fold with `kd.make_X(sub, host_emb, virus_emb, mode=...)`
   — `mode="pair"` for script 1 (host ‖ virus), `mode="virus"` plus
   `kd.attach_serotype(...)` for 2/4/5/6.

If script 1's ~5 GB float32 matrix is still too big, `kd.PairBatchIter` feeds
`xgboost.QuantileDMatrix` in batches without materialising it.

### `attach_groups` catches a silent failure

The original `pairs['accession'].map(groups_dict)` yields NaN for any accession
missing from the pickle, and NaN groups flow into `LeaveOneGroupOut` either to
crash somewhere confusing or — worse — to lump every unmapped accession into
one pseudo-group. A dtype mismatch makes *every* row NaN. `attach_groups`
raises instead:

```
ValueError: No accession matched grouping/bad.pkl.
  pairs['accession'] example: 'ACC000' (str)
  grouping key example:       0 (int)
  Usually a dtype mismatch — try casting the grouping keys to str.
```

Partial misses warn and drop the affected rows rather than failing silently.

`collapse_averaged_exact` memoises: the group means are fold-independent (each
`(accession, phage_ID)` group lies entirely inside one LOGO fold), so they are
computed once globally and subset per fold — a large speedup over recomputing
in all 185 folds.

---

## 4. A bug worth knowing about

Scripts 2, 3 and 5 call `df.drop_duplicates()` after dropping the `host_*`
columns, with an in-code comment describing it as *"rendundant with the
remove_duplicates above"*. It is **a no-op — it removes zero rows.** The host
`protein_index` column is never dropped, so rows sharing an RBP still differ.

Confirmed on synthetic data: `drop_duplicates() removes 0 of 2,728 rows`.

Consequences:

- **Compute:** 2/3/5 train on the full ~487k-row table, not a reduced one. My
  earlier "~1/15 of full" estimate was wrong; preflight is corrected.
- **Statistics, and this one matters:** each (RBP, serotype) example is
  replicated once per host locus protein — so hosts with **larger K-loci get
  proportionally more weight** in training. Not fatal (it doesn't touch
  evaluation, where max-aggregation collapses duplicates), but it is an
  unintended non-uniform sample weighting, and Reviewer #2 is already asking
  hard questions about the K-locus grouping. Better to find it yourself.

Deduplicating properly would change the published numbers, so don't do it as
part of the reproduction. Do it as a separate, reported experiment.

---

## 5. Environment

```powershell
conda create -n keymotif python=3.12 -y
conda activate keymotif
pip install pandas==2.2.3 numpy==2.2.2 scikit-learn==1.6.1 tqdm matplotlib pyarrow psutil
pip install xgboost==2.1.4        # PyPI wheel — bundles CUDA
```

**Install XGBoost from PyPI, not conda-forge.** conda-forge often resolves to a
`cpu_only` build that accepts `device="cuda"` and silently trains on CPU.
Preflight catches exactly this — XGBoost 2.x does *not* raise, it warns and
falls back, so a naive check reports success.

Add `torch` + `fair-esm` only for §6.

Zenodo (<https://zenodo.org/records/11061100>) into `Data/`:
`esm2_embeddings_loci.csv`, `esm2_embeddings_rbp.csv`,
`phage_host_interactions.csv`, `Locibase.json`.

---

## 6. Embeddings: one real bug

`single_protein_loci.py` passes `return_contacts=True` but never uses the
contacts. That materialises the full attention stack — 33 layers × 20 heads ×
L × L floats. For a 1,000-residue protein that's ~2.6 GB of activations on top
of 2.6 GB of fp32 weights: guaranteed OOM on a 1650, and wasted compute
everywhere else.

```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ...
            batch_tokens = batch_tokens.to(device)
            try:
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33],
                                    return_contacts=False)          # <-- the fix
                rep = results["representations"][33]
                emb = rep[0, 1:len(sequence) + 1].mean(0).cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()                            # long protein
                with torch.no_grad():
                    results = model.cpu()(batch_tokens.cpu(), repr_layers=[33],
                                          return_contacts=False)
                rep = results["representations"][33]
                emb = rep[0, 1:len(sequence) + 1].mean(0).numpy()
                model = model.to(device)
```

**Keep fp32.** `model.half()` would fit more comfortably but perturbs the
embeddings feeding every AUC you're trying to match.

Expect 20–40 min on the 1650 vs 1–3 h on CPU. Run once.

---

## 7. Device flags: `patch_device.py`

The eight scripts are **not consistent**, so a blind `sed` (which you don't
have on Windows anyway) would leave script 1 in a different state:

| scripts | flags |
|---|---|
| 0, 2, 3, 4, 5, 6a, 6b | `gpu_hist`, `predictor=`, `device="cuda"` |
| **1** | `gpu_hist`, `predictor=`, `max_bin=256` — **no `device=`** |

```powershell
python patch_device.py --device cuda     # all eight
python patch_device.py --device cpu      # fallback
python patch_device.py --restore         # undo from .bak
```

Normalises everything to `tree_method="hist", device="cuda"`, drops the dead
`predictor=` / `use_label_encoder=` args, preserves each file's line endings
(the repo mixes CRLF and LF). Verified against the live repo: all eight patch
cleanly and compile.

If preflight says script 1 is TIGHT on VRAM, **run it on CPU rather than
lowering `max_bin`** — script 1's `max_bin=256` is part of the published model,
so changing it changes the 0.796; CPU vs GPU only moves the third decimal.

---

## 8. Verifying

```powershell
python verify_reproduction.py --device cuda --report rerun_report.json
```

| file | 100% | 99.5% | 99% | 95% | 90% | 85% | 80% | 75% |
|---|---|---|---|---|---|---|---|---|
| 0 PHL-AVG   | 0.807 | 0.740 | 0.695 | 0.672 | 0.668 | 0.655 | 0.674 | 0.690 |
| 1 PHL-RBP   | 0.796 | 0.715 | 0.663 | 0.634 | 0.631 | 0.614 | 0.641 | 0.655 |
| 2 PHL-S     | 0.816 | 0.727 | 0.663 | 0.626 | 0.621 | 0.600 | 0.637 | 0.655 |
| 3 PHL-RBP+S | 0.817 | 0.747 | 0.690 | 0.644 | 0.636 | 0.615 | 0.659 | 0.672 |
| 4 PHL-M+    | 0.706 | 0.690 | 0.687 | 0.671 | 0.673 | 0.658 | 0.655 | 0.662 |
| 5 PHL-M     | 0.658 | 0.652 | 0.656 | 0.651 | 0.647 | 0.636 | 0.638 | 0.635 |
| 6a Random   | 0.545 | — | — | — | — | — | — | — |

Tolerances: ±0.003 on GPU, ±0.005 on CPU. A 1650 won't be bit-identical to
whatever card produced the originals (GPU histogram gradient summation order is
architecture-dependent), so third-decimal movement is expected. A consistent
shift at *every* threshold points at data prep, not the device.

**Fix before reporting anything:** scripts 5, 6a and 6b call `.sample(1)` with
no `random_state`, so they aren't reproducible run to run. Script 4 already
seeds its one call with 42. Add `random_state=42` to the others.

---

## 9. Still open

1. **Fig 1's PHL-M bar at 100%** reads ≈0.795, but `5_*.pkl` gives 0.658 and
   `4_*.pkl` gives 0.706. Confirm which series that bar came from.
2. **Fig 1 and Fig 2 have no generating code.** `Graphs/auc_graphs.ipynb` only
   produces the S2 Appendix panels. Adding the bar-chart script closes
   Reviewer #1's repo-hygiene point cheaply — as does a README line about the
   two-repository confusion.
