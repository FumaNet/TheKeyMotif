# TheKeyMotif

Code for a revision of "Protein-level prediction of *Klebsiella* phage
adsorption identifies conserved receptor-binding motifs" (bioRxiv
2026.05.21.726843), after three reviewer reports.

The paper has been reframed. The original novelty claim (sub-protein-resolution
prediction) was largely anticipated by Concha-Eloko et al. 2025 (Nat Commun),
and the motif-conservation analysis this repo was originally built around has
a confirmed circularity problem (see "The withdrawn motif analysis" below).
The surviving contributions are:

1. A decomposition of what the published AUCs on this benchmark actually
   measure — how much of PHL-RBP+S's accuracy is carried by serotype alone
   versus RBP identity, and whether it survives phage-level (not just
   bacterial-level) holdout.
2. Evidence that interaction-level supervision gives accurate prediction
   without correct protein-level attribution — the model can predict phage-host
   adsorption well while still not identifying *which* receptor-binding
   protein is doing the work, which is what the motif analysis needed and
   didn't have.

This extends the PhageHostLearn tool by Boeckaerts et al. (2024, original code
at https://github.com/dimiboeckaerts/PhageHostLearn) to individual protein
embeddings, and separately shows a one-hot serotype encoding is viable when
the host capsule type is known.

## Figures and where they come from

| Figure | Script |
|---|---|
| Fig 1, Fig 2 | **No generating code in this repo.** These predate the current pipeline; the numbers in the manuscript were not reproduced from a script here. |
| Fig 3 | `analysis/01_model_variants/` (scripts `0`–`3`, PHL-AVG/PHL-RBP/PHL-S/PHL-RBP+S) |
| Fig 4 | `analysis/02_ablation/` (`2b_serotype_only.py`, `2c_rbp_only.py`, `2d_phage_holdout.py`) |
| Fig 5 | `manuscript/fig5_rebuild.py` → `Fig5_motif_conservation.{png,pdf}`, `Fig5_motif_identity.csv` |
| Fig 6 | `analysis/04_external/` (the KlebPhaCol head-to-head benchmark; see `results/klebphacol/stage6_head_to_head.md`) |
| S4 Appendix | `manuscript/s4_appendix_build.py` → `S4_appendix_table.{csv,tex}` |

`analysis/03_motif_caution/` holds the motif-caution scripts (`4`, `5`, `5b`,
`5c`, `6a`, `6b`, `fold_internal_motifs.py`) — these back the cautionary
result described below, not a manuscript figure.

## Layout

```
src/                    shared data-loading module and small utilities
analysis/
  01_model_variants/     PHL-AVG / PHL-RBP / PHL-S / PHL-RBP+S (+ memory-safe _compact variants)
  02_ablation/            serotype-only / RBP-only / phage-holdout baselines
  03_motif_caution/       the motif-conservation analysis and its retained cautionary result
  04_external/            KlebPhaCol external benchmark and TropiSEQ comparator
manuscript/              scripts and tables that produce a specific manuscript figure/table
results/                 local run outputs (AUC pickles, klebphacol/ CSVs)
results_published/       canonical published results — never overwritten
data/                    inputs (see "Data not included" below)
motifs/
  occurrences/            MEME motif occurrences per serotype (was Motifs_KO/)
  foldwise/               fold-internal motif discovery inputs (was Motifs_foldwise/)
figures/                 figure-generation notebook (was Graphs/)
grouping/                host genomic-derived LOGO groupings at each identity threshold
logs/                    run logs referenced by REPRODUCIBILITY.md and SETUP_LOCAL.md
```

Root-level `run_overnight.py`, `run_phase1.py`, `test_equivalence.py`, and
`verify_reproduction.py` are orchestration and verification tools that operate
across the whole pipeline rather than belonging to one analysis stage.

## The withdrawn motif analysis

The original thesis searched for conserved sequence elements among RBPs
infecting phages of the same capsular serotype (`analysis/03_motif_caution/`,
`motifs/occurrences/`, `motifs/foldwise/`). This was withdrawn: the pooled
MEME motif search let RBPs see other phages' sequences during motif discovery
that were also used at prediction time, and a fold-internal rebuild
(`fold_internal_motifs.py`) found essentially every phage contributing a
motif-bearing RBP had also contributed to that motif's own discovery set.
The retained cautionary result is the consistency-control comparison
(`5b_consistency_control.py`, `run_phase1.py --collate`): an arbitrary
selection rule ("first RBP") performs comparably to or better than the motif
rule at some thresholds, meaning PHL-M's headline flatness across thresholds
is not on its own evidence that the motif is doing anything. `motifs/` is
kept because Fig 5 and the fold-internal-motif retained result depend on the
occurrence and input files in it, not because the motif-conservation claim
itself survived review.

## Data not included

Not committed, for size or licensing reasons. Regenerate or fetch from source
before running anything that needs them:

- `data/esm2_embeddings_rbp.csv`, `data/esm2_embeddings_loci.csv`,
  `data/esm2_embeddings_loci_per_protein.csv`, `data/cache/` — regenerable via
  `single_protein_loci.py` / `src/keymotif_data.py` from the Zenodo inputs
  below.
- `data/genomes/` and the two `data/*.zip` archives — Boeckaerts et al.'s
  PhageHostLearn inputs, Zenodo 11061100.
- `data/tropiseq/TropiSEQ_model/` — TropiSEQ's released depolymerase-cluster
  database; redistribution is a licensing question this repo doesn't answer.
  Obtain from the TropiSEQ release.
- `data/klebphacol/*.docx`, `*.xlsb` — KlebPhaCol's supplementary data and
  tables, from the KlebPhaCol NAR paper's supplementary materials.
- `motifs/occurrences/**/*.cif` — 101 AlphaFold structure models backing
  figures no longer in the paper.
- `motifs/foldwise/**/meme.{xml,html,txt}` — MEME scratch output (~195MB);
  archived to Zenodo rather than committed. `input.fasta`, `kept_motif.txt`,
  and `foldwise_motifs.csv` (what the retained result actually needs) are
  tracked.
- `results/*_checkpoint*.pkl` — resumable per-threshold state for long runs,
  not results; regenerated automatically by the `_compact` scripts.

## Environment

See `packagelist.txt` for exact package versions and `SETUP_LOCAL.md` for
machine-specific setup (CUDA/XGBoost gotchas, PowerShell invocation examples,
known device-dependent results). XGBoost must come from PyPI, not
conda-forge, which silently produces a CPU-only build.

## Reproducibility

`REPRODUCIBILITY.md` documents a known, investigated gap between the
published script-0 (PHL-AVG) ROC-AUC and what re-running it produces on a
different GPU — ruled out as nondeterminism and as a data-handling bug in the
memory-safe refactor, still open as an environment/build question. Read it
before treating any single-machine rerun as a red flag.
