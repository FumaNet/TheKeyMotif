# TheKeyMotif — project context

Revision of a bioRxiv preprint (2026.05.21.726843) on protein-level prediction
of Klebsiella phage adsorption, after three reviewer reports. Repo:
github.com/FumaNet/TheKeyMotif

## What this project is doing now

Responding to reviewers. The paper is being reframed: the original novelty
(sub-protein-resolution prediction) was largely taken by Concha-Eloko et al.
2025 (Nat Commun), and the motif analysis has a confirmed circularity problem.
The surviving contributions are (a) a decomposition of what published AUCs on
this benchmark actually measure, and (b) evidence that interaction-level
supervision gives accurate prediction without correct protein attribution.

## Environment

- Windows + PowerShell, conda env `keymotif` (python 3.11, xgboost 2.1.4 from
  PyPI — NOT conda-forge, which silently gives a cpu_only build)
- GPU: GTX 1650, 4 GB. CPU fallback via `$env:KM_DEVICE="cpu"`
- MEME/FIMO only exist in WSL2 (`conda activate motif`). No Windows build.
  Run motif scripts from WSL at /mnt/c/Users/IamAh/Desktop/TheKeyMotif
- Data from Zenodo 11061100 in `Data/`. `Results_published/` is the untouched
  backup of the original pickles — NEVER overwrite it.

## Conventions that matter

- Every long script checkpoints per threshold to `Results/<n>_checkpoint.pkl`
  and only writes the final 8-threshold pickle when ALL thresholds are present.
  A partial file would misalign every threshold against the published values.
- `$env:KM_THRESHOLDS="100,90"` restricts which thresholds run. It PERSISTS in
  the PowerShell session — clear it with `Remove-Item Env:\KM_THRESHOLDS`.
- `keymotif_data.py` replaces the original wide-frame data prep (which needed
  ~10 GB and OOM'd). Verified bit-identical by `test_equivalence.py` — run that
  after touching it.
- `run_overnight.py` / `run_phase1.py` orchestrate multi-hour runs with logging,
  resume, and result collation.

## Established numbers

Reproduction (script 3 exact, script 0 within ±0.010, mixed signs):
  PHL-AVG    [0.817, 0.745, 0.700, ?, 0.661, 0.649, 0.678, 0.699]
  PHL-RBP+S  [0.817, 0.747, 0.690, 0.644, 0.636, 0.615, 0.659, 0.672] (exact)

The 2x2 (all new; none of the 8 original scripts drops the host side):
                    bacterial split    phage split
  RBP + serotype    0.817 / PR 0.403   0.749 / PR 0.223
  RBP only          0.701 / PR 0.144   0.673 / PR 0.095
  serotype only     0.589 / PR 0.041   —
Chance: ROC 0.5, PR 0.033.

Phage-split identity sweep (deduplicated):
  100% id -> 0.764 / 0.262 (99 groups)   95% -> 0.756 / 0.246 (89)
   90% id -> 0.731 / 0.222 (80)          80% -> 0.691 / 0.167 (69)
Graceful decay, no collapse.

Consistency controls (PHL-Random at 0.545 was a WEAK control):
  PHL-M    0.658 0.652 0.656 0.651 0.647 0.636 0.638 0.635  range 0.023
  central  0.735 0.658 0.617 0.581 0.577 0.567 0.592 0.619  range 0.168
  longest  0.664 0.600 0.567 0.544 0.536 0.526 0.549 0.558  range 0.138
  first    0.767 0.694 0.635 0.592 0.581 0.567 0.600 0.620  range 0.200
At 100%, an arbitrary rule ("first") BEATS the motif rule. PHL-M's only
distinctive property is flatness — plausibly the signature of leakage.
Open prediction: after fold-internal motif discovery, PHL-M should start
decaying like the controls. If it does, the flatness was leakage.

## Bugs and discrepancies found (all need fixing in the manuscript)

- `drop_duplicates()` in scripts 2/3/5 is a NO-OP (host protein_index survives),
  so they train on the full 487k table and hosts with large K-loci are
  unintentionally upweighted. Do not "fix" it in reproduction runs.
- Fig 1's PHL-M bar reads ~0.795; stored results say 0.658 (file 5) / 0.706 (file 4).
- `validation_predictions_motif_focus_80.csv` is byte-identical to the 100% file.
- `.sample(1)` unseeded in scripts 5, 6a, 6b.
- Methods say motif selection kept "the most statistically significant motif";
  the actual rule was the lowest-numbered motif covering every input sequence.
- Methods say relabelling used 8 serotypes; the code used 28, and 20 of those
  have <6 phages, contributing 75 of 134 positives.
- Methods say 8 serotypes have >=6 phages; the data also shows K22 (9) and
  K60 (7). UNRESOLVED — check whether pool construction differs.
- FIMO exclusivity check was run in the thesis and FAILED (no serotype-exclusive
  motif). Absent from the preprint. K26==K47 motifs identical; K26 in K32 in K47;
  K54 in K58.
- Fig 1/Fig 2 have no generating code in the repo.

## Current task: Phase 2

`fold_internal_motifs.py` rebuilds serotype pools inside each fold (fixing the
leak: 131/131 phages contributing motif-bearing RBPs were in their own MEME
input). `5c_foldwise_phlm.py` then FIMO-scans held-out RBPs and retrains.

BLOCKER: MEME parameters don't match the original invocation (not recorded
anywhere). With `-mod zoops -nmotifs 15 -minw 21 -maxw 99`, only 5/10 FULL
pools yield a motif, though the paper reports one for every eligible serotype.
Full and reduced pools fail at similar rates, so this is a parameter mismatch,
NOT leakage. K64's published motif IS recovered (rank 12, 9/10 sites,
core EYVGTEHRAI.YMDGFGR matches published YVEEYVGTEHRAIIYMDGFGREDAWSFR).

Next: sweep `--nmotifs 40` then `--maxw 50` until FULL pools reach coverage
1.0, then rerun fold-internal with that setting. Only then is any drop
attributable to leakage.

## Still to do

- Phase 3: TropiGAT attention dispersion (`tropigat_dispersion.py`, needs
  torch_geometric + Zenodo 14065540 weights). Tests whether attention
  discriminates among a prophage's depolymerases. Necessary-not-sufficient:
  spread shows the model CAN discriminate, not that it does so CORRECTLY.
