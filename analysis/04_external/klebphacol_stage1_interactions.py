#!/usr/bin/env python3
"""
klebphacol_stage1_interactions.py — Stage 1 of the KlebPhaCol head-to-head
benchmark: build the four interaction tables from Table S4 (host-range
titres) of the KlebPhaCol supplementary workbook.

Table S4 layout (0-indexed rows/cols, as returned by pyxlsb, confirmed by
direct inspection before writing this):
  row 1  = "LB titres (PFU/mL)" section header
  row 2  = strain-name header row, cols 1-74 (col 0 blank, col 75 blank
           separator, cols 76-149 repeat the same 74 strains for the EOP
           block -- confirmed identical names/order via direct diff, one
           formatting difference "NCTC 7427" vs "NCTC_7427" at position 41,
           immaterial since only cols 1-74 are used)
  rows 3-54 = LB data (52 phages: Roth37...Roth01); rows 55-58 blank filler
  row 59 = "TSB titres (PFU/mL)" section header
  row 60 = strain-name header row (identical to row 2)
  rows 61-112 = TSB data (52 phages)

Row boundaries are NOT hardcoded as a fixed range: each data block is read
from its header row to the next blank/header row, and any row with no phage
name (col 0 is None) is dropped. This makes the exact row-count irrelevant
and self-correcting if the sheet is ever re-exported with different padding.

Encoding (given, verified against the paper's counts below):
  value > 0  -> productive infection
  value == 0 -> no infection
  value == -1 -> undetermined lysis (opaque lysis, no plaques)

Four outputs, all (phage, strain, label) long-format CSVs:
  interactions_LB_strict.csv       >0 -> 1, 0 -> 0, -1 -> DROPPED
  interactions_LB_permissive.csv   >0 or -1 -> 1, 0 -> 0
  interactions_TSB_strict.csv      same, TSB
  interactions_TSB_permissive.csv  same, TSB

WHY BOTH MAPPINGS (not optional -- see CLAUDE.md / task instructions):
Boeckaerts' training labels are NOT plaque-based -- Beamud et al. 2023
confirmed every spot-assay positive with a planktonic killing assay, so a
phage that can't form plaques but does impair growth is still labelled
positive in training. KlebPhaCol's "undetermined lysis" (-1) is exactly that
case. Strict mapping (dropping -1) makes the test set's positive criterion
STRICTER than the training criterion and would penalise the model on cases
its label definition was never meant to exclude. Permissive is closer to the
training criterion. Report both; the gap measures label-definition
sensitivity, not model quality.

LB is PRIMARY (KlebPhaCol's default assay medium; Beamud et al. also used
LB). TSB is a sensitivity analysis, not a second primary result.
"""
import os
import pyxlsb
import pandas as pd

XLSB_PATH = "data/klebphacol/Supplementary_Tables_R2.xlsb"
OUT_DIR = "results/klebphacol"


def _as_label(v):
    """Some strain names are purely numeric (e.g. '16', '2660') and pyxlsb
    returns those as float, not str -- coerce whole-number floats to their
    integer string form so they still match Table S1's isolate names."""
    if isinstance(v, str):
        return v
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return v


def read_s4_block(rows, header_row_idx, n_strains=74):
    """From the strain-name header row, read forward until a row with no
    phage name (col 0 None) is hit -- i.e. the blank filler / next section."""
    header = [c.v for c in rows[header_row_idx]]
    strains = [_as_label(v) for v in header[1:1 + n_strains]]
    assert all(isinstance(s, str) for s in strains), \
        f"Expected {n_strains} strain name strings at cols 1-{n_strains}, got: {strains[:5]}..."

    data = []
    r = header_row_idx + 1
    while r < len(rows):
        row_vals = [c.v for c in rows[r]]
        phage = row_vals[0]
        if phage is None:
            break
        titres = row_vals[1:1 + n_strains]
        data.append((phage, titres))
        r += 1
    return strains, data


def build_long_table(strains, data):
    recs = []
    for phage, titres in data:
        for strain, val in zip(strains, titres):
            recs.append((phage, strain, val))
    df = pd.DataFrame(recs, columns=["phage", "strain", "raw_value"])
    df["raw_value"] = df["raw_value"].fillna(0.0).astype(float)
    return df


def to_strict(df):
    out = df[df.raw_value != -1].copy()
    out["label"] = (out.raw_value > 0).astype(int)
    return out[["phage", "strain", "label"]]


def to_permissive(df):
    out = df.copy()
    out["label"] = ((out.raw_value > 0) | (out.raw_value == -1)).astype(int)
    return out[["phage", "strain", "label"]]


def report(name, df):
    n = len(df)
    pos = int(df.label.sum())
    print(f"  {name:<28} n={n:>6,}  positives={pos:>5,}  base_rate={100*pos/n:.1f}%")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with pyxlsb.open_workbook(XLSB_PATH) as wb:
        with wb.get_sheet("Table S4") as sheet:
            rows = list(sheet.rows())

    # locate section header rows by content, not hardcoded index, so this
    # doesn't silently misalign if the sheet layout shifts
    lb_header_idx = next(i for i, r in enumerate(rows)
                          if r and r[0].v and "LB titres" in str(r[0].v))
    tsb_header_idx = next(i for i, r in enumerate(rows)
                           if r and r[0].v and "TSB titres" in str(r[0].v))
    print(f"LB section header at row {lb_header_idx}, strain-name row at {lb_header_idx + 1}")
    print(f"TSB section header at row {tsb_header_idx}, strain-name row at {tsb_header_idx + 1}")

    lb_strains, lb_data = read_s4_block(rows, lb_header_idx + 1)
    tsb_strains, tsb_data = read_s4_block(rows, tsb_header_idx + 1)
    assert lb_strains == tsb_strains, "LB and TSB strain header order differ"
    print(f"LB phages: {len(lb_data)}   TSB phages: {len(tsb_data)}   strains: {len(lb_strains)}")

    lb_df = build_long_table(lb_strains, lb_data)
    tsb_df = build_long_table(tsb_strains, tsb_data)

    print(f"\nLB raw_value counts: {lb_df.raw_value.apply(lambda v: 'pos' if v>0 else ('undetermined' if v==-1 else 'neg')).value_counts().to_dict()}")
    print(f"TSB raw_value counts: {tsb_df.raw_value.apply(lambda v: 'pos' if v>0 else ('undetermined' if v==-1 else 'neg')).value_counts().to_dict()}")

    outputs = {
        "interactions_LB_strict.csv": to_strict(lb_df),
        "interactions_LB_permissive.csv": to_permissive(lb_df),
        "interactions_TSB_strict.csv": to_strict(tsb_df),
        "interactions_TSB_permissive.csv": to_permissive(tsb_df),
    }

    print("\n" + "=" * 60)
    print("STAGE 1 CHECKPOINT")
    print("=" * 60)
    for fname, df in outputs.items():
        df.to_csv(os.path.join(OUT_DIR, fname), index=False)
        report(fname, df)

    lb_strict_n = len(outputs["interactions_LB_strict.csv"])
    lb_strict_pos = int(outputs["interactions_LB_strict.csv"].label.sum())
    lb_perm_pos = int(outputs["interactions_LB_permissive.csv"].label.sum())
    lb_perm_n = len(outputs["interactions_LB_permissive.csv"])
    tsb_strict_pos = int(outputs["interactions_TSB_strict.csv"].label.sum())
    # positives (>0) are identical between strict and permissive for the same
    # medium -- permissive only adds the -1 rows on top -- so "486"/"638" are
    # the raw >0 counts, checked against STRICT (permissive would be higher
    # by however many -1 rows exist)
    print(f"\nExpected vs actual (paper-verified figures from the task spec):")
    print(f"  LB strict positives:  expected 486   actual {lb_strict_pos}"
          f"  {'MATCH' if lb_strict_pos == 486 else 'MISMATCH -- STOP'}")
    print(f"  TSB strict positives: expected 638   actual {tsb_strict_pos}"
          f"  {'MATCH' if tsb_strict_pos == 638 else 'MISMATCH -- STOP'}")
    print(f"  LB strict base rate:     expected 13.5% actual {100*lb_strict_pos/lb_strict_n:.1f}%  (n={lb_strict_n}, expected n=3611)")
    print(f"  LB permissive base rate: expected 18.8% actual {100*lb_perm_pos/lb_perm_n:.1f}%  (n={lb_perm_n})")


if __name__ == "__main__":
    main()
