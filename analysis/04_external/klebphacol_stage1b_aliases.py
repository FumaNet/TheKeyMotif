#!/usr/bin/env python3
"""
klebphacol_stage1b_aliases.py — build and verify the Table S4 <-> Table S1
strain name alias map, and verify the Table S4 <-> Table S2 phage name join.

Manual curation (the 4 known S4-vs-S1 naming mismatches, found by direct set
comparison in the Stage 1 checkpoint) is written to a versioned CSV rather
than kept inline in code, so it's inspectable and diffable:
    data/klebphacol/strain_aliases.csv   (columns: s4_name, s1_name, note)

No fuzzy matching anywhere. Every S4 strain name must either equal an S1
isolate name exactly, or be one of the 4 pre-identified exceptions below. If
a name matches neither, this raises and stops -- it does not guess.

Then asserts completeness rather than assuming it:
  - all 74 S4 strain names map to exactly one S1 row
  - no S1 row is claimed by more than one S4 name
  - all 52 S4 phage names map to exactly one S2 row (direct match; Stage 1's
    checkpoint already found phage names identical, verified again here)
  - no S2 row is claimed twice
Fails loudly (raises) on any violation, and reports the count either way.
"""
import os
import pyxlsb
import pandas as pd

XLSB_PATH = "data/klebphacol/Supplementary_Tables_R2.xlsb"
ALIAS_PATH = "data/klebphacol/strain_aliases.csv"

# The 4 exceptions found in Stage 1's checkpoint (S4 name -> S1 name), by
# direct set-difference inspection of the workbook -- not inferred, not
# fuzzy-matched. If the workbook ever changes and a 5th S4 name fails to
# match S1 directly AND isn't in this dict, build_alias_map() raises.
KNOWN_EXCEPTIONS = {
    "16": ("16 (aka KP16)", "S1 carries a parenthetical alias; same isolate"),
    "MKP103": ("MKP103 (aka KPNIH1)", "S1 carries a parenthetical alias; same isolate"),
    "NCTC 7427": ("NCTC_7427", "space vs underscore between S4 and S1"),
    "164413U12": ("164413U/2 (aka 164413U12)", "S1 carries a parenthetical alias; same isolate"),
}


def _as_label(v):
    if isinstance(v, str):
        return v
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return v


def load_s1_isolate_names():
    with pyxlsb.open_workbook(XLSB_PATH) as wb:
        with wb.get_sheet("Table S1") as sheet:
            rows = list(sheet.rows())
    names = []
    for r in rows[3:]:
        v = r[1].v if len(r) > 1 else None
        if v is None:
            break
        names.append(_as_label(v))
    return names


def load_s4_strain_names():
    with pyxlsb.open_workbook(XLSB_PATH) as wb:
        with wb.get_sheet("Table S4") as sheet:
            rows = list(sheet.rows())
    header_idx = next(i for i, r in enumerate(rows)
                       if r and r[0].v and "LB titres" in str(r[0].v)) + 1
    header = [c.v for c in rows[header_idx]]
    return [_as_label(v) for v in header[1:75]]


def load_s2_phage_names():
    with pyxlsb.open_workbook(XLSB_PATH) as wb:
        with wb.get_sheet("Table S2") as sheet:
            rows = list(sheet.rows())
    names = []
    for r in rows[2:]:
        v = r[0].v if len(r) > 0 else None
        if v is None:
            break
        names.append(v)
    return names


def load_s4_phage_names():
    lb = pd.read_csv("results/klebphacol/interactions_LB_strict.csv")
    return sorted(lb.phage.unique())


def build_alias_map(s4_strains, s1_names):
    s1_set = set(s1_names)
    rows = []
    for s4_name in s4_strains:
        if s4_name in s1_set:
            rows.append((s4_name, s4_name, "direct match"))
            continue
        if s4_name in KNOWN_EXCEPTIONS:
            s1_name, note = KNOWN_EXCEPTIONS[s4_name]
            if s1_name not in s1_set:
                raise ValueError(
                    f"KNOWN_EXCEPTIONS entry for '{s4_name}' points at "
                    f"'{s1_name}', which is not an S1 isolate name. The "
                    f"workbook changed -- fix KNOWN_EXCEPTIONS.")
            rows.append((s4_name, s1_name, note))
            continue
        raise ValueError(
            f"NEW MISMATCH, not fuzzy-matching: S4 strain name '{s4_name}' "
            f"has no direct match in Table S1 and is not in KNOWN_EXCEPTIONS. "
            f"Stopping -- add it to KNOWN_EXCEPTIONS by hand after checking "
            f"it's really the same isolate, then rerun.")
    return pd.DataFrame(rows, columns=["s4_name", "s1_name", "note"])


def assert_bijective(df_map, left_col, right_col, left_total, right_total, label):
    n_left = df_map[left_col].nunique()
    n_right_claimed = df_map[right_col].nunique()
    dupes_right = df_map[right_col][df_map[right_col].duplicated(keep=False)]
    unmapped_left = left_total - n_left
    if len(dupes_right) > 0:
        raise ValueError(f"{label}: {right_col} values claimed more than once: "
                          f"{sorted(dupes_right.unique())}")
    if n_left != len(df_map):
        raise ValueError(f"{label}: {left_col} has duplicate rows in the map itself")
    if n_left != left_total:
        raise ValueError(f"{label}: mapped {n_left} distinct {left_col}, "
                          f"expected {left_total}")
    if n_right_claimed != right_total:
        raise ValueError(f"{label}: mapped to {n_right_claimed} distinct "
                          f"{right_col}, expected {right_total} (some target "
                          f"rows unclaimed or the map is wrong)")
    print(f"  {label}: {n_left}/{left_total} mapped, "
          f"{n_right_claimed}/{right_total} targets claimed, no duplicates -- OK")


def main():
    print("=" * 60)
    print("STRAIN ALIAS MAP (Table S4 <-> Table S1)")
    print("=" * 60)
    s1_names = load_s1_isolate_names()
    s4_strains = load_s4_strain_names()
    print(f"Table S1 isolate names: {len(s1_names)}")
    print(f"Table S4 strain names:  {len(s4_strains)}")

    alias_df = build_alias_map(s4_strains, s1_names)
    n_exceptions = (alias_df.note != "direct match").sum()
    print(f"Direct matches: {(alias_df.note == 'direct match').sum()}, "
          f"aliased matches: {n_exceptions}")
    if n_exceptions != len(KNOWN_EXCEPTIONS):
        raise AssertionError(
            f"Expected exactly {len(KNOWN_EXCEPTIONS)} aliased strains "
            f"(the ones found in the Stage 1 checkpoint), got {n_exceptions}. "
            f"Something changed -- inspect before trusting this map.")

    os.makedirs(os.path.dirname(ALIAS_PATH), exist_ok=True)
    alias_df.to_csv(ALIAS_PATH, index=False)
    print(f"Wrote {ALIAS_PATH} ({len(alias_df)} rows)")

    print("\nAliased rows (the 4 non-identical ones):")
    print(alias_df[alias_df.note != "direct match"].to_string(index=False))

    print("\nCompleteness check:")
    assert_bijective(alias_df, "s4_name", "s1_name",
                      left_total=74, right_total=74,
                      label="S4 strains -> S1 isolates")

    print("\n" + "=" * 60)
    print("PHAGE NAME JOIN (Table S4 <-> Table S2)")
    print("=" * 60)
    s2_phages = load_s2_phage_names()
    s4_phages = load_s4_phage_names()
    print(f"Table S2 phage names: {len(s2_phages)}")
    print(f"Table S4 phage names: {len(s4_phages)}")

    s2_set = set(s2_phages)
    unmatched = [p for p in s4_phages if p not in s2_set]
    if unmatched:
        raise ValueError(f"S4 phage names with no direct S2 match (not "
                          f"fuzzy-matching): {unmatched}")
    phage_map = pd.DataFrame({"s4_name": s4_phages, "s2_name": s4_phages,
                               "note": "direct match"})
    assert_bijective(phage_map, "s4_name", "s2_name",
                      left_total=52, right_total=52,
                      label="S4 phages -> S2 phages")

    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
