"""Confirm the 100% partitions match, not just the counts.

Both difflib-ratio and rapidfuzz-Levenshtein agree on 99 groups at threshold
1.00 (grouping_method_compare.py / grouping_method_compare_difflib.py). This
checks the stronger claim: that the two metrics put the SAME phages in the
SAME groups at that threshold, not just the same number of groups by
coincidence.

Read-only. No GPU. Seconds.
"""

from collections import defaultdict

from grouping_method_compare import build_phage_identity_matrix, cluster_union_find
from grouping_method_compare_difflib import build_phage_identity_matrix_difflib


def partition_as_sets(uniq, groups):
    d = defaultdict(set)
    for phage, g in zip(uniq, groups):
        d[g].add(phage)
    return set(frozenset(s) for s in d.values())


def main():
    uniq_rf, mat_rf = build_phage_identity_matrix()
    uniq_dl, mat_dl = build_phage_identity_matrix_difflib()
    assert uniq_rf == uniq_dl, "phage ordering differs between the two builds"

    groups_rf = cluster_union_find(mat_rf, 1.00)
    groups_dl = cluster_union_find(mat_dl, 1.00)

    part_rf = partition_as_sets(uniq_rf, groups_rf)
    part_dl = partition_as_sets(uniq_dl, groups_dl)

    print(f"rapidfuzz (Levenshtein) groups at 1.00: {len(part_rf)}")
    print(f"difflib (ratio, autojunk)  groups at 1.00: {len(part_dl)}")
    print(f"Identical partition (same phages in same groups): {part_rf == part_dl}")

    if part_rf != part_dl:
        only_rf = part_rf - part_dl
        only_dl = part_dl - part_rf
        print(f"\nGroups only under rapidfuzz ({len(only_rf)}):")
        for s in sorted(only_rf, key=lambda s: sorted(s)):
            print(" ", sorted(s))
        print(f"\nGroups only under difflib ({len(only_dl)}):")
        for s in sorted(only_dl, key=lambda s: sorted(s)):
            print(" ", sorted(s))


if __name__ == "__main__":
    main()
