#!/usr/bin/env python3
"""
klebphacol_stage5c_stratum_composition.py — is the 23-phage TropiSEQ-covered
subset enriched for near-identical/related relative to the full 52-phage
set? If so, Stage 5b Part 1's pooled PHL-RBP+S 0.411 is largely the
near-identical-stratum weakness (Stage 4/4b's already-characterised
inversion) resurfacing under a different selection, not an independent
head-to-head loss to TropiSEQ. Compares the two models WITHIN each stratum
on the covered subset (not pooled), with n_pos stated throughout.
"""
import pandas as pd

OUT_DIR = "Results/klebphacol"


def main():
    rbp_full = pd.read_csv(f"{OUT_DIR}/stage3_rbps_tagged.csv")
    phage_stratum = (rbp_full.assign(risk=rbp_full.stratum.map(
        {"novel": 0, "related": 1, "near-identical": 2}))
        .groupby("phage_ID")["risk"].max()
        .map({0: "novel", 1: "related", 2: "near-identical"}))

    covered_phages = {'Roth08', 'Roth09', 'Roth10', 'Roth24', 'Roth30', 'Roth42',
                       'Roth44', 'Roth47', 'Roth61', 'Roth71', 'Roth72', 'Roth74',
                       'Roth75', 'Roth83', 'Roth84', 'Roth85', 'Roth87', 'Roth93',
                       'Roth96', 'RothD', 'RothG', 'RothI', 'RothJ'}

    print("Phage-level stratum, full 52:")
    print(phage_stratum.value_counts())
    print(f"\nPhage-level stratum, 23 covered phages:")
    covered_strat = phage_stratum.loc[phage_stratum.index.isin(covered_phages)]
    print(covered_strat.value_counts())

    for label, path in {
        "LB strict": f"{OUT_DIR}/interactions_LB_strict.csv",
        "LB permissive": f"{OUT_DIR}/interactions_LB_permissive.csv",
        "TSB strict": f"{OUT_DIR}/interactions_TSB_strict.csv",
    }.items():
        inter = pd.read_csv(path)
        inter["stratum"] = inter.phage.map(phage_stratum)
        full = inter
        cov = inter[inter.phage.isin(covered_phages)]

        print(f"\n{'='*66}\n{label}: stratum composition, full 52 vs 23-covered\n{'='*66}")
        print(f"{'stratum':<16}{'full: pairs':>14}{'full: %pairs':>14}{'full: pos':>11}"
              f"{'full: %pos':>12}  |{'cov: pairs':>12}{'cov: %pairs':>13}{'cov: pos':>10}{'cov: %pos':>11}")
        for s in ("novel", "related", "near-identical"):
            f_n = (full.stratum == s).sum()
            f_pos = full[full.stratum == s].label.sum()
            c_n = (cov.stratum == s).sum()
            c_pos = cov[cov.stratum == s].label.sum()
            print(f"{s:<16}{f_n:>14}{100*f_n/len(full):>13.1f}%{f_pos:>11}"
                  f"{100*f_pos/full.label.sum():>11.1f}%  |{c_n:>12}{100*c_n/len(cov):>12.1f}%"
                  f"{c_pos:>10}{100*c_pos/cov.label.sum():>10.1f}%")
        print(f"{'TOTAL':<16}{len(full):>14}{100.0:>13.1f}%{full.label.sum():>11}"
              f"{100.0:>11.1f}%  |{len(cov):>12}{100.0:>12.1f}%{cov.label.sum():>10}{100.0:>10.1f}%")


if __name__ == "__main__":
    main()
