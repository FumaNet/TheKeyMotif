#!/usr/bin/env python3
"""
klebphacol_resolve_missing_hosts.py — writes the record of how the 5
KlebPhaCol hosts with no accession in Table S1 were resolved to a genome.

Table S1 gives NO Bioproject/Accession for NCTC_13368, ATCC_11296,
NCTC_13438, NCTC_7427, NCTC_13443 (both columns "NA"). Per the user's
explicit direction, these were resolved by searching NCBI's assembly
database on the strain name (unlike every other host, which was resolved
from an accession already present in the table) -- so this file exists to
make that name-based resolution inspectable and overridable, and to keep
these 5 rows distinguishable downstream (resolution="name-based").

Search method: esearch db=assembly, term = Klebsiella[Organism] AND
"<strain>"[All Fields] (or without the internal space, when the quoted
phrase search returned nothing -- NCBI's strain metadata for these entries
has no space, e.g. "NCTC13368").

None of the 5 reached "Complete Genome" assembly level -- all are
Contig/Scaffold, consistent with these being older NCTC-3000-project
short-read draft depositions rather than long-read finished assemblies.
Reported as-is rather than silently treated as equivalent to the (mostly
WGS-master, i.e. also draft) accessions used for the other 69 hosts.
"""
import os
import pandas as pd

OUT_PATH = "Results/klebphacol/host_accessions_resolved.csv"

ROWS = [
    dict(strain="NCTC_7427", accession="GCF_983172345.1",
         assembly_level="Contig", source_db="RefSeq", n_candidates=1,
         reason="Only assembly in Klebsiella matching strain-name search "
                "'NCTC 7427'. Organism: Klebsiella pneumoniae."),
    dict(strain="NCTC_13368", accession="GCF_900451875.1",
         assembly_level="Contig", source_db="RefSeq", n_candidates=1,
         reason="Quoted-phrase search for 'NCTC 13368' found nothing; "
                "unquoted/no-space 'NCTC13368' (matching NCBI's own strain "
                "field, which has no space) found exactly one hit. "
                "Organism: Klebsiella quasipneumoniae (reclassified member "
                "of the K. pneumoniae species complex, not K. pneumoniae "
                "sensu stricto -- noted, not corrected)."),
    dict(strain="NCTC_13438", accession="GCF_020251405.1",
         assembly_level="Contig", source_db="RefSeq", n_candidates=1,
         reason="Only assembly matching strain-name search 'NCTC 13438'. "
                "Organism: Klebsiella pneumoniae."),
    dict(strain="NCTC_13443", accession="GCF_900451585.1",
         assembly_level="Contig", source_db="RefSeq", n_candidates=1,
         reason="Quoted 'NCTC 13443' found nothing; no-space 'NCTC13443' "
                "found exactly one hit. Organism: Klebsiella pneumoniae."),
    dict(strain="ATCC_11296", accession="GCF_000826585.2",
         assembly_level="Scaffold", source_db="RefSeq", n_candidates=2,
         reason="2 hits for 'ATCC 11296', but both are versions of the SAME "
                "base assembly (GCF_000826585.1, Contig, 2015 vs "
                "GCF_000826585.2, Scaffold, 2017) -- not two distinct "
                "candidate strains, so not an ambiguous case requiring a "
                "judgement call. Used the later version, which supersedes "
                "the earlier one. Organism: Klebsiella pneumoniae subsp. "
                "ozaenae."),
]


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = pd.DataFrame(ROWS)
    df["resolution"] = "name-based"
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(df)} rows)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
