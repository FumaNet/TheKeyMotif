#!/usr/bin/env python3
"""
klebphacol_fetch_genomes.py — download KlebPhaCol phage genomes (Table S2
GenBank accessions) and KlebPhaCol host genome assemblies (Table S1
Bioproject/Accession number) for the fastANI overlap check against the
Boeckaerts collections.

Table S1's "Accession number" column mixes three accession types, each
needing different resolution before a genome FASTA can be fetched:
  - direct nuccore (CP.../NZ_CP.../NC_...): efetch db=nuccore works directly
  - WGS master (e.g. JBKFVS000000000): NOT a `datasets` assembly accession;
    resolve via esearch db=assembly -> esummary -> AssemblyAccession (GCA_/
    GCF_), then `datasets download genome accession <that>`
  - BioSample (SAMN...): same resolution path as WGS master

5 of 74 hosts (NCTC_13368, ATCC_11296, NCTC_13438, NCTC_7427, NCTC_13443)
have NO accession of any kind in Table S1 ("NA" in both Bioproject and
Accession number columns). These are well-known NCTC/ATCC reference strains
and could probably be found by searching NCBI on the strain name, but doing
that would mean choosing which public genome "is" that strain by name, not
by anything in the source table -- exactly the kind of un-traceable
assumption the task instructions were explicit about avoiding for the
cross-collection SEQUENCE match itself. Flagged and left unfetched; reported
at the end rather than silently guessed.
"""
import os
import re
import sys
import time
import subprocess
import pyxlsb

XLSB_PATH = "Data/klebphacol/Supplementary_Tables_R2.xlsb"
PHAGE_OUT_DIR = "Data/genomes/klebphacol_phages"
HOST_OUT_DIR = "Data/genomes/klebphacol_hosts"
NCBI_EMAIL = "fumanet@outlook.com"
DATASETS = ["conda", "run", "-n", "genomics", "datasets"]


def get_s2_phage_accessions():
    with pyxlsb.open_workbook(XLSB_PATH) as wb:
        with wb.get_sheet("Table S2") as sheet:
            rows = list(sheet.rows())
    accs = []
    for r in rows[2:]:
        name = r[0].v if len(r) > 0 else None
        if name is None:
            break
        accs.append((name, r[1].v))
    return accs


def get_s1_host_accessions():
    with pyxlsb.open_workbook(XLSB_PATH) as wb:
        with wb.get_sheet("Table S1") as sheet:
            rows = list(sheet.rows())
    header = [c.v for c in rows[2]]
    idx_name = header.index("Isolate name")
    idx_bp = header.index("Bioproject")
    idx_acc = header.index("Accession number")
    out = []
    for r in rows[3:]:
        vals = [c.v for c in r]
        name = vals[idx_name]
        if name is None:
            break
        if isinstance(name, float) and name.is_integer():
            name = str(int(name))
        bp = vals[idx_bp]
        acc = vals[idx_acc]
        acc_s = str(acc).strip() if acc is not None else None
        out.append((name, bp, acc_s))
    return out


def fetch_phage_fasta(accession, out_path, retries=3):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True
    url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           f"?db=nuccore&id={accession}&rettype=fasta&retmode=text"
           f"&email={NCBI_EMAIL}&tool=klebphacol_benchmark")
    for attempt in range(retries):
        r = subprocess.run(["curl", "-s", "--max-time", "30", url],
                            capture_output=True, text=True)
        if r.stdout.startswith(">"):
            with open(out_path, "w") as f:
                f.write(r.stdout)
            return True
        time.sleep(2)
    return False


def resolve_to_gca(query, retries=3):
    """WGS-master or BioSample accession -> (GCA_/GCF_ accession) via
    esearch db=assembly + esummary. Returns None if no unique hit."""
    for attempt in range(retries):
        r = subprocess.run(["curl", "-s", "--max-time", "20",
                             f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                             f"?db=assembly&term={query}&email={NCBI_EMAIL}"],
                            capture_output=True, text=True)
        ids = re.findall(r"<Id>(\d+)</Id>", r.stdout)
        if len(ids) == 1:
            r2 = subprocess.run(["curl", "-s", "--max-time", "20",
                                  f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                                  f"?db=assembly&id={ids[0]}&email={NCBI_EMAIL}"],
                                 capture_output=True, text=True)
            m = re.search(r"<AssemblyAccession>([^<]+)", r2.stdout)
            if m:
                return m.group(1)
            return None
        elif len(ids) == 0:
            time.sleep(2)
            continue
        else:
            return None  # ambiguous, don't guess
    return None


def fetch_assembly_fasta(gca_accession, out_path, retries=2):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True
    zip_path = out_path + ".zip"
    for attempt in range(retries):
        r = subprocess.run(DATASETS + ["download", "genome", "accession",
                                        gca_accession, "--include", "genome",
                                        "--filename", zip_path],
                            capture_output=True, text=True, timeout=120)
        if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1000:
            import zipfile
            try:
                with zipfile.ZipFile(zip_path) as z:
                    fasta_members = [n for n in z.namelist() if n.endswith(".fna")]
                    if fasta_members:
                        with z.open(fasta_members[0]) as src, open(out_path, "wb") as dst:
                            dst.write(src.read())
                        os.remove(zip_path)
                        return True
            except zipfile.BadZipFile:
                pass
        if os.path.exists(zip_path):
            os.remove(zip_path)
        time.sleep(2)
    return False


def main():
    os.makedirs(PHAGE_OUT_DIR, exist_ok=True)
    os.makedirs(HOST_OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("KlebPhaCol PHAGE GENOMES (Table S2 GenBank accessions)")
    print("=" * 60)
    phage_accs = get_s2_phage_accessions()
    print(f"{len(phage_accs)} phages listed in Table S2")
    failed_phages = []
    for i, (name, acc) in enumerate(phage_accs):
        out_path = os.path.join(PHAGE_OUT_DIR, f"{name}__{acc}.fasta")
        ok = fetch_phage_fasta(acc, out_path)
        if not ok:
            failed_phages.append((name, acc))
        print(f"  [{i+1}/{len(phage_accs)}] {name} ({acc}): {'OK' if ok else 'FAILED'}")
        time.sleep(0.4)

    print("\n" + "=" * 60)
    print("KlebPhaCol HOST GENOMES (Table S1)")
    print("=" * 60)
    host_accs = get_s1_host_accessions()
    print(f"{len(host_accs)} hosts listed in Table S1")

    na_hosts = [(n, bp, ac) for n, bp, ac in host_accs if ac in (None, "NA", "")]
    resolvable = [(n, bp, ac) for n, bp, ac in host_accs if ac not in (None, "NA", "")]
    print(f"No accession in table at all: {len(na_hosts)} -> "
          f"{[n for n, bp, ac in na_hosts]}")
    print(f"Have some accession to resolve: {len(resolvable)}")

    failed_hosts = list(na_hosts)  # these fail by construction; already flagged above
    for i, (name, bp, acc) in enumerate(resolvable):
        out_path = os.path.join(HOST_OUT_DIR, f"{name}.fasta")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"  [{i+1}/{len(resolvable)}] {name}: already fetched")
            continue
        if re.match(r'^(CP|NZ_CP|NC_)\d+', acc):
            ok = fetch_phage_fasta(acc, out_path)  # same efetch nuccore path
            resolved_as = acc
        else:
            gca = resolve_to_gca(acc)
            if gca is None:
                print(f"  [{i+1}/{len(resolvable)}] {name} ({acc}): "
                      f"COULD NOT RESOLVE to a unique assembly accession")
                failed_hosts.append((name, bp, acc))
                continue
            resolved_as = gca
            ok = fetch_assembly_fasta(gca, out_path)
        if not ok:
            failed_hosts.append((name, bp, acc))
        print(f"  [{i+1}/{len(resolvable)}] {name} ({acc} -> {resolved_as}): "
              f"{'OK' if ok else 'FAILED'}")
        time.sleep(0.4)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Phages: {len(phage_accs) - len(failed_phages)}/{len(phage_accs)} retrieved")
    if failed_phages:
        print(f"  FAILED: {failed_phages}")
    print(f"Hosts:  {len(host_accs) - len(failed_hosts)}/{len(host_accs)} retrieved")
    if failed_hosts:
        print(f"  UNRETRIEVED: {failed_hosts}")

    if failed_phages or failed_hosts:
        print(f"\nSTOPPING: partial genome set ({len(failed_hosts)} hosts, "
              f"{len(failed_phages)} phages unretrieved). Not proceeding to "
              f"ANI with incomplete data -- see task instructions.")
        sys.exit(1)
    print("\nAll genomes retrieved successfully.")


if __name__ == "__main__":
    main()
