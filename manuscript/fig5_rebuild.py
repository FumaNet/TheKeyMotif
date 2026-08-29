"""Rebuild Fig 5: median pairwise identity between MEME motif occurrences, per serotype.

Identity = normalised Levenshtein identity, 1 - dist/max(len), the same definition
used for RBP redundancy control elsewhere in the paper.
Reproduces the published K11 values (median 21%, 18/21 pairs below 30%).
"""
import os, glob, itertools, statistics, csv
import Levenshtein
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/claude/TheKeyMotif/Motifs_KO"
THRESHOLD = 40.0
MIN_OCC = 3          # need >=3 occurrences for a median over >=3 pairs
DROP = {"K64O1ab"}   # duplicate of K64: identical motif, identical 10 occurrences

def sites(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"): out.append("")
        elif line and out: out[-1] += line
    return out

def ident(a, b):
    return 100.0 * (1 - Levenshtein.distance(a, b) / max(len(a), len(b)))

rows = []
for d in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, d)
    if not os.path.isdir(p) or d == "Analysis" or d in DROP: continue
    fs = glob.glob(os.path.join(p, "*_fasta.txt"))
    if not fs: continue
    s = sites(fs[0])
    if len(s) < MIN_OCC: continue
    ids = [ident(a, b) for a, b in itertools.combinations(s, 2)]
    rows.append(dict(serotype=d, n_occ=len(s), n_pairs=len(ids),
                     motif_width=len(s[0]),
                     median=round(statistics.median(ids), 1),
                     minimum=round(min(ids), 1), maximum=round(max(ids), 1),
                     pairs_below_30=sum(1 for x in ids if x < 30),
                     pairs_below_40=sum(1 for x in ids if x < 40)))

rows.sort(key=lambda r: r["median"])
below = sum(1 for r in rows if r["median"] < THRESHOLD)

print(f"{'sero':<9}{'occ':>4}{'pairs':>6}{'width':>6}{'median':>8}{'min':>7}{'max':>7}{'<30':>5}{'<40':>5}")
for r in rows:
    print(f"{r['serotype']:<9}{r['n_occ']:>4}{r['n_pairs']:>6}{r['motif_width']:>6}"
          f"{r['median']:>8.1f}{r['minimum']:>7.1f}{r['maximum']:>7.1f}"
          f"{r['pairs_below_30']:>5}{r['pairs_below_40']:>5}")
print(f"\n{below} of {len(rows)} serotypes have median identity below {THRESHOLD:.0f}%")

with open("Fig5_motif_identity.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

fig, ax = plt.subplots(figsize=(9.0, 3.4))
colours = ["#C0392B" if r["median"] < THRESHOLD else "#2E6DA4" for r in rows]
ax.bar([r["serotype"] for r in rows], [r["median"] for r in rows],
       color=colours, width=0.72)
ax.axhline(THRESHOLD, ls=":", lw=1.0, color="0.35")
ax.text(-0.4, THRESHOLD + 1.5, f"{THRESHOLD:.0f}%",
        ha="left", va="bottom", fontsize=8, color="0.35")
ax.set_ylim(0, 100)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels([f"{v}%" for v in [0, 20, 40, 60, 80, 100]])
ax.set_ylabel("Median pairwise identity\nbetween motif occurrences", fontsize=9)
ax.set_xlabel("Capsular serotype", fontsize=9)
ax.tick_params(axis="x", labelsize=8, rotation=90)
ax.tick_params(axis="y", labelsize=8)
for side in ("top", "right"): ax.spines[side].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"/mnt/user-data/outputs/Fig5_motif_conservation.{ext}", dpi=300)
print("figure written")
