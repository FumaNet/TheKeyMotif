# KlebPhaCol external benchmark: PHL-RBP+S vs. TropiSEQ

External validation of PHL-RBP+S against KlebPhaCol (Concha-Eloko is a
co-author of both KlebPhaCol and Boeckaerts' training collection, hence the
Stage 2/2b overlap-removal work below), with TropiSEQ run under identical
conditions as a comparator. All results use LB medium as primary (KlebPhaCol's
default assay, also used by Beamud et al.); TSB is reported as a sensitivity
check. n, n_pos and base rate are stated on every row — PR-AUC is not
comparable across the training base rate (2.77%, post-exclusion) and
KlebPhaCol's (13.5–18.8% depending on medium/label definition), so lift
(PR-AUC / base rate — how many times better than a no-skill classifier, whose
PR-AUC equals the base rate) is reported beside every PR-AUC instead.

## Headline

**Neither method transfers across collections in a way that would support
deployment.** On genuinely novel RBPs (no ≥80% identity match anywhere in
PHL-RBP+S's post-exclusion training RBP set), PHL-RBP+S reaches ROC-AUC
0.649 (LB strict; 95% CI 0.601–0.696, excludes chance) — a real but modest
signal, lift 1.74× over base rate on PR-AUC. TropiSEQ abstains on 87.2% of
the same RBPs (12.8% coverage; 12.7% on its own designed depolymerase-domain
input, so this is not an artifact of scoring it against PHL-RBP+S's RBP
definition — see Coverage, below). Where both models actually have signal to
compare, in the `related` stratum, **neither discriminates well, and where
they can be compared on identical pairs, TropiSEQ has a real but thin
advantage** (below).

---

## 1. PHL-RBP+S: full results, its own RBP set (193 RBPs, all 52 phages)

Trained on Boeckaerts minus 7 RBP-overlap-excluded phages (98 phages,
458,504 pairs, 2.77% base rate); KlebPhaCol held out entirely, no CV.
Reported both including and excluding the 16/74 KlebPhaCol hosts whose KL
type has no training one-hot column (Guard 2) — PHL-RBP+S has no mechanism
to represent an unseen capsule type other than an all-zero vector, so
`incl` is the deployment-realistic number and `excl` isolates RBP-side
generalisation from that host-vocabulary gap. Strata are phage-level,
riskiest-RBP-wins, restratified against the 98-phage post-exclusion
training set (Stage 3b) — not the pre-exclusion 105-phage tagging.

### LB strict

| scope | stratum | n | n_pos | base rate | ROC-AUC | PR-AUC | lift |
|---|---|---:|---:|---:|---:|---:|---:|
| excl unseen-KL | overall | 2842 | 370 | 13.0% | 0.558 | 0.182 | 1.40× |
| excl unseen-KL | **novel** | **1347** | **215** | **16.0%** | **0.649** | **0.279** | **1.74×** |
| excl unseen-KL | related | 225 | 11 | 4.9% | 0.318 | 0.038 | 0.78× |
| excl unseen-KL | near-identical | 1270 | 144 | 11.3% | 0.426 | 0.101 | 0.89× |
| incl unseen-KL | overall | 3611 | 486 | 13.5% | 0.548 | 0.155 | 1.15× |
| incl unseen-KL | novel | 1706 | 307 | 18.0% | 0.647 | 0.271 | 1.72× |
| incl unseen-KL | related | 289 | 13 | 4.5% | 0.351 | 0.039 | 0.83× |
| incl unseen-KL | near-identical | 1616 | 166 | 10.3% | 0.403 | 0.083 | 0.65× |

Novel-stratum bootstrap 95% CI (n_boot=2000): ROC-AUC [0.601, 0.696] (excl),
[0.612, 0.680] (incl). `related` and `near-identical` sit at or below lift
1.0× — no better than guessing the base rate, and `near-identical` is
markedly worse than `novel` despite being (by construction) the stratum with
a close training match. See §4.

### LB permissive / TSB strict (excl-unseen-KL scope)

| medium | stratum | n | n_pos | base rate | ROC-AUC | PR-AUC | lift |
|---|---|---:|---:|---:|---:|---:|---:|
| LB permissive | overall | 3016 | 544 | 18.0% | 0.545 | 0.237 | 1.32× |
| LB permissive | novel | 1508 | 376 | 24.9% | 0.604 | 0.355 | 1.43× |
| LB permissive | related | 232 | 18 | 7.8% | 0.269 | 0.056 | 0.72× |
| LB permissive | near-identical | 1276 | 150 | 11.8% | 0.428 | 0.105 | 0.89× |
| TSB strict | overall | 2728 | 498 | 18.3% | 0.592 | 0.266 | 1.45× |
| TSB strict | novel | 1234 | 325 | 26.3% | 0.654 | 0.415 | 1.58× |
| TSB strict | related | 227 | 10 | 4.4% | 0.277 | 0.032 | 0.73× |
| TSB strict | near-identical | 1267 | 163 | 12.9% | 0.493 | 0.128 | 0.99× |

**RBP-count sensitivity** (KlebPhaCol phages average 3.71 RBPs/phage vs.
Boeckaerts' training 2.61; top-3/phage subsample tests whether this drives
the result): novel-stratum ROC-AUC moves −0.026 (LB strict), −0.011 (LB
permissive), −0.026 (TSB strict). Materially in two of three media —
reported, not smoothed over; the subsampled novel-stratum ROC-AUC is 0.623
(LB strict) / 0.593 (LB permissive) / 0.628 (TSB strict).

---

## 2. TropiSEQ: coverage is the first-class result

| | RBPs scored | phages scored |
|---|---|---|
| On PHL-RBP+S's RBP set (193/187 unique, RBPdetect≥0.5) | 24/187 (**12.8%**) | 23/52 |
| On its own designed input (69/63 unique, DepoScope≥0.5) | 8/63 (**12.7%**) | 5/46 |

**87.2% abstention rate**, statistically identical whether scored against
RBPdetect-called RBPs or its own DepoScope-domain input — the low coverage
is a property of TropiSEQ on this collection, not an artifact of this
benchmark's query-set choice. Fairness rule (best-hit identity ≥95% →
already in TropiSEQ's own training corpus): **0/24 predictions dropped**, a
genuine zero.

**PHL-RBP+S scores 193/193 RBPs and 52/52 phages — 100% coverage, by
construction.** This is not a capability advantage: PHL-RBP+S has no
mechanism to abstain or express "no basis for a prediction here," so every
KlebPhaCol RBP gets a confident-looking score whether or not the model has
anything resembling it in training. A model that never says "I don't know"
is a limitation, not a strength, and this asymmetry should inform how
either tool's output is used in practice — TropiSEQ's silence on 87% of
cases is at least legible; PHL-RBP+S's near-chance scores on `related` and
`near-identical` are not distinguishable, on the score alone, from its
genuine novel-stratum signal.

---

## 3. Head-to-head: covered subset, within stratum, not pooled

The only pairs where TropiSEQ produces a real (non-abstained) score are the
23 phages above. Comparing the two models pooled across all strata on this
subset is misleading, because the subset's composition is not
representative:

### Stratum composition, full 52 vs. 23-covered (LB strict; same pattern in
permissive and TSB)

| stratum | full 52: %pairs / %positives | 23-covered: %pairs / %positives |
|---|---|---|
| novel | 47.2% / 63.2% | **4.4% / 1.9%** |
| related | 8.0% / 2.7% | 17.2% / 8.2% |
| near-identical | 44.8% / 34.2% | **78.4% / 89.9%** |

Phage-level: 26/52 (50%) full-set phages are novel-stratum vs. **1/23
(4%)** of the covered subset; 22/52 (42%) are near-identical vs. **18/23
(78%)** covered. The covered subset is almost pure near-identical/related.
**A pooled ROC-AUC on this subset (PHL-RBP+S 0.411, LB strict) is not an
independent head-to-head result — it is >90% the near-identical stratum's
already-characterised weakness (§4) resurfacing under a different
selection.** Reporting it as "PHL-RBP+S loses to TropiSEQ overall" would be
wrong; reporting it at all without this composition context would be
misleading. It is not used as a headline number in this section.

### Within-stratum, same 23-phage pairs, LB strict

| stratum | n / n_pos | PHL-RBP+S ROC-AUC / PR-AUC / lift | TropiSEQ ROC-AUC / PR-AUC / lift |
|---|---|---|---|
| novel | 74 / 3 | 0.155 / 0.048 / — | *(n_pos=3, both models uninterpretable)* |
| **related** | **289 / 13** | 0.351 / 0.039 / 0.78× | **0.692 / 0.412 / 9.16×** |
| near-identical | 1321 / 142 | 0.413 / 0.091 / 0.86× | **no prediction** |

LB permissive: related 296/20, PHL-RBP+S 0.281/0.050/0.74×, TropiSEQ
**0.725/0.487/7.16×**. TSB strict: related 291/11, PHL-RBP+S
0.282/0.029/0.76×, TropiSEQ **0.678/0.266/7.00×**.

TropiSEQ's `near-identical`-stratum and `novel`-stratum cells are marked
**no prediction**, not 0.500: within those strata, most of TropiSEQ's
per-host scores are either the constant 0 (RBP never matched) or a small
number of tied nonzero values shared across every host of a given KL type,
so the resulting AUC is the artifact of a near-uniform score distribution
rather than a discrimination attempt. Reporting 0.500 there would present
non-participation as a comparable result.

**The only genuine head-to-head is `related`: a real, if thin (n_pos
11–20), TropiSEQ advantage (ROC-AUC 0.68–0.73, lift 7–9×) over PHL-RBP+S
(ROC-AUC 0.28–0.35, lift <1×) in every medium tested.** This is the one
result in this benchmark that should be read as a genuine architecture
comparison rather than a coverage or composition artifact.

---

## 4. Standalone findings

**RBP-serotype reassignment is near-total in the near-identical stratum.**
Every one of 34 near-identical KlebPhaCol RBPs (22 phages), when queried in
KlebPhaCol, is paired with at least one real positive host serotype the
matched training RBP's phage was never positively associated with in
Boeckaerts training (166/166 positive near-identical pairs, 100%; 1596/1616
of all near-identical pairs, 98.8%). A mechanism test for "the model
memorised RBP-to-serotype pairings" could not be run as designed — the
MATCHED comparison group has 20 pairs and zero positives, so its AUC is
undefined, not measurably near the MISMATCHED group's ~0.43 — but the raw
count stands regardless of that test's outcome: **sequence identity between
a query RBP and a training RBP does not imply the training label transfers
to a new host context.** A caution for any pipeline (this one included)
that treats "this protein looks like one I've seen" as license to reuse its
associated supervision.

**RBP-level and genome-level redundancy controls do not nest.** Stage 2's
genome-ANI screen (>95% whole-genome ANI → exclude) and Stage 2b's
RBP-identity screen (>95% RBP identity → exclude) disagree in both
directions on the same 105-phage training set:
- K7PH164C4 is flagged by genome ANI (96.2% to KlebPhaCol's Roth50) but has
  **no RBP above 95% identity to anything in KlebPhaCol** — a genomically
  similar phage whose RBPs happen not to be the similar part.
- K2064PH2, K30lambda2, K40PH129C1, K52PH129C1 carry an RBP ≥95% identical
  to a KlebPhaCol RBP but were never anyone's best genome-ANI hit above
  95% — an RBP shared or recombined across phages whose genomes otherwise
  diverged enough to be genuinely different.

Genome ANI is the wrong redundancy control for an RBP-based predictor, in
both directions — it both misses real feature-level leakage (case 2) and
over-excludes on the strength of whole-genome similarity that the model
never actually sees (case 1). RBP-identity is exclusion Stage 4 trained
against; genome-ANI exclusion (Stage 2) was superseded, not used.

**Both tools degrade on phages resembling previously characterised ones,
via different mechanisms.** TropiSEQ's coverage and PHL-RBP+S's
within-stratum weakness are not two separate findings — the 23-phage
covered subset's near-identical enrichment (§3) ties them to the same root
property. TropiSEQ only produces a score when a query RBP clears its
BLAST bitscore/coverage bar against a reference database — i.e. only for
RBPs resembling something already characterised — and even then its
`near-identical`-stratum output is largely non-discriminating ties (§3).
PHL-RBP+S always produces a score, but its `near-identical`-stratum score
is no better (often worse) than its `novel`-stratum one, consistent with
§4's serotype-reassignment finding: memorisation-shaped behaviour that
misleads rather than helps when a familiar-looking RBP recurs in an
unfamiliar host context. One tool expresses this as silence, the other as
confident wrong answers — the underlying vulnerability (both are, in
different ways, "keyed on similarity to the training set" rather than a
transferable binding rule) looks like the same thing from two directions.

---

## 5. Limitations to keep in the text, not a footnote

**Input difference.** PHL-RBP+S embeds whole RBPdetect-called RBPs
(ESM-2, full protein). TropiSEQ does direct sequence identity of
DepoScope/BLAST-defined depolymerase *domains* against a reference
database. These are not always the same region of the same protein —
RBPdetect and DepoScope don't reliably call the same boundary, or even the
same protein, as the receptor-binding/depolymerase element. Every
comparison in this document inherits this confound: a performance gap
could reflect architecture, feature definition, or both, and nothing here
can cleanly separate them. This is a genuine limitation of the comparison,
not a caveat to mention once and set aside.

**Sample size.** `related`-stratum results (n_pos 10–20 throughout) and the
covered-subset `novel` cell (n_pos=3) are thin. Reported with n and n_pos
on every row for exactly this reason — treat any single-stratum result with
n_pos under ~20 as suggestive, not confirmatory.

**Host serotype vocabulary gap.** 16/74 KlebPhaCol hosts (22%) have a KL
type absent from PHL-RBP+S's training one-hot vocabulary; those hosts'
metrics, reported separately in Stage 4, are at or below chance
(ROC-AUC 0.458–0.505), a real and structural limitation of a one-hot
capsule representation against a growing, open-ended set of capsule types
— not fixable by more training data of the same kind.

**Single train/test split.** No cross-validation on the training side (by
design — KlebPhaCol is the held-out set); bootstrap CIs quantify sampling
uncertainty on the KlebPhaCol side only, not model-training variance.
