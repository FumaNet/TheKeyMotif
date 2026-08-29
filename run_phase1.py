"""
run_phase1.py — run the phase-1 experiments back to back, then collate.

    python run_phase1.py --dry-run     # show the plan
    python run_phase1.py               # run everything outstanding
    python run_phase1.py --only 2d     # just the identity sweep
    python run_phase1.py --collate     # re-print the tables, run nothing
    python run_phase1.py --force       # rerun even if outputs exist

Behaviour:
  * Steps run in order; a step whose output already exists is SKIPPED, so an
    interrupted run resumes by rerunning the same command.
  * A failing step does NOT stop the rest. The summary reports what happened.
  * Console output is tee'd to logs/phase1_<step>.log.
  * At the end it READS the result files and prints the two comparison tables,
    so the answer is on screen rather than buried in logs.

Steps:
  identity sweep   phage-split AUC at 100/95/90/80% RBP identity. Gives the
                   phage side a graded test matching the bacterial 100-75%
                   sweep, closing the asymmetry objection to the 0.749 result.
  consistency      is PHL-M's advantage the motif, or just a coherent training
                   set? 'central' is the sharp one -- it mimics what MEME
                   rewards without running MEME.
"""

import argparse
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime

TSTR = ["100", "99.5", "99", "95", "90", "85", "80", "75"]

STEPS = [
    # (id, label, argv, output file that marks completion)
    ("2d-100", "phage split @ 100% identity",
     ["2d_phage_holdout.py", "--cluster-identity", "1.00", "--dedup"],
     "results/2d_phage_holdout_id100.pkl"),
    ("2d-95", "phage split @ 95% identity",
     ["2d_phage_holdout.py", "--cluster-identity", "0.95", "--dedup"],
     "results/2d_phage_holdout_id95.pkl"),
    ("2d-90", "phage split @ 90% identity",
     ["2d_phage_holdout.py", "--cluster-identity", "0.90", "--dedup"],
     "results/2d_phage_holdout_id90.pkl"),
    ("2d-80", "phage split @ 80% identity",
     ["2d_phage_holdout.py", "--cluster-identity", "0.80", "--dedup"],
     "results/2d_phage_holdout_id80.pkl"),
    ("5b-central", "consistency control: central",
     ["5b_consistency_control.py", "--rule", "central"],
     "results/5b_AUCs_consistency_central.pkl"),
    ("5b-longest", "consistency control: longest",
     ["5b_consistency_control.py", "--rule", "longest"],
     "results/5b_AUCs_consistency_longest.pkl"),
    ("5b-first", "consistency control: first",
     ["5b_consistency_control.py", "--rule", "first"],
     "results/5b_AUCs_consistency_first.pkl"),
]

PUBLISHED = {
    "PHL-M (motif)": [0.658, 0.652, 0.656, 0.651, 0.647, 0.636, 0.638, 0.635],
    "PHL-RBP+S": [0.817, 0.747, 0.690, 0.644, 0.636, 0.615, 0.659, 0.672],
}


def run_step(step, logdir, device):
    sid, label, argv, out = step
    log = os.path.join(logdir, f"phase1_{sid}.log")
    env = dict(os.environ, KM_DEVICE=device, PYTHONUNBUFFERED="1")
    env.pop("KM_THRESHOLDS", None)          # these scripts manage their own

    print(f"  [{sid}] {label}")
    print(f"  [{sid}] log -> {log}")
    t0 = time.time()
    try:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n{'='*70}\n{datetime.now():%Y-%m-%d %H:%M:%S}  "
                     f"{' '.join(argv)}\n{'='*70}\n")
            fh.flush()
            proc = subprocess.Popen([sys.executable] + argv,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True, env=env, bufsize=1)
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                low = line.lower()
                if any(k in low for k in ("auc", "groups", "error", "traceback",
                                          "wrote", "chose", "clustering")):
                    print(f"      {line.rstrip()[:110]}")
            proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    mins = (time.time() - t0) / 60
    ok = proc.returncode == 0 and os.path.exists(out)
    return ("complete" if ok else f"failed (rc={proc.returncode})"), mins


def collate():
    print("\n" + "=" * 72)
    print("IDENTITY SWEEP — phage split, RBP+serotype, deduplicated")
    print("=" * 72)
    print("Compare against the bacterial-split value of 0.817 and the")
    print("exact-match phage-split value of 0.749.\n")
    print(f"{'RBP identity':<16}{'ROC-AUC':>10}{'PR-AUC':>10}")
    any_id = False
    for pct in ["100", "95", "90", "80"]:
        f = f"results/2d_phage_holdout_id{pct}.pkl"
        if not os.path.exists(f):
            print(f"{pct + '%':<16}{'—':>10}{'—':>10}")
            continue
        any_id = True
        labels, scores, a = pickle.load(open(f, "rb"))
        from sklearn.metrics import auc as _auc, precision_recall_curve
        pr, rc, _ = precision_recall_curve(labels, scores)
        print(f"{pct + '%':<16}{a:>10.3f}{_auc(rc, pr):>10.3f}")
    if any_id:
        print("\nStable across thresholds -> generalisation is solid.")
        print("Decaying as the threshold loosens -> matching transfers to")
        print("moderately novel RBPs but not distant ones (still a real,")
        print("narrower finding).")

    print("\n" + "=" * 72)
    print("CONSISTENCY CONTROL — is PHL-M's edge the motif, or coherence?")
    print("=" * 72)
    hdr = f"{'selection rule':<22}" + "".join(f"{t:>7}" for t in TSTR)
    print(hdr)
    for name, vals in PUBLISHED.items():
        print(f"{name:<22}" + "".join(f"{v:>7.3f}" for v in vals))
    print(f"{'PHL-Random':<22}{0.545:>7.3f}  (best across thresholds)")
    print("-" * len(hdr))
    for rule in ["central", "longest", "first"]:
        f = f"results/5b_AUCs_consistency_{rule}.pkl"
        if not os.path.exists(f):
            print(f"{rule:<22}" + "".join(f"{'—':>7}" for _ in TSTR))
            continue
        d = pickle.load(open(f, "rb"))
        print(f"{rule:<22}" + "".join(f"{t[2]:>7.3f}" for t in d))
    print("""
Reading it:
  PHL-M >> all three   motif carries information beyond consistency
  PHL-M ~= central     the motif is a conservation artifact, not functional
                       (the likely outcome given FIMO found no exclusive motif)
  PHL-M ~= longest     the 'motif' may be tracking protein length
  all ~= 0.545         consistency is not the story either

This control is INDEPENDENT of the leakage question. Read it alongside
fold_internal_motifs.py: together they separate leakage, mere consistency,
and genuine motif signal.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="substring filter on step id, e.g. 2d or 5b")
    ap.add_argument("--device", default=os.environ.get("KM_DEVICE", "cuda"),
                    choices=["cuda", "cpu"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collate", action="store_true",
                    help="Only print the tables from existing results.")
    ap.add_argument("--force", action="store_true",
                    help="Rerun steps whose output already exists.")
    ap.add_argument("--logdir", default="logs")
    args = ap.parse_args()

    if args.collate:
        collate()
        return

    steps = [s for s in STEPS if not args.only or args.only in s[0]]
    os.makedirs(args.logdir, exist_ok=True)

    print("=" * 72)
    print(f"PHASE 1   started {datetime.now():%Y-%m-%d %H:%M}   "
          f"device={args.device}")
    print("=" * 72)
    todo = []
    for sid, label, argv, out in steps:
        if os.path.exists(out) and not args.force:
            print(f"  [{sid:<11}] {label:<36} done")
        else:
            print(f"  [{sid:<11}] {label:<36} to run")
            todo.append((sid, label, argv, out))
    if not todo:
        print("\nNothing outstanding.")
        collate()
        return
    print(f"\n  {len(todo)} step(s) to run; roughly "
          f"{len(todo) * 25} min total (rough estimate).")

    if args.dry_run:
        print("\nDry run. Re-run without --dry-run to start.")
        return

    results = []
    t0 = time.time()
    for step in todo:
        print(f"\n{'-'*72}\n[{datetime.now():%H:%M}] {step[0]}\n{'-'*72}")
        try:
            status, mins = run_step(step, args.logdir, args.device)
        except KeyboardInterrupt:
            print("\nInterrupted. Completed steps are saved; rerun to resume.")
            break
        except Exception as e:
            status, mins = f"crashed: {str(e)[:60]}", 0.0
        print(f"  [{step[0]}] {status}  ({mins:.1f} min)")
        results.append((step[0], step[1], status, mins))

    print("\n" + "=" * 72)
    print(f"SUMMARY   total {(time.time() - t0)/60:.1f} min")
    print("=" * 72)
    for sid, label, status, mins in results:
        print(f"  [{sid:<11}] {label:<36} {status:<22} {mins:6.1f} min")

    collate()


if __name__ == "__main__":
    main()
