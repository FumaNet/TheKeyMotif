"""
run_overnight.py — run the reproduction stages in priority order, unattended.

    python run_overnight.py                 # run everything still outstanding
    python run_overnight.py --dry-run       # show the plan and exit
    python run_overnight.py --only 0,2b     # run selected stages
    python run_overnight.py --device cpu    # force CPU for all stages

Design notes:

* Stages run in PRIORITY order, cheapest-and-most-informative first. If the
  machine dies at 3am you will still have the results that matter most.
* Every stage is independently checkpointed per threshold. A crash costs one
  threshold, not one stage.
* A failing stage does NOT stop the run. The next one starts; the summary at
  the end reports what happened.
* Each stage's console output is tee'd to logs/<stage>.log for morning reading.
* Nothing here overwrites results_published/.

The stages:

  0   PHL-AVG               finish the 3 remaining thresholds. Fast now that
                            precompute_collapsed is in (~20 min/threshold).
  2b  serotype-only         Reviewer #1's baseline. ~10k rows, minutes total.
                            The single most decisive number you are missing.
  2c  RBP-only              the other half of the bracket. If BOTH single-sided
                            baselines sit near chance while the combination
                            reaches 0.817, the model is learning a genuine
                            RBP-by-serotype matching function.
  2d  phage-held-out CV     same model, PHAGE split instead of bacterial. Tests
                            whether the published number survives unseen phages.
                            One split, not eight thresholds.
  3   PHL-RBP+S             your headline model. SLOW: trains on the full
                            487k-row table per fold (drop_duplicates is a
                            no-op), so budget hours per threshold.

Deliberately NOT included: fold_internal_motifs.py needs MEME on PATH and wants
a --dry-run inspection before you commit compute. Run it by hand tomorrow.
"""

import argparse
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timedelta

TSTR = ["100", "99.5", "99", "95", "90", "85", "80", "75"]

STAGES = [
    {
        "id": "0",
        "script": "0_original_replica_compact.py",
        "ckpt": "results/0_checkpoint.pkl",
        "name": "PHL-AVG (script 0)",
        "why": "finishes the reproduction baseline",
        "est_min_per_threshold": 25,
    },
    {
        "id": "2b",
        "script": "2b_serotype_only.py",
        "ckpt": "results/2b_checkpoint.pkl",
        "name": "serotype-only baseline",
        "why": "Reviewer #1 major comment 2 — is PHL-RBP+S memorising serotype?",
        "est_min_per_threshold": 3,
    },
    {
        "id": "2c",
        "script": "2c_rbp_only.py",
        "ckpt": "results/2c_checkpoint.pkl",
        "name": "RBP-only baseline",
        "why": "Reviewer #1 major comment 2 — the other half of the bracket",
        "est_min_per_threshold": 8,
    },
    {
        "id": "2d",
        "script": "2d_phage_holdout.py",
        "ckpt": None,                       # single split, not per-threshold
        "out": "results/2d_phage_holdout.pkl",
        "name": "phage-held-out CV",
        "why": "the decisive test for memorised phage-serotype associations",
        "est_min_per_threshold": 100,
        "single": True,
    },
    {
        "id": "3",
        "script": "3_max_max_sero_compact.py",
        "ckpt": "results/3_checkpoint.pkl",
        "name": "PHL-RBP+S (script 3)",
        "why": "headline model; the comparison the baseline is measured against",
        "est_min_per_threshold": 90,
    },
]


def remaining(stage):
    """Thresholds a stage still needs (or a single-shot marker)."""
    if stage.get("single"):
        return [] if os.path.exists(stage["out"]) else ["(single split)"]
    ckpt = stage["ckpt"]
    if not os.path.exists(ckpt):
        return list(TSTR)
    try:
        with open(ckpt, "rb") as fh:
            done = pickle.load(fh)
        return [t for t in TSTR if t not in done]
    except Exception:
        return list(TSTR)


def run_stage(stage, device, logdir):
    todo = remaining(stage)
    if not todo:
        print(f"  [{stage['id']}] nothing outstanding — skipping")
        return "already complete", 0.0

    log = os.path.join(logdir, f"stage_{stage['id']}.log")
    env = dict(os.environ)
    env["KM_DEVICE"] = device
    if not stage.get("single"):
        env["KM_THRESHOLDS"] = ",".join(todo)
    else:
        env.pop("KM_THRESHOLDS", None)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"  [{stage['id']}] {len(todo)} threshold(s): {todo}")
    print(f"  [{stage['id']}] log -> {log}")

    t0 = time.time()
    try:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n{'='*70}\n{datetime.now():%Y-%m-%d %H:%M:%S}  "
                     f"{stage['script']}  device={device}  thresholds={todo}\n"
                     f"{'='*70}\n")
            fh.flush()
            proc = subprocess.Popen(
                [sys.executable, stage["script"]],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, bufsize=1)
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                # surface only the lines worth watching from a distance
                low = line.lower()
                if any(k in low for k in ("auc:", "auc ", "error", "traceback",
                                          "wrote ", "checkpoint", "missing")):
                    print(f"      {line.rstrip()[:110]}")
            proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    except Exception as e:
        return f"launch failed: {e}", (time.time() - t0) / 60

    mins = (time.time() - t0) / 60
    left = remaining(stage)
    if proc.returncode != 0:
        return f"exited {proc.returncode}, {len(left)} threshold(s) still missing", mins
    if left:
        return f"incomplete — still missing {left}", mins
    return "complete", mins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated stage ids, e.g. 0,2b")
    ap.add_argument("--device", default=os.environ.get("KM_DEVICE", "cuda"),
                    choices=["cuda", "cpu"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--logdir", default="logs")
    args = ap.parse_args()

    sel = {s.strip() for s in args.only.split(",")} if args.only else None
    stages = [s for s in STAGES if sel is None or s["id"] in sel]

    os.makedirs(args.logdir, exist_ok=True)

    print("=" * 70)
    print(f"OVERNIGHT RUN   started {datetime.now():%Y-%m-%d %H:%M}   "
          f"device={args.device}")
    print("=" * 70)

    total = 0
    for s in stages:
        todo = remaining(s)
        est = len(todo) * s["est_min_per_threshold"]
        total += est
        status = "done" if not todo else f"{len(todo)} threshold(s), ~{est/60:.1f} h"
        print(f"  [{s['id']:<3}] {s['name']:<24} {status}")
        print(f"        {s['why']}")

    print(f"\n  estimated total: ~{total/60:.1f} h "
          f"(finishes around {(datetime.now() + timedelta(minutes=total)):%H:%M})")
    print("  estimates are rough; stage 3 in particular may run long.")

    if args.dry_run:
        print("\nDry run. Re-run without --dry-run to start.")
        return

    if not os.path.exists("results_published"):
        print("\n  WARNING: results_published/ not found. Back up the original")
        print("  pickles before running anything that writes to results/:")
        print("      Copy-Item -Recurse results results_published")
        print("  Continuing in 10s — Ctrl+C to stop.")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    results = []
    t_start = time.time()
    for s in stages:
        print(f"\n{'-'*70}\n[{datetime.now():%H:%M}] STAGE {s['id']}: {s['name']}\n{'-'*70}")
        try:
            status, mins = run_stage(s, args.device, args.logdir)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Checkpoints are intact — rerun to resume.")
            results.append((s["id"], s["name"], "interrupted", 0))
            break
        except Exception as e:
            status, mins = f"crashed: {e}", 0
        print(f"  [{s['id']}] {status}  ({mins:.1f} min)")
        results.append((s["id"], s["name"], status, mins))

    print("\n" + "=" * 70)
    print(f"SUMMARY   total {(time.time() - t_start)/60:.1f} min")
    print("=" * 70)
    for sid, name, status, mins in results:
        print(f"  [{sid:<3}] {name:<24} {status:<40} {mins:6.1f} min")

    print("\nIn the morning:")
    print("  python verify_reproduction.py --device cuda --report rerun_report.json")
    print("  python -c \"import pickle;d=pickle.load(open('results/2b_checkpoint.pkl','rb'));"
          "print({k:v[2] for k,v in d.items()})\"")
    print("\nThen compare serotype-only against published PHL-RBP+S:")
    print("  [0.817, 0.747, 0.690, 0.644, 0.636, 0.615, 0.659, 0.672]")
    print("Close numbers mean the viral side is contributing little — which is")
    print("the finding that decides how the paper gets reframed.")
    print("\nLogs: " + args.logdir + "/stage_*.log")


if __name__ == "__main__":
    main()
