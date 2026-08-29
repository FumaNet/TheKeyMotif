#!/usr/bin/env python3
"""
verify_reproduction.py — drop in the root of TheKeyMotif and run.

Two modes:

  --preflight   Check the CUDA/XGBoost toolchain and estimate whether each
                model script fits in your VRAM. Run this BEFORE committing to
                a long run. Takes seconds.

  (default)     Compare the ROC-AUCs in Results/*.pkl against the values
                published in "Protein-level prediction of Klebsiella phage
                adsorption identifies conserved receptor-binding motifs"
                (bioRxiv 2026.05.21.726843).

Usage:
    python verify_reproduction.py --preflight
    python verify_reproduction.py
    python verify_reproduction.py --results-dir Results_rerun --device cuda
"""

import argparse
import glob
import json
import os
import pickle
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

THRESHOLDS = ["100%", "99.5%", "99%", "95%", "90%", "85%", "80%", "75%"]

# ROC-AUCs as stored in the repo at time of submission.
PUBLISHED = {
    "0":  ("PHL-AVG",    [0.807, 0.740, 0.695, 0.672, 0.668, 0.655, 0.674, 0.690]),
    "1":  ("PHL-RBP",    [0.796, 0.715, 0.663, 0.634, 0.631, 0.614, 0.641, 0.655]),
    "2":  ("PHL-S",      [0.816, 0.727, 0.663, 0.626, 0.621, 0.600, 0.637, 0.655]),
    "3":  ("PHL-RBP+S",  [0.817, 0.747, 0.690, 0.644, 0.636, 0.615, 0.659, 0.672]),
    "4":  ("PHL-M+",     [0.706, 0.690, 0.687, 0.671, 0.673, 0.658, 0.655, 0.662]),
    "5":  ("PHL-M",      [0.658, 0.652, 0.656, 0.651, 0.647, 0.636, 0.638, 0.635]),
    "6a": ("PHL-Random", [0.545]),
}

# Tolerances. GPU-to-GPU is tighter than GPU-to-CPU: the published numbers came
# off a CUDA build, so matching device class removes one source of drift.
TOL_CUDA = {"0": 0.003, "1": 0.003, "2": 0.003, "3": 0.003,
            "4": 0.008, "5": 0.030, "6a": 0.030}
TOL_CPU  = {"0": 0.005, "1": 0.005, "2": 0.005, "3": 0.005,
            "4": 0.010, "5": 0.030, "6a": 0.030}

# Feature width after each script's in-fold column selection.
#   host ESM-2 (1280) + virus ESM-2 (1280)      = 2560
#   virus ESM-2 (1280) + one-hot serotype (~80) = ~1360
# Feature width after each script's in-fold column selection, and how many
# rows each actually trains on.
#   host ESM-2 (1280) + virus ESM-2 (1280)      = 2560
#   virus ESM-2 (1280) + one-hot serotype (~80) = ~1360
#
# IMPORTANT: scripts 2/3/5 call df.drop_duplicates() after dropping the host_*
# columns, but the host `protein_index` column survives -- so that call removes
# ZERO rows. They train on the full pair table, not a deduplicated one.
# Only script 0 genuinely collapses (via groupby.mean()).
SCRIPT_PROFILE = {
    "0":  ("PHL-AVG",    2560, "collapsed", "groupby.mean() -> 1 row per (host, phage)"),
    "1":  ("PHL-RBP",    2560, "full",      "every host-protein x RBP pair"),
    "2":  ("PHL-S",      1360, "full",      "drop_duplicates() is a no-op -- full table"),
    "3":  ("PHL-RBP+S",  1360, "full",      "drop_duplicates() is a no-op -- full table"),
    "4":  ("PHL-M+",     1360, "undersamp", "relabelled + undersampled negatives"),
    "5":  ("PHL-M",      1360, "undersamp", "relabelled + undersampled negatives"),
    "6b": ("PHL-Random", 1360, "undersamp", "relabelled control"),
}

CACHE = "Data/combined_embeddings_per_protein.csv"


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

def _nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            name, total, used, drv = [s.strip() for s in
                                      out.stdout.strip().splitlines()[0].split(",")]
            return {"name": name, "total_mb": int(total),
                    "used_mb": int(used), "driver": drv}
    except Exception:
        pass
    return None


def _xgb_gpu_smoketest():
    """
    Actually train a tiny model on the GPU. Import success != working CUDA.

    Careful: XGBoost 2.x does NOT raise when device="cuda" but no usable GPU is
    present — it emits a UserWarning and silently trains on CPU. A naive
    try/except therefore reports success on a CPU-only box. We capture warnings
    and treat the fallback messages as failure, which is the whole point: you
    want to know now, not after a 5-hour run that quietly used the wrong device.
    """
    import warnings
    fallback_markers = ("No visible GPU", "not compiled with CUDA",
                        "setting device to CPU")
    try:
        import xgboost as xgb
        from xgboost import XGBClassifier
        X = np.random.rand(512, 32).astype(np.float32)
        y = (np.random.rand(512) > 0.7).astype(int)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = XGBClassifier(tree_method="hist", device="cuda",
                              n_estimators=10, max_depth=3, eval_metric="logloss")
            m.fit(X, y)
            _ = m.predict_proba(X)
        for c in caught:
            msg = str(c.message)
            if any(mark in msg for mark in fallback_markers):
                short = msg.strip().splitlines()[0]
                short = short.split("WARNING:")[-1].strip()[:160]
                return False, xgb.__version__, f"silently fell back to CPU — {short}"
        return True, xgb.__version__, None
    except Exception as e:
        try:
            import xgboost as xgb
            ver = xgb.__version__
        except Exception:
            ver = "?"
        return False, ver, str(e).strip().splitlines()[0][:200]


def _host_ram():
    """Total and available host RAM in MB. Works on Linux, macOS and Windows."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total // 2**20, vm.available // 2**20
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MS()
        st.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullTotalPhys // 2**20, st.ullAvailPhys // 2**20
        return None, None
    try:
        with open("/proc/meminfo") as fh:
            info = {}
            for line in fh:
                k, v = line.split(":", 1)
                info[k] = int(v.strip().split()[0]) // 1024
        return info.get("MemTotal"), info.get("MemAvailable")
    except Exception:
        return None, None


def _cache_rows():
    if not os.path.exists(CACHE):
        return None
    n = 0
    with open(CACHE, "rb") as fh:
        buf = fh.read(1 << 24)
        while buf:
            n += buf.count(b"\n")
            buf = fh.read(1 << 24)
    return max(n - 1, 0)  # minus header


def preflight():
    print("=" * 72)
    print("PREFLIGHT")
    print("=" * 72)
    print(f"python   {platform.python_version()}   {platform.system()} {platform.machine()}")

    gpu = _nvidia_smi()
    free_mb = 0
    if gpu:
        free_mb = gpu["total_mb"] - gpu["used_mb"]
        print(f"gpu      {gpu['name']}  driver {gpu['driver']}")
        print(f"vram     {gpu['total_mb']} MB total, {gpu['used_mb']} MB in use, "
              f"~{free_mb} MB free")
        if gpu["used_mb"] > 400:
            print("         note: something (desktop compositor?) is holding VRAM.")
            print("         Running from a TTY, or driving the display off")
            print("         integrated graphics, buys back several hundred MB.")
    else:
        print("gpu      not detected via nvidia-smi")

    ok, ver, err = _xgb_gpu_smoketest()
    print(f"xgboost  {ver}   CUDA training: {'WORKS' if ok else 'FAILED'}")
    if not ok:
        print(f"         -> {err}")
        print("         Fix: pip install --force-reinstall xgboost==2.1.4")
        print("         (the PyPI wheel bundles CUDA; a conda-forge cpu_only")
        print("          build is the usual culprit)")

    total_mb, avail_mb = _host_ram()
    if total_mb:
        print(f"ram      {total_mb:,} MB total, ~{avail_mb:,} MB available")

    rows = _cache_rows()

    # The data-prep stage is the usual first failure, and it is HOST ram, not
    # VRAM. The original scripts merge in float64 because .astype(np.float32)
    # runs after the merge.
    print()
    print("-" * 72)
    print("HOST RAM: DATA PREP")
    print("-" * 72)
    est_rows = rows if rows else 487_400
    f64 = est_rows * 2560 * 8 / 1e9
    print(f"original prep builds a {est_rows:,} x 2560 frame in float64:"
          f" ~{f64:.1f} GB peak")
    print(f"  ...and would cache it as ~{est_rows * 2560 * 13 / 1e9:.0f} GB of CSV text.")
    if avail_mb and f64 * 1024 > avail_mb * 0.7:
        print(f"  -> WILL FAIL on this machine ({avail_mb:,} MB available).")
        print("     Use keymotif_data.py (compact cache) instead. See SETUP_LOCAL.md.")
    else:
        print("  -> may fit, but keymotif_data.py is still far cheaper.")
    print(f"compact cache instead: ~{est_rows * 6 / 1e6:.0f} MB pair index"
          f" + a few MB of embeddings")

    print()
    print("-" * 72)
    print("VRAM ESTIMATE PER SCRIPT")
    print("-" * 72)

    measured = rows is not None
    if not measured:
        print(f"{CACHE} not built yet — run script 0 once, then re-run")
        print("--preflight for real numbers. Using a placeholder 500,000 rows.")
        rows = 500_000
    else:
        print(f"measured: {rows:,} rows in {CACHE}")

    print()
    print(f"{'script':<7}{'model':<12}{'feats':>7}{'train rows':>13}{'est peak':>12}   verdict")
    for k, (name, feats, mode, _note) in SCRIPT_PROFILE.items():
        if mode == "collapsed":
            r = rows // 49      # groupby.mean() -> ~1 row per (host, phage) pair
        elif mode == "undersamp":
            r = rows // 20      # negatives undersampled to one protein per pair
        else:
            r = rows            # full table (see SCRIPT_PROFILE note)
        # 'hist' quantises to max_bin=256 -> ~1 byte per feature value in the
        # ELLPACK page, plus staging + gradients (~1.6x in practice).
        gb = r * feats * 1.6 / 1e9
        if free_mb:
            verdict = "fits" if gb < (free_mb / 1024) * 0.8 else "TIGHT / likely OOM"
        else:
            verdict = "fits (4 GB)" if gb < 2.6 else "TIGHT on 4 GB"
        print(f"{k:<7}{name:<12}{feats:>7}{r:>13,}{gb:>9.2f} GB   {verdict}")

    print()
    if not measured:
        print("Row multipliers are rules of thumb (/49 collapsed, /20 undersampled).")
        print("Re-run once the cache exists for numbers you can trust.")
        print()
    print("If script 1 comes out TIGHT: run it on CPU rather than lowering")
    print("max_bin. Script 1 already sets max_bin=256, so changing it changes")
    print("the published result; CPU vs GPU only perturbs the 3rd decimal.")
    print()
    print("  python patch_device.py --device cuda")
    print("  python 0_original_replica.py && python 2_... && python 3_...")
    print("  python patch_device.py --device cpu  && python 1_max_max.py")
    print("  python patch_device.py --device cuda")
    print("=" * 72)
    return 0


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def metrics(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)
    return auc(fpr, tpr), auc(rec, prec)


def key_for(filename):
    stem = os.path.basename(filename).split("_")[0]
    return stem if stem in PUBLISHED else None


def verify(results_dir, device, write_report):
    files = sorted(glob.glob(os.path.join(results_dir, "*.pkl")))
    if not files:
        raise SystemExit(f"No .pkl files found in {results_dir}/")

    tol_table = TOL_CUDA if device == "cuda" else TOL_CPU
    gpu = _nvidia_smi() if device == "cuda" else None

    print(f"results dir : {results_dir}")
    print(f"device      : {device}" + (f"  ({gpu['name']})" if gpu else ""))
    print(f"tolerances  : {'GPU (tight)' if device == 'cuda' else 'CPU (relaxed)'}")

    all_ok = True
    report = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "results_dir": results_dir, "device": device,
              "gpu": gpu, "models": {}}

    for path in files:
        k = key_for(path)
        if k is None:
            print(f"[skip] {path} — not a recognised results file")
            continue

        name, ref = PUBLISHED[k]
        tol = tol_table[k]
        data = load(path)

        print(f"\n{'='*72}\n{name}  ({os.path.basename(path)})   tolerance +/-{tol}\n{'='*72}")
        print(f"{'thresh':<8}{'ROC now':>10}{'ROC pub':>10}{'delta':>9}{'':>4}{'PR-AUC':>9}")

        rows = []
        for i, entry in enumerate(data):
            labels, scores = np.asarray(entry[0]), np.asarray(entry[1])
            roc, pr = metrics(labels, scores)
            if i < len(ref):
                delta = roc - ref[i]
                ok = abs(delta) <= tol
                all_ok &= ok
                flag = "ok" if ok else "!!"
                print(f"{THRESHOLDS[i]:<8}{roc:>10.3f}{ref[i]:>10.3f}"
                      f"{delta:>+9.3f}{flag:>4}{pr:>9.3f}")
                rows.append({"threshold": THRESHOLDS[i], "roc": round(roc, 4),
                             "roc_published": ref[i], "delta": round(delta, 4),
                             "pr_auc": round(pr, 4), "pass": bool(ok)})
            else:
                print(f"{THRESHOLDS[i]:<8}{roc:>10.3f}{'-':>10}{'-':>9}{'':>4}{pr:>9.3f}")

        n_pos = int(np.asarray(data[0][0]).sum())
        n_tot = len(data[0][0])
        print(f"\n  positives at 100% threshold: {n_pos} / {n_tot} "
              f"({100*n_pos/n_tot:.2f}%)")
        report["models"][name] = {"file": os.path.basename(path), "rows": rows}

    print("\n" + "=" * 72)
    print("REPRODUCTION: PASS" if all_ok else
          "REPRODUCTION: MISMATCH — check data prep before trusting any rerun")
    print("=" * 72)

    if not all_ok:
        print("\nIf the deltas are large and in the same direction at every")
        print("threshold, suspect the data prep (a stale or partial")
        print(f"{CACHE}) rather than the GPU. Delete it and let script 0 rebuild.")

    print("\nPR-AUC is printed alongside ROC because Reviewers #1 and #3 both")
    print("asked for PR to be primary. These are pooled across folds; per-fold")
    print("confidence intervals need the scripts to retain fold-level scores.")

    if write_report:
        with open(write_report, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote {write_report} — keep it with the rerun as provenance.")

    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true",
                    help="Check CUDA/XGBoost and estimate VRAM per script, then exit.")
    ap.add_argument("--results-dir", default="Results")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="Device the rerun used. Sets the tolerance band.")
    ap.add_argument("--report", metavar="PATH", default=None,
                    help="Write a JSON provenance report.")
    args = ap.parse_args()

    if args.preflight:
        sys.exit(preflight())
    sys.exit(verify(args.results_dir, args.device, args.report))


if __name__ == "__main__":
    main()
