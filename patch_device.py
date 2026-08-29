#!/usr/bin/env python3
"""
patch_device.py — normalise the XGBoost device flags across TheKeyMotif scripts.

The eight model scripts are NOT consistent with each other:

  0, 2, 3, 4, 5, 6a, 6b :  tree_method="gpu_hist", predictor="gpu_predictor", device="cuda"
  1                     :  tree_method="gpu_hist", predictor="gpu_predictor", max_bin=256
                           ^ no device= at all

so a blind sed leaves script 1 in a different state from the rest. This does it
properly: rewrites `gpu_hist` -> `hist`, drops the dead `predictor=` and
`use_label_encoder=` args, and guarantees exactly one `device=` per classifier.

Usage:
    python patch_device.py --device cuda          # GPU (what you want)
    python patch_device.py --device cpu           # CPU fallback
    python patch_device.py --device cuda --max-bin 128   # shrink VRAM use
    python patch_device.py --restore              # undo, from .bak files

Backups are written to <script>.bak on first run.
"""

import argparse
import glob
import os
import re
import shutil

SCRIPTS = sorted(
    f for f in glob.glob("[0-9]*.py")
    if re.match(r"^\d[a-z]?_", os.path.basename(f))
)


def patch_file(path, device, max_bin):
    with open(path, "r", newline="") as fh:
        src = fh.read()

    if "gpu_hist" not in src and "tree_method" not in src:
        return None

    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)

    # Preserve the file's line endings (repo mixes CRLF and LF).
    crlf = "\r\n" in src

    # 1. gpu_hist -> hist
    src = re.sub(r'tree_method\s*=\s*["\']gpu_hist["\']', 'tree_method="hist"', src)

    # 2. Drop dead args (removed/ignored in XGBoost 2.x). Handles trailing
    #    comma-newline-indent so we don't leave dangling commas.
    src = re.sub(r'\s*predictor\s*=\s*["\']gpu_predictor["\']\s*,?', "", src)
    src = re.sub(r'\s*use_label_encoder\s*=\s*(True|False)\s*,?', "", src)

    # 3. Normalise device=. Remove any existing, then attach to tree_method.
    src = re.sub(r'\s*device\s*=\s*["\'](cuda|cpu)["\']\s*,?', "", src)
    src = src.replace('tree_method="hist"',
                      f'tree_method="hist", device="{device}"')

    # 4. Optional max_bin, for squeezing script 1 into 4 GB.
    if max_bin is not None:
        src = re.sub(r'\s*max_bin\s*=\s*\d+\s*,?', "", src)
        src = src.replace(f'tree_method="hist", device="{device}"',
                          f'tree_method="hist", device="{device}", max_bin={max_bin}')

    # 5. Tidy: collapse ",," and ",)" left by the deletions above.
    src = re.sub(r",(\s*),", r",\1", src)
    src = re.sub(r",(\s*)\)", r"\1)", src)

    if crlf:
        src = src.replace("\r\n", "\n").replace("\n", "\r\n")

    with open(path, "w", newline="") as fh:
        fh.write(src)
    return path


def restore():
    n = 0
    for bak in glob.glob("*.py.bak"):
        shutil.copy2(bak, bak[:-4])
        n += 1
    print(f"Restored {n} file(s) from .bak")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--max-bin", type=int, default=None,
                    help="Set max_bin on every classifier. 128 or 64 roughly "
                         "halves/quarters GPU memory for the histogram build.")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    if args.restore:
        restore()
        return

    if not SCRIPTS:
        raise SystemExit("No model scripts found — run this from the repo root.")

    done = [p for p in (patch_file(f, args.device, args.max_bin) for f in SCRIPTS) if p]
    print(f"Patched {len(done)} script(s) to device={args.device}"
          + (f", max_bin={args.max_bin}" if args.max_bin else ""))
    for p in done:
        print("  ", p)
    print("\nVerify with:  grep -n 'tree_method\\|device=\\|max_bin' *.py")


if __name__ == "__main__":
    main()
