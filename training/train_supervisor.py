#!/usr/bin/env python3
"""
Run train.py in FRESH PROCESSES a few epochs at a time.

Why: the augmentation library (audiomentations) leaks RAM across epochs — and not
only in the DataLoader workers (already respawned per epoch): the MAIN process
accumulates ~8.5 GB/epoch (measured 30.7 GB private after ~3.5 epochs on the
mic-adaptation run), evicting the OS file cache and then paging, so each epoch
runs slower than the last until the machine crawls. A process restart resets it.
This supervisor bounds the leak by construction: train.py exits with code 75
after --epochs-per-run epochs (full state in last<tag>.pt), and is relaunched
with --resume until it finishes (exit 0).

  python training/train_supervisor.py --epochs 27 --tag _s123_mic \
      --train-manifest data/raw/phase2/combined_train.csv \
      -- --lr 1.6e-4 --augment --frame-budget 24000 --num-workers 10

Everything after `--` is passed through to train.py verbatim.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "training" / "exp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, required=True, help="TOTAL epochs")
    ap.add_argument("--tag", default="", help="checkpoint tag (last<tag>.pt drives resume)")
    ap.add_argument("--train-manifest", default=None)
    ap.add_argument("--epochs-per-run", type=int, default=2)
    ap.add_argument("--init-from", default=None,
                    help="warm-start for the FIRST run if no resumable last<tag>.pt exists")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="args after -- are passed to train.py")
    args = ap.parse_args()
    passthru = [a for a in args.rest if a != "--"]

    last = EXP / f"last{args.tag}.pt"

    run = 0
    while True:
        run += 1
        cmd = [sys.executable, str(REPO / "training" / "train.py"),
               "--epochs", str(args.epochs), "--tag", args.tag,
               "--epochs-per-run", str(args.epochs_per_run)] + passthru
        if args.train_manifest:
            cmd += ["--train-manifest", args.train_manifest]
        resumable = False
        if last.exists():
            import torch
            resumable = "opt" in torch.load(last, map_location="cpu", weights_only=False)
        if resumable:
            cmd += ["--resume", str(last.relative_to(REPO))]
        elif args.init_from:
            cmd += ["--init-from", args.init_from]
        print(f"[supervisor] run {run}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=REPO).returncode
        if rc == 75:
            continue                     # chunk finished -> fresh process resumes
        if rc == 0:
            print("[supervisor] training complete")
            return
        print(f"[supervisor] train.py exited {rc} — stopping (inspect the log)")
        sys.exit(rc)


if __name__ == "__main__":
    main()
