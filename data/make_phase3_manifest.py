"""Build the phase-3 MIXED training manifest: the phase-2 combined manifest (per-ayah
clips + cleaned RetaSy) UNION the continuous-corpus multi-ayah windows.

  python data/make_phase3_manifest.py [--win-frac 1.0]

-> data/raw/phase3/combined_train.csv  (heterogeneous rows; AyahDataset handles both kinds)

The per-ayah rows keep the base task sharp; the window rows teach continuous multi-ayah
decoding (repetition emission across ayah boundaries — the phase-3 objective). --win-frac
subsamples windows if the mix needs rebalancing (windows are ~overlapping, so 1.0 is fine
as a starting point; rebalance only on gate evidence).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PHASE2 = REPO / "data/raw/phase2/combined_train.csv"
WINDOWS = REPO / "data/raw/phase3/windows_train.csv"
OUT = REPO / "data/raw/phase3/combined_train.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win-frac", type=float, default=1.0)
    args = ap.parse_args()

    base = pd.read_csv(PHASE2)
    win = pd.read_csv(WINDOWS)
    if args.win_frac < 1.0:
        win = win.sample(frac=args.win_frac, random_state=0).reset_index(drop=True)

    # Normalize window rows to the manifest schema (ints — float NaN-tainted ids would
    # break the "surah:ayah" label keys of the per-ayah rows).
    win_rows = pd.DataFrame({
        "recording_id": [f"win_{Path(p).parent.name}_{s}_{a}_{i}"
                         for i, (p, s, a) in enumerate(zip(win["path"], win["surah"], win["ayah_from"]))],
        "reciter_id": [f"cont_{Path(p).parent.name}" for p in win["path"]],
        "surah_id": win["surah"].astype(int),
        "ayah_id": win["ayah_from"].astype(int),
        "path": win["path"],
        "duration": win["duration"],
        "phonemes": win["phonemes"],
        "start_s": win["start_s"],
        "end_s": win["end_s"],
    })
    out = pd.concat([base, win_rows], ignore_index=True)
    out["surah_id"] = out["surah_id"].astype(int)
    out["ayah_id"] = out["ayah_id"].astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"{OUT}: {len(base)} per-ayah + {len(win_rows)} window rows "
          f"({win_rows.duration.sum()/3600:.2f} h windows)")


if __name__ == "__main__":
    main()
