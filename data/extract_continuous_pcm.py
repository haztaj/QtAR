"""Decode each continuous surah MP3 ONCE to raw PCM (int16 mono 16 kHz, .pcm sibling) so
phase-3 window rows can be sliced via np.memmap in the Dataset — decoding a 2.3 h MP3 per
training item would dominate the epoch. Idempotent; ~5 GB for the 45 h corpus.

  python data/extract_continuous_pcm.py [--reciter X]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
from data import load_wav_16k    # noqa: E402

ROOT = REPO / "data/raw/continuous"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reciter", default=None)
    args = ap.parse_args()
    total = 0
    for rdir in sorted(d for d in ROOT.iterdir()
                       if d.is_dir() and d.name not in ("sources", "alignments")):
        if args.reciter and rdir.name != args.reciter:
            continue
        for mp3 in sorted(rdir.glob("s*.mp3")):
            pcm = mp3.with_suffix(".pcm")
            if pcm.exists() and pcm.stat().st_size > 0:
                continue
            wav = load_wav_16k(str(mp3)).numpy()
            (np.clip(wav, -1, 1) * 32767).astype(np.int16).tofile(pcm)
            total += 1
            print(f"{rdir.name}/{mp3.stem}: {wav.shape[0]/16000/60:.1f} min -> {pcm.name}",
                  flush=True)
    print(f"extracted {total} files")


if __name__ == "__main__":
    main()
