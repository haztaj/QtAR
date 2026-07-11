"""Repetition-suppression probe — the MECHANISTIC acceptance gate for phase-3.

For a set of audio regions that follow acoustically similar content (repetitive short
surahs), decode each region twice with a given checkpoint:
  (a) STANDALONE  — region audio only (fresh context)
  (b) IN-CONTEXT  — preceded by its true preceding audio (the model's left-context memory
                    holds the repeated phrase)
and report the phoneme-count ratio (b)/(a). A suppressing model deletes most of (b)
(measured 5/16 on the live 114 take with best_s123_mic — research/CLAUDE.md 2026-07-11);
a phase-3 model should hold ratio ~1.

Cases: the two live user takes (fixed regions from the investigation) + every ayah >= 2 of
surahs 112/113/114 from the held-out continuous reciter's alignments (real wasl context).

  python research/probe_suppression.py [--checkpoint training/exp/best_s123_mic.pt]
"""
from __future__ import annotations

import argparse
import csv
import sys
import wave
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
from data import load_wav_16k, logmel_16k, load_tokens    # noqa: E402
from model import EmformerCTC                              # noqa: E402

ALIGN = REPO / "data/raw/continuous/alignments"
CONT = REPO / "data/raw/continuous"
SESS = REPO / "data/raw/audio_bench/real/sessions"
CTX_S = 12.0        # how much true preceding audio to include in (b)


def rd_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as f:
        return np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768.0


def rd_pcm(p: Path) -> np.ndarray:
    return np.fromfile(p, dtype=np.int16).astype(np.float32) / 32768.0


def cases():
    """(label, full_audio, region_t0, region_t1) — region must follow similar content."""
    out = []
    # live user takes (regions from the 2026-07-11 investigation)
    for name, spans in (("session_1783772836602", [(5.0, 10.0), (7.5, 12.5)]),   # 114 cont
                        ("session_1783772776305", [(5.0, 10.0), (10.0, 15.0)])):  # 112
        p = SESS / f"{name}.wav"
        if p.exists():
            w = rd_wav(p)
            for (a, b) in spans:
                out.append((f"user/{name[-6:]} [{a:.0f}-{b:.0f}s]", w, a, b))
    # held-out continuous reciter: ayat >= 2 of the repetitive short surahs
    for rdir in sorted(ALIGN.glob("yasser_ad_dussary")):
        for surah in (112, 113, 114):
            acsv = rdir / f"s{surah:03d}.csv"
            pcm = CONT / rdir.name / f"s{surah:03d}.pcm"
            if not (acsv.exists() and pcm.exists()):
                continue
            w = rd_pcm(pcm)
            rows = list(csv.DictReader(open(acsv, encoding="utf-8")))
            for r in rows:
                if int(r["ayah"]) < 2 or r["flag"] != "OK":
                    continue
                out.append((f"{rdir.name}/s{surah}:{r['ayah']}", w,
                            float(r["start_s"]), float(r["end_s"])))
    return out


@torch.no_grad()
def n_phonemes(model, wav: np.ndarray, count_from_s: float) -> int:
    feats = logmel_16k(torch.from_numpy(np.ascontiguousarray(wav))).unsqueeze(0)
    lp, ln = model(feats, torch.tensor([feats.shape[1]]))
    ids = lp[0, : int(ln[0])].argmax(-1).tolist()
    n, prev = 0, -1
    for f, i in enumerate(ids):
        if i != prev and i != 0 and f * 0.04 >= count_from_s:
            n += 1
        prev = i
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="training/exp/best_s123_mic.pt")
    args = ap.parse_args()
    ck = torch.load(REPO / args.checkpoint, map_location="cpu", weights_only=False)
    model = EmformerCTC(num_tokens=len(load_tokens()))
    model.load_state_dict(ck["model"])
    model.eval()

    ratios = []
    print(f"{'case':42} {'alone':>6} {'in-ctx':>6} {'ratio':>6}")
    for label, w, a, b in cases():
        sr = 16000
        alone = n_phonemes(model, w[int(a * sr): int(b * sr)], 0.0)
        c0 = max(0.0, a - CTX_S)
        inctx = n_phonemes(model, w[int(c0 * sr): int(b * sr)], a - c0)
        r = inctx / max(1, alone)
        ratios.append(r)
        print(f"{label:42} {alone:>6} {inctx:>6} {r:>6.2f}")
    print(f"\nmean in-context/alone ratio: {np.mean(ratios):.3f}  "
          f"(suppressing model ~0.3-0.6; healthy ~0.9-1.1)  n={len(ratios)}")


if __name__ == "__main__":
    main()
