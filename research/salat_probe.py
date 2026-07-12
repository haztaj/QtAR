#!/usr/bin/env python3
"""Salat-phrase transfer probe — raw per-phrase decode inspection.

Companion to research/salat_eval.py. Where salat_eval classifies + scores, this
just shows HOW WELL the recitation-trained encoder decodes each audible prayer
phrase, cold: energy-segments the clip, greedy-decodes each utterance, and reports
the infix-normalized edit distance to the G2P phoneme reference of each phrase.
Use it to eyeball decode quality (e.g. "akbar never decodes; sami'allah is clean").

Only the three AUDIBLE phrases are included — the ruku/sujud tasbih and tashahhud
are said silently in prayer, so there is nothing to detect.

  python research/salat_probe.py
  python research/salat_probe.py --audio data/salat_phrases.m4a --ckpt training/exp/best_s123_mic.pt
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "data"))
from data import logmel_16k, load_tokens   # noqa: E402
from model import EmformerCTC              # noqa: E402
from quran_g2p import g2p_ayah             # noqa: E402

SR = 16000
PHRASES = {                                # audible prayer markers only
    "takbir":    "اللَّهُ أَكْبَرُ",
    "samiallah": "سَمِعَ اللَّهُ لِمَنْ حَمِدَهُ",
    "salam":     "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ",
}


def load_audio_16k(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() != ".wav":
        tmp = Path(tempfile.gettempdir()) / f"{path.stem}_16k.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", str(SR), str(tmp)],
                       check=True, capture_output=True)
        path = tmp
    wav, _ = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(1)
    return wav.astype(np.float32)


def infix_cost(ref, hyp):
    """Min edit distance to match ref as an (approx) contiguous substring of hyp,
    normalized by len(ref). hyp prefix/suffix free; interior extra hyp = insertions."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0
    prev = [0.0] * (m + 1)
    for i in range(1, n + 1):
        cur = [float(i)] + [0.0] * m
        for j in range(1, m + 1):
            sub = prev[j - 1] + (0.0 if ref[i - 1] == hyp[j - 1] else 1.0)
            cur[j] = min(sub, prev[j] + 1.0, cur[j - 1] + 1.0)
        prev = cur
    return min(prev) / n


def greedy(lp, length, id2tok):
    ids = lp[0, :length].argmax(-1).tolist()
    out, prev = [], -1
    for s in ids:
        if s != prev and s != 0:
            out.append(id2tok[s])
        prev = s
    return out


def segments(wav, sr, thresh_frac=0.15, min_dur=0.30, merge_gap=0.25, pad=0.10):
    fl = int(0.025 * sr); hop = int(0.010 * sr)
    rms = np.array([np.sqrt((wav[i:i+fl]**2).mean() + 1e-9) for i in range(0, len(wav)-fl, hop)])
    active = rms > thresh_frac * rms.max()
    segs, i = [], 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            segs.append([i*hop, j*hop]); i = j
        else:
            i += 1
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < merge_gap*sr:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(max(0, int(a-pad*sr)), min(len(wav), int(b+pad*sr)))
            for a, b in merged if (b-a)/sr >= min_dur]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=str(REPO / "data" / "salat_phrases.m4a"))
    ap.add_argument("--ckpt", default="training/exp/best_s123_p31.pt")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok2id = load_tokens(); id2tok = {v: k for k, v in tok2id.items()}
    ck = torch.load(REPO / args.ckpt, map_location=dev)
    model = EmformerCTC(num_tokens=ck["vocab"]).to(dev).eval(); model.load_state_dict(ck["model"])

    print(f"model: {Path(args.ckpt).name}\n\n=== phrase phoneme references (G2P) ===")
    refs = {}
    for k, txt in PHRASES.items():
        refs[k] = g2p_ayah(txt)
        print(f"  {k:10s} {' '.join(refs[k])}")

    wav = load_audio_16k(args.audio)
    segs = segments(wav, SR)
    print(f"\n=== {len(segs)} speech segments over {len(wav)/SR:.1f}s ===\n")

    @torch.no_grad()
    def decode(buf):
        rms = float(np.sqrt((buf**2).mean()) + 1e-9)
        b = np.clip(buf * (0.1 / rms), -1, 1).astype(np.float32)
        feats = logmel_16k(torch.from_numpy(np.ascontiguousarray(b))).unsqueeze(0).to(dev)
        lp, ol = model(feats, torch.tensor([feats.shape[1]], device=dev))
        return greedy(lp.cpu(), int(ol[0]), id2tok)

    for a, b in segs:
        ph = decode(wav[a:b])
        costs = {k: infix_cost(refs[k], ph) for k in PHRASES}
        best = min(costs, key=costs.get)
        mark = "OK " if costs[best] < 0.35 else ("~  " if costs[best] < 0.55 else "XX ")
        print(f"[{a/SR:5.1f}-{b/SR:5.1f}s] {mark} -> {best:10s} cost={costs[best]:.2f}"
              f"  ({' '.join(ph)})")


if __name__ == "__main__":
    main()
