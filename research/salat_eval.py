#!/usr/bin/env python3
"""Salat-marker detection eval — the no-retrain ceiling (path a).

Feasibility probe for the standalone salah-state-detection feature (see
research/CLAUDE.md and the project memory). Only three prayer phrases are
audible — takbir / sami'allah / salam — so this classifies each VAD segment into
those three (or 'none') using the CURRENT recitation-trained encoder, no
retraining, and scores against a known ground-truth marker sequence.

Pipeline:  Silero VAD  ->  Emformer+CTC greedy phonemes  ->  fuzzy-onset + duration
           3-marker discriminator  ->  edit-distance alignment vs ground truth.

Finding (2026-07-12, data/salat_phrases.m4a, 14 utterances, one voice): 11/14,
0 confusions, 0 false markers. sami'allah 4/4, salam 4/4, takbir 3/6 ("akbar"
never decodes, so takbir is detected by ABSENCE of the other cues). Same on
best_s123_p31 and best_s123_mic. MVP-viable on the current model: the sami'allah
anchor + salam end are 100% cold, and missed takbirs are recoverable via the
deterministic takbir count between anchors (zero false transitions).

  python research/salat_eval.py                                   # default clip + best_s123_p31
  python research/salat_eval.py --audio data/salat_phrases.m4a --ckpt training/exp/best_s123_mic.pt

Ground truth is per-recording; override with --truth (comma-separated) for a new clip.
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
from data import logmel_16k, load_tokens            # noqa: E402
from model import EmformerCTC                        # noqa: E402
from silero_vad import load_silero_vad, get_speech_timestamps  # noqa: E402

SR = 16000
# takbir x5, sami'allah x4, takbir x1, salam x4  (the order recited in salat_phrases.m4a)
DEFAULT_TRUTH = ["takbir"]*5 + ["samiallah"]*4 + ["takbir"]*1 + ["salam"]*4

SIB = {"s", "sh", "S", "z", "th", "f", "t"}          # sibilant-ish onset (s->t/f/sh confusions)


def load_audio_16k(path: Path) -> np.ndarray:
    """Load any audio as 16 kHz mono float32 (ffmpeg-converts non-wav; no wav committed)."""
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


def greedy(lp, length, id2tok):
    ids = lp[0, :length].argmax(-1).tolist()
    out, prev = [], -1
    for s in ids:
        if s != prev and s != 0:
            out.append(id2tok[s])
        prev = s
    return out


def first_idx(seq, pats):
    """Token index of the earliest occurrence of any pattern (contiguous), else None."""
    s = " ".join(seq)
    best = None
    for p in pats:
        k = s.find(" ".join(p))
        if k >= 0:
            ti = len(s[:k].split()) if k > 0 else 0
            best = ti if best is None else min(best, ti)
    return best


def classify(ph, dur):
    """3-marker discriminator over PARTIAL decodes. Cue confusion classes are
    principled (systematic model substitutions), not overfit to one clip:
      salam:     alaykum/rahmat tail (k~q) or sibilant salaam body (s~S~sh)
      samiallah: sam-like onset (s~t~sh) that PRECEDES 'allahu'
      takbir:    'allahu' at a glottal/vowel onset (no sibilant); detected by
                 absence of the others, since 'akbar' rarely decodes."""
    if len(ph) < 2:
        return "none"
    sib_onset = ph[0] in SIB or (len(ph) > 1 and ph[1] in SIB)
    allahu = first_idx(ph, [["l", "aa", "h"], ["l", "l", "aa"], ["a", "l", "l", "aa"]])
    alaykum = first_idx(ph, [["l", "i", "k", "u"], ["a", "y", "k", "u"], ["y", "k", "u"],
                             ["l", "i", "q", "u"], ["a", "l", "k", "u"], ["l", "k", "u", "m"],
                             ["l", "q", "u", "m"]]) is not None
    rahmat = first_idx(ph, [["r", "a", "H", "m"], ["H", "m", "a", "t"], ["r", "a", "m", "a"],
                            ["u", "r", "a", "t"], ["m", "u", "r", "a"]]) is not None
    sam = first_idx(ph, [["s", "a", "m"], ["t", "a", "m"], ["sh", "a", "m"],
                         ["t", "a", "m", "y"], ["s", "a", "m", "y"]])
    salaam_body = first_idx(ph, [["s", "a", "l", "aa"], ["sh", "a", "l", "aa"],
                                 ["S", "a", "l", "aa"], ["sh", "sh", "a", "l"]]) is not None

    if alaykum or rahmat or salaam_body:
        return "salam"
    if sam is not None and allahu is not None and sam < allahu:
        return "samiallah"
    if allahu is not None and not sib_onset:
        return "takbir"
    if allahu is not None and sib_onset:
        return "samiallah"
    return "none"


def align(truth, pred):
    """Edit-distance alignment; ops list with '-' gaps (handles over/under segmentation)."""
    n, m = len(truth), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if truth[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + c, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if truth[i - 1] == pred[j - 1] else 1):
            ops.append((truth[i - 1], pred[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append((truth[i - 1], "-")); i -= 1
        else:
            ops.append(("-", pred[j - 1])); j -= 1
    return ops[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=str(REPO / "data" / "salat_phrases.m4a"))
    ap.add_argument("--ckpt", default="training/exp/best_s123_p31.pt")
    ap.add_argument("--truth", default=None, help="comma-separated marker sequence override")
    args = ap.parse_args()

    truth = args.truth.split(",") if args.truth else DEFAULT_TRUTH
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok2id = load_tokens(); id2tok = {v: k for k, v in tok2id.items()}
    ck = torch.load(REPO / args.ckpt, map_location=dev)
    model = EmformerCTC(num_tokens=ck["vocab"]).to(dev).eval(); model.load_state_dict(ck["model"])
    vad = load_silero_vad()

    wav = load_audio_16k(args.audio)
    ts = get_speech_timestamps(torch.from_numpy(wav), vad, sampling_rate=SR,
                               min_silence_duration_ms=350, min_speech_duration_ms=250)

    @torch.no_grad()
    def decode(buf):
        rms = float(np.sqrt((buf ** 2).mean()) + 1e-9)
        b = np.clip(buf * (0.1 / rms), -1, 1).astype(np.float32)
        feats = logmel_16k(torch.from_numpy(np.ascontiguousarray(b))).unsqueeze(0).to(dev)
        lp, ol = model(feats, torch.tensor([feats.shape[1]], device=dev))
        return greedy(lp.cpu(), int(ol[0]), id2tok)

    print(f"model: {Path(args.ckpt).name}   audio: {Path(args.audio).name}   silero segments: {len(ts)}\n")
    pred = []
    for t in ts:
        a, b = t["start"], t["end"]
        ph = decode(wav[a:b])
        mk = classify(ph, (b - a) / SR)
        pred.append(mk)
        print(f"[{a/SR:5.1f}-{b/SR:5.1f}s {(b-a)/SR:3.1f}s] -> {mk:10s}  {' '.join(ph)}")

    pred_markers = [p for p in pred if p != "none"]
    print(f"\nTRUTH: {truth}")
    print(f"PRED : {pred_markers}   (+{pred.count('none')} 'none' non-emissions)")

    ops = align(truth, pred_markers)
    sub = sum(1 for t, p in ops if t != "-" and p != "-" and t != p)
    dele = sum(1 for t, p in ops if p == "-")
    ins = sum(1 for t, p in ops if t == "-")
    correct = sum(1 for t, p in ops if t == p and t != "-")
    print(f"\naligned: {correct} correct, {sub} confusions(sub), {dele} missed(del), {ins} false(ins)")
    for cls in ["takbir", "samiallah", "salam"]:
        tot = truth.count(cls)
        hit = sum(1 for t, p in ops if t == cls and p == cls)
        conf = sum(1 for t, p in ops if t == cls and p not in (cls, "-"))
        miss = sum(1 for t, p in ops if t == cls and p == "-")
        print(f"  {cls:10s} recall {hit}/{tot}   confused->{conf}  missed(none)->{miss}")


if __name__ == "__main__":
    main()
