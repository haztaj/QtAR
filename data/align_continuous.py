"""Align the CONTINUOUS per-surah recitations (data/raw/continuous/<reciter>/sNNN.mp3) to
per-ayah time boundaries — the labeling step for phase-3 concatenation training.

Files run minutes to HOURS (Baqarah up to 4.4 h), far beyond the model's 30 s training
window and beyond a single forced-align DP, so the pipeline is hierarchical:

  1. CHUNKED full-file log-probs: 28 s windows, 4 s overlap, keep each window's settled
     interior — the Emformer masks padding, so interior frames match a whole-file forward.
  2. ROUGH ayah boundaries: greedy phoneme decode -> banded edit DP against the surah's
     concatenated per-ayah refs (free skip at start/end absorbs isti'adha + basmala, which
     the per-ayah refs don't contain). Boundary = decode position of each ayah's ref edge.
  3. REFINE per ayah: torchaudio forced_align on the log-prob slice (+-2 s slack; CTC
     absorbs slack as blanks) vs the ayah's phonemes -> frame-exact bounds + a score.

Outputs (gitignored like all raw data):
  data/raw/continuous/alignments/<reciter>/sNNN.csv   ayah,start_s,end_s,score,flag
  data/raw/continuous/alignments/report.csv           per-file coverage/QA rollup

Notes: refs are pausal (waqf) forms; fully-connected (wasl) recitation mismatches the last
vowel at some boundaries — absorbed by the DP band and noted via the score, and CTC training
tolerates it. Run: python data/align_continuous.py [--reciter X] [--surah N] [--device cuda]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
from data import load_wav_16k, logmel_16k, load_tokens        # noqa: E402
from model import EmformerCTC                                  # noqa: E402

ROOT = REPO / "data/raw/continuous"
OUT = ROOT / "alignments"
CKPT = REPO / "training/exp/best_s123_mic.pt"   # deployment champion (taint audit 2026-07-11)
SR = 16000
FRAME = 0.04                                     # model output frame (s)
WIN_S, OVL_S = 28.0, 4.0                         # chunked forward: window / overlap
BAND_FRAC = 0.12                                 # DP band half-width as a fraction of len
SCORE_FLAG = -1.5                                # mean per-frame forced-align logprob below -> flag


def load_model(device):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    tokens = load_tokens()
    m = EmformerCTC(num_tokens=len(tokens))
    m.load_state_dict(ck["model"])
    m.eval().to(device)
    return m, tokens


def surah_ref(surah: int, ayah_ph: dict, tok2id: dict):
    """Concatenated per-ayah phoneme ids + the ref index where each ayah starts."""
    ids, starts, ayat = [], [], []
    a = 1
    while f"{surah}:{a}" in ayah_ph:
        starts.append(len(ids))
        ayat.append(a)
        ids.extend(tok2id[p] for p in ayah_ph[f"{surah}:{a}"].split())
        a += 1
    return ids, starts, ayat


@torch.no_grad()
def full_logprobs(wav: torch.Tensor, model, device):
    """Chunked whole-file log-probs [T,V] with settled-interior stitching."""
    hop_s = WIN_S - OVL_S
    n = wav.numel()
    pieces = []
    t0 = 0.0
    while t0 * SR < n:
        seg = wav[int(t0 * SR): int(min(t0 + WIN_S, n / SR) * SR)]
        if seg.numel() < SR:                    # trailing sliver: pad to 1 s
            seg = torch.nn.functional.pad(seg, (0, SR - seg.numel()))
        feats = logmel_16k(seg).unsqueeze(0).to(device)
        lp, ln = model(feats, torch.tensor([feats.shape[1]], device=device))
        lp = lp[0, : int(ln[0])].float().cpu()  # [Tw, V]
        # settled interior, contiguous by construction: each non-last piece spans exactly
        # [t0 + ovl/2, t0 + hop + ovl/2) — the conv subsampling shortens Tout a few frames
        # below nominal, so trimming symmetrically off Tout leaves seam holes.
        lead = 0 if t0 == 0.0 else int((OVL_S / 2) / FRAME)
        is_last = (t0 + WIN_S) * SR >= n
        tail = lp.shape[0] if is_last else int((hop_s + OVL_S / 2) / FRAME)
        pieces.append((int(round(t0 / FRAME)) + lead, lp[lead:tail]))
        if is_last:
            break
        t0 += hop_s
    T = pieces[-1][0] + pieces[-1][1].shape[0]
    out = torch.full((T, pieces[0][1].shape[1]), float("nan"))
    for off, piece in pieces:
        out[off: off + piece.shape[0]] = piece
    assert not torch.isnan(out).any(), "stitch left holes"
    return out


def greedy(lp: torch.Tensor):
    """Collapsed phoneme ids + their frame indices."""
    ids = lp.argmax(-1).tolist()
    ph, fr, prev = [], [], -1
    for f, i in enumerate(ids):
        if i != prev and i != 0:
            ph.append(i)
            fr.append(f)
        prev = i
    return ph, fr


def banded_boundaries(ref: list[int], starts: list[int], dec: list[int]):
    """Monotonic banded edit DP ref->decode with FREE leading/trailing skip on the decode
    side (isti'adha/basmala + trailing content). Returns each ayah-start ref index's mapped
    decode position (int) via backtrace."""
    L, N = len(ref), len(dec)
    band = max(200, int(BAND_FRAC * max(L, N)))
    INF = 1e9
    # dp rows over ref index i; columns are a moving decode band
    lo_of = lambda i: max(0, min(N - 1, int(i * N / L) - band))
    hi_of = lambda i: min(N, int(i * N / L) + band)
    prev_lo, W = lo_of(0), hi_of(0) - lo_of(0)
    prev = np.zeros(hi_of(0) - lo_of(0) + 1)          # row i=0: free skip -> all zero
    bp = []                                            # backpointers: 0=diag,1=up(del),2=left(ins)
    rows_lo = [prev_lo]
    for i in range(1, L + 1):
        lo, hi = lo_of(i), hi_of(i)
        cur = np.full(hi - lo + 1, INF)
        bpr = np.zeros(hi - lo + 1, dtype=np.int8)
        r = ref[i - 1]
        # vectorized over the band
        js = np.arange(lo, hi + 1)
        # from prev row (index shift prev_lo)
        def prow(j):                                   # prev[j - prev_lo] with bounds
            idx = j - prev_lo
            v = np.full(j.shape, INF)
            ok = (idx >= 0) & (idx < prev.shape[0])
            v[ok] = prev[idx[ok]]
            return v
        diag = prow(js - 1) + (np.array([dec[j - 1] if 0 < j <= N else -1 for j in js]) != r)
        up = prow(js) + 1.0                            # ref consumed, decode not (deletion)
        cur, bpr = np.minimum(diag, up), np.where(diag <= up, 0, 1).astype(np.int8)
        # left (insertion) needs a sequential pass within the row
        for k in range(1, cur.shape[0]):
            if cur[k - 1] + 1.0 < cur[k]:
                cur[k] = cur[k - 1] + 1.0
                bpr[k] = 2
        prev, prev_lo = cur, lo
        bp.append(bpr)
        rows_lo.append(lo)
    # free trailing skip: best end column in the last row
    j = int(np.argmin(prev)) + rows_lo[-1]
    # backtrace, recording decode position at each ayah-start ref index
    want = {s: None for s in starts}
    i = L
    while i > 0:
        row = bp[i - 1]
        k = j - rows_lo[i]
        k = max(0, min(row.shape[0] - 1, k))
        move = row[k]
        if i in want and want[i] is None:
            want[i] = j
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i = i - 1
        else:
            j = j - 1
    want[0] = j if 0 in want else want.get(0)
    return {s: (want[s] if want[s] is not None else 0) for s in starts}


@torch.no_grad()
def refine(lp: torch.Tensor, tgt_ids: list[int], f0: int, f1: int, slack_f: int = 50):
    """forced_align on lp[f0-slack : f1+slack] vs tgt -> exact frame bounds + mean score."""
    a, b = max(0, f0 - slack_f), min(lp.shape[0], f1 + slack_f)
    seg = lp[a:b].unsqueeze(0)
    tgt = torch.tensor([tgt_ids], dtype=torch.int32)
    try:
        aligned, scores = F.forced_align(seg, tgt, blank=0)
    except Exception:
        return None
    al = aligned[0].tolist()
    nz = [k for k, v in enumerate(al) if v != 0]
    if not nz:
        return None
    return a + nz[0], a + nz[-1] + 1, float(scores[0].mean())


def align_file(mp3: Path, surah: int, model, tokens, ayah_ph, device):
    ref, starts, ayat = surah_ref(surah, ayah_ph, tokens)
    wav = load_wav_16k(str(mp3))
    lp = full_logprobs(wav, model, device)
    dec, fr = greedy(lp)
    bounds = banded_boundaries(ref, starts, dec)
    rows = []
    for k, a in enumerate(ayat):
        d0 = bounds[starts[k]]
        d1 = bounds[starts[k + 1]] if k + 1 < len(starts) else len(dec)
        f0 = fr[min(d0, len(fr) - 1)]
        f1 = fr[min(max(d1 - 1, 0), len(fr) - 1)] + 1
        seg_ids = ref[starts[k]: starts[k + 1] if k + 1 < len(starts) else len(ref)]
        r = refine(lp, seg_ids, f0, f1)
        if r is None:
            rows.append((a, f0 * FRAME, f1 * FRAME, float("nan"), "ROUGH_ONLY"))
            continue
        rf0, rf1, score = r
        flag = "OK" if score >= SCORE_FLAG else "LOW_SCORE"
        rows.append((a, rf0 * FRAME, rf1 * FRAME, score, flag))
    return rows, len(wav) / SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reciter", default=None)
    ap.add_argument("--surah", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, tokens = load_model(args.device)
    ayah_ph = json.loads((REPO / "data/lang/ayah_phonemes.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for rdir in sorted(d for d in ROOT.iterdir() if d.is_dir() and d.name not in ("sources", "alignments")):
        if args.reciter and rdir.name != args.reciter:
            continue
        (OUT / rdir.name).mkdir(exist_ok=True)
        for mp3 in sorted(rdir.glob("s*.mp3")):
            surah = int(mp3.stem[1:])
            if args.surah and surah != args.surah:
                continue
            out_csv = OUT / rdir.name / f"{mp3.stem}.csv"
            if out_csv.exists():
                continue
            rows, dur = align_file(mp3, surah, model, tokens, ayah_ph, args.device)
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ayah", "start_s", "end_s", "score", "flag"])
                w.writerows(rows)
            ok = sum(1 for r in rows if r[4] == "OK")
            cov = (rows[-1][2] - rows[0][1]) / dur if rows else 0.0
            print(f"{rdir.name}/{mp3.stem}: {ok}/{len(rows)} OK, span {cov:.0%} of {dur/60:.1f} min",
                  flush=True)
            report.append([rdir.name, mp3.stem, len(rows), ok, f"{cov:.3f}", f"{dur:.0f}"])
    with open(OUT / "report.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(report)


if __name__ == "__main__":
    main()
