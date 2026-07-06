#!/usr/bin/env python3
"""
Waqf-based ayah segmentation via CTC forced alignment (research infrastructure).

Splits ayah text at waqf pause marks (ۖ ۗ ۘ ۚ — standalone tokens in the Uthmani text),
builds per-segment phoneme references (pausal form at each segment end, matching how
reciters actually pause), force-aligns them against clip audio with our own trained
CTC model (torchaudio.functional.forced_align), and cuts the audio at segment
boundaries.

Primary v1 purpose: **human audition** of the automatic splits — exports segment WAVs
plus an audition.html page (click-through listening, per-segment Arabic text, durations,
alignment confidence) so the split quality can be judged by ear before building the
segment-level matcher experiments on top.

  python data/segment_waqf.py                    # default audition selection
  python data/segment_waqf.py --keys 2:282 2:255 --reciters 4
  python data/segment_waqf.py --min-phonemes 200 --sample 20

Output: data/raw/segment_audition/  (wavs + audition.html; gitignored via data/raw)
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA_DIR = Path(__file__).parent
REPO = DATA_DIR.parent
sys.path.insert(0, str(REPO / "training"))

from data import load_wav_16k, logmel_16k, load_tokens  # noqa: E402
from model import EmformerCTC                            # noqa: E402
from quran_g2p import g2p_word                           # noqa: E402

# Waqf marks that indicate a recommended/permitted pause -> split points.
SPLIT_MARKS = {"ۖ", "ۗ", "ۘ", "ۚ"}   # ۖ ۗ ۘ ۚ
# Non-splitting marks: ۙ (la: do NOT pause), ۛ (mu'anaqah pair: v1 skips), ۜ (sin).
OTHER_MARKS = {"ۙ", "ۛ", "ۜ"}

MIN_SEG_PHONEMES = 8      # merge tiny fragments into the previous segment
ENC_FRAME_SEC = 0.04      # encoder frame (25 fps: 10 ms mel hop x 4 subsampling)
SR = 16000


# ---------------------------------------------------------------------------
# Text -> waqf segments -> per-segment phoneme references
# ---------------------------------------------------------------------------

def waqf_segments(text: str) -> list[list[str]]:
    """Split an ayah's tokens at waqf split-marks. Returns list of word lists."""
    segs: list[list[str]] = [[]]
    for tok in text.split():
        if tok in SPLIT_MARKS:
            if segs[-1]:
                segs.append([])
        elif tok in OTHER_MARKS:
            continue
        else:
            segs[-1].append(tok)
    return [s for s in segs if s]


def segment_phonemes(segs: list[list[str]]) -> list[list[str]]:
    """G2P each segment; pausal (waqf) form on each segment's last word — reciters pause
    there. Tiny segments are merged into their predecessor."""
    out: list[list[str]] = []
    for si, words in enumerate(segs):
        ph: list[str] = []
        for wi, w in enumerate(words):
            ph.extend(g2p_word(w, ayah_initial=(si == 0 and wi == 0),
                               word_final_in_ayah=(wi == len(words) - 1)))
        out.append(ph)
    # merge fragments < MIN_SEG_PHONEMES into the previous segment
    merged_ph: list[list[str]] = []
    merged_words: list[list[str]] = []
    for ph, words in zip(out, segs):
        if merged_ph and len(ph) < MIN_SEG_PHONEMES:
            merged_ph[-1].extend(ph)
            merged_words[-1].extend(words)
        else:
            merged_ph.append(list(ph))
            merged_words.append(list(words))
    segs[:] = merged_words
    return merged_ph


# ---------------------------------------------------------------------------
# Forced alignment
# ---------------------------------------------------------------------------

def align_clip(model, device, wav: torch.Tensor, ref_ids: list[int]):
    """CTC forced alignment. Returns (token_spans, mean_score) where token_spans[i] =
    (start_frame, end_frame) of reference token i at the encoder frame rate."""
    import torchaudio.functional as F
    feats = logmel_16k(wav).unsqueeze(0).to(device)
    flen = torch.tensor([feats.shape[1]], device=device)
    with torch.no_grad():
        log_probs, out_lens = model(feats, flen)
    T = int(out_lens[0])
    lp = log_probs[:, :T].float().cpu()          # [1, T, V]
    tgt = torch.tensor([ref_ids], dtype=torch.int32)
    if T < len(ref_ids):
        raise ValueError(f"clip too short to align: {T} frames < {len(ref_ids)} tokens")
    aligned, scores = F.forced_align(lp, tgt, blank=0)
    spans = F.merge_tokens(aligned[0], scores[0])   # per emitted token: start/end/score
    if len(spans) != len(ref_ids):
        raise ValueError(f"alignment produced {len(spans)} spans for {len(ref_ids)} tokens")
    token_spans = [(s.start, s.end) for s in spans]
    return token_spans, float(np.mean([s.score for s in spans]))


def cut_points(token_spans, seg_lens: list[int]) -> list[tuple[int, int]]:
    """Segment boundaries in encoder frames: midpoint between the last phoneme of seg i
    and the first phoneme of seg i+1. Returns per-segment (start_frame, end_frame)."""
    bounds = [0]
    pos = 0
    for L in seg_lens[:-1]:
        pos += L
        prev_end = token_spans[pos - 1][1]
        next_start = token_spans[pos][0]
        bounds.append((prev_end + next_start) // 2)
    bounds.append(token_spans[-1][1] + 8)   # tail padding (~0.3 s)
    return [(bounds[i], bounds[i + 1]) for i in range(len(seg_lens))]


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="training/exp/best_s123.pt")
    ap.add_argument("--keys", nargs="*", default=None,
                    help="specific ayat (surah:ayah); default = flagship + random long sample")
    ap.add_argument("--reciters", type=int, default=4, help="reciters per flagship ayah")
    ap.add_argument("--sample", type=int, default=12, help="random long ayat (1 reciter each)")
    ap.add_argument("--min-phonemes", type=int, default=150, help="'long ayah' cutoff for the sample")
    ap.add_argument("--max-clip-sec", type=float, default=240.0, help="skip clips longer than this")
    ap.add_argument("--out", default="data/raw/segment_audition")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(REPO / args.checkpoint, map_location=device)
    model = EmformerCTC(num_tokens=ck["vocab"]).to(device).eval()
    model.load_state_dict(ck["model"])
    tok2id = load_tokens()

    text = json.loads((DATA_DIR / "manifests" / "ayah_text.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(DATA_DIR / "raw" / "audio" / "manifest.csv")

    # --- selection ---
    rng = random.Random(0)
    if args.keys:
        chosen = {k: args.reciters for k in args.keys}
    else:
        chosen = {"2:282": args.reciters, "2:255": args.reciters, "3:154": args.reciters}
        long_keys = [k for k, t in text.items()
                     if k not in chosen and len(waqf_segments(t)) >= 3
                     and sum(len(g2p_word(w)) for w in t.split() if w not in SPLIT_MARKS | OTHER_MARKS)
                         >= args.min_phonemes]
        for k in rng.sample(long_keys, min(args.sample, len(long_keys))):
            chosen[k] = 1

    all_reciters = sorted(manifest["reciter_id"].unique())
    out_dir = REPO / args.out
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)

    rows_html: list[str] = []
    seg_durs: list[float] = []
    n_clips = n_fail = 0

    for key, n_rec in chosen.items():
        t = text.get(key)
        if t is None:
            print(f"  {key}: not in corpus, skipped"); continue
        segs = waqf_segments(t)
        if len(segs) < 2:
            print(f"  {key}: no waqf splits, skipped"); continue
        seg_ph = segment_phonemes(segs)
        seg_lens = [len(p) for p in seg_ph]
        ref_ids = [tok2id[p] for ph in seg_ph for p in ph]

        cands = manifest[(manifest.surah_id == int(key.split(":")[0]))
                         & (manifest.ayah_id == int(key.split(":")[1]))]
        picks = [r for r in all_reciters if r in set(cands.reciter_id)]
        rng.shuffle(picks)
        for rec in picks[:n_rec]:
            row = cands[cands.reciter_id == rec].iloc[0]
            if float(row.get("duration", 0)) > args.max_clip_sec:
                print(f"  {key} [{rec}]: {row.duration:.0f}s > cap, skipped"); continue
            try:
                wav = load_wav_16k(row["path"])
                spans, score = align_clip(model, device, wav, ref_ids)
                frames = cut_points(spans, seg_lens)
            except Exception as e:
                print(f"  {key} [{rec}]: ALIGN FAILED — {e}"); n_fail += 1; continue

            n_clips += 1
            clip_id = f"s{int(key.split(':')[0]):03d}_a{int(key.split(':')[1]):03d}_{rec}"
            cells = []
            for si, (f0, f1) in enumerate(frames):
                s0, s1 = int(f0 * ENC_FRAME_SEC * SR), min(int(f1 * ENC_FRAME_SEC * SR), len(wav))
                seg_wav = wav[s0:s1].numpy()
                fname = f"{clip_id}_seg{si+1:02d}.wav"
                import soundfile as sf
                sf.write(out_dir / "wavs" / fname, seg_wav, SR)
                dur = (s1 - s0) / SR
                seg_durs.append(dur)
                arab = " ".join(segs[si])
                cells.append(
                    f"<tr><td>{si+1}/{len(frames)}</td>"
                    f"<td><audio controls preload='none' src='wavs/{fname}'></audio></td>"
                    f"<td>{dur:5.1f}s</td><td dir='rtl' style='font-size:22px'>{html.escape(arab)}</td></tr>")
            rows_html.append(
                f"<h3>{key} — {html.escape(rec)} ({row.duration:.0f}s, align score {score:.2f})</h3>"
                f"<table border='1' cellpadding='6' style='border-collapse:collapse'>"
                f"<tr><th>seg</th><th>audio</th><th>dur</th><th>text</th></tr>"
                + "".join(cells) + "</table>")
            print(f"  {key} [{rec}]: {len(frames)} segments, align score {score:.2f}")

    page = ("<!doctype html><meta charset='utf-8'><title>Waqf segment audition</title>"
            "<body style='font-family:sans-serif;max-width:1100px;margin:20px auto'>"
            "<h1>Waqf segment audition</h1>"
            f"<p>{n_clips} clips, {len(seg_durs)} segments | segment duration "
            f"min {min(seg_durs):.1f}s / median {np.median(seg_durs):.1f}s / max {max(seg_durs):.1f}s"
            f" | {n_fail} alignment failures</p>"
            "<p>Listen for: cut mid-word? pause audible at the end of each segment? "
            "boundary too early/late?</p>" + "".join(rows_html))
    (out_dir / "audition.html").write_text(page, encoding="utf-8")
    print(f"\n{n_clips} clips segmented ({n_fail} failures). Open:\n  {out_dir / 'audition.html'}")


if __name__ == "__main__":
    main()
