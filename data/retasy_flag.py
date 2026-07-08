#!/usr/bin/env python3
"""
Stage 1 of the RetaSy cleanup — auto-flag junk / mislabeled learner clips.

RetaSy (~2,235 clips) is crowd-sourced learner audio: many clips are complete silence,
mic-noise only, false starts, or recitations of a DIFFERENT ayah than labeled. Feeding
these to phase-2 training corrupts gradients (a forced transcript over noise) and the
honest learner eval counts unjudgeable clips as misses. This pass triages every clip so
a human only reviews the borderline band (data/retasy_review.py, Stage 2).

Three cheap signal tiers, all reusing in-repo machinery:
  1. Energy / VAD  — RMS + Silero speech fraction  -> silent / noise_only / too_short
  2. Decode sanity — best_s123_mic greedy phonemes vs the labeled ayah's expected length
  3. Label match   — infix cost of the decode against (a) the LABELED ayah reference and
                     (b) the whole unit index. High-vs-label + low-vs-other = likely
                     MISLABEL (recoverable: the better-matching ayah is suggested);
                     high-vs-both = garbage.

Output: data/raw/retasy_audio/flags.csv — one row per clip with bucket + the signals the
review page shows. Buckets: ok, silent, noise_only, too_short, garbage, possible_mislabel,
borderline. Also prints the distribution so we know the size of the problem before spending
human time.

  python data/retasy_flag.py                       # all clips
  python data/retasy_flag.py --limit 200           # quick pass
  python data/retasy_flag.py --checkpoint training/exp/best_s123_mic.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "research"))

RETASY_MANIFEST = REPO / "data" / "raw" / "retasy_audio" / "manifest.csv"
OUT_CSV = REPO / "data" / "raw" / "retasy_audio" / "flags.csv"
VAD_ONNX = REPO / "conformance" / "assets" / "silero_vad.onnx"
ENC_FRAME_SEC = 0.04

# thresholds (tuned to be permissive — the review page is the arbiter for the middle band)
RMS_SILENT = 0.005          # below this = effectively silence
SPEECH_FRAC_NOISE = 0.15    # non-silent but almost no VAD speech = noise/babble
MIN_SPEECH_SEC = 1.0        # too little speech to judge an ayah
LABEL_COST_OK = 0.45        # infix cost vs labeled ayah at/below this = decode supports label
MISLABEL_MARGIN = 0.15      # other-ayah beats labeled by this much -> suggest relabel


def speech_stats(vad_sess, wav: np.ndarray) -> tuple[float, float]:
    """Silero VAD -> (speech_fraction, speech_seconds). 512-sample chunks, 64-sample
    context prepend (the ported runtime contract, sdk/core/src/vad.cpp)."""
    import numpy as _np
    h = _np.zeros((2, 1, 128), dtype=_np.float32)
    ctx = _np.zeros(64, dtype=_np.float32)
    sr = _np.array(16000, dtype=_np.int64)   # 0-d scalar tensor (input shape [])
    speech_chunks = 0
    total = 0
    CH = 512
    for i in range(0, len(wav) - CH + 1, CH):
        chunk = wav[i:i + CH].astype(_np.float32)
        inp = _np.concatenate([ctx, chunk])[None, :]
        ctx = chunk[-64:]
        out, h = vad_sess.run(None, {"input": inp, "state": h, "sr": sr})
        speech_chunks += int(out[0, 0] >= 0.5)
        total += 1
    frac = speech_chunks / max(1, total)
    return frac, speech_chunks * CH / 16000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="training/exp/best_s123_mic.pt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    from data import load_wav_16k, logmel_16k, load_tokens
    from model import EmformerCTC
    from chain_sliding import build_ngram_index, _infix_norm, window_best

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(REPO / args.checkpoint, map_location=device, weights_only=False)
    model = EmformerCTC(num_tokens=ck["vocab"]).to(device).eval()
    model.load_state_dict(ck["model"])
    id2tok = {v: k for k, v in load_tokens().items()}

    # references: unit index (segments + whole ayat) for the "matches a DIFFERENT ayah" check
    ayah_ph = {k: v.split() for k, v in json.loads(
        (REPO / "data/lang/ayah_phonemes.json").read_text(encoding="utf-8")).items()}
    ref_lens = {k: len(v) for k, v in ayah_ph.items()}
    ngram_idx = build_ngram_index(ayah_ph)

    vad_sess = None
    if VAD_ONNX.exists():
        import onnxruntime as ort
        vad_sess = ort.InferenceSession(str(VAD_ONNX), providers=["CPUExecutionProvider"])
    else:
        print(f"  (no VAD at {VAD_ONNX}; speech stats disabled — run conformance/generate.py)")

    man = pd.read_csv(RETASY_MANIFEST)
    if args.limit:
        man = man.head(args.limit)
    print(f"flagging {len(man)} RetaSy clips with {args.checkpoint}")

    def greedy(wav: np.ndarray) -> list[str]:
        feats = logmel_16k(torch.from_numpy(wav)).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast(device):
            lp, ol = model(feats, torch.tensor([feats.shape[1]], device=device))
        ids = lp.float().cpu()[0, :int(ol[0])].argmax(-1).tolist()
        out, prev = [], -1
        for s in ids:
            if s != prev and s != 0:
                out.append(id2tok[s])
            prev = s
        return out

    rows = []
    for i, r in enumerate(man.itertuples()):
        key = f"{r.surah_id}:{r.ayah_id}"
        try:
            wav = load_wav_16k(r.path).numpy()
        except Exception as e:  # noqa: BLE001 — unreadable clip is itself a flag
            rows.append(dict(recording_id=r.recording_id, key=key, bucket="garbage",
                             rms=0.0, speech_frac=0.0, speech_sec=0.0, n_phon=0,
                             label_cost=1.0, best_key="", best_cost=1.0, suggest="",
                             note=f"read error: {e}"))
            continue

        rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) if len(wav) else 0.0
        sfrac, ssec = speech_stats(vad_sess, wav) if vad_sess is not None else (1.0, len(wav) / 16000)

        # tier 1: energy / VAD gates (skip the expensive decode when clearly unusable)
        if rms < RMS_SILENT:
            bucket, phon, lcost, bkey, bcost = "silent", [], 1.0, "", 1.0
        elif sfrac < SPEECH_FRAC_NOISE:
            bucket, phon, lcost, bkey, bcost = "noise_only", [], 1.0, "", 1.0
        elif ssec < MIN_SPEECH_SEC:
            bucket, phon, lcost, bkey, bcost = "too_short", [], 1.0, "", 1.0
        else:
            phon = greedy(wav)
            ref = ayah_ph.get(key, [])
            lcost = _infix_norm(ref, phon) if ref and len(phon) >= 3 else 1.0
            # best-matching ANY ayah over the full window (mislabel detection)
            bkey, bcost = window_best(phon, ngram_idx, ayah_ph, ref_lens, fire_cost=1.0)
            bkey, bcost = (bkey or ""), float(bcost)
            if lcost <= LABEL_COST_OK:
                bucket = "ok"
            elif bkey and bkey != key and bcost <= lcost - MISLABEL_MARGIN and bcost <= LABEL_COST_OK:
                bucket = "possible_mislabel"
            elif len(phon) < 3:
                bucket = "noise_only"
            else:
                bucket = "borderline" if lcost <= 0.6 else "garbage"

        suggest = bkey if bucket == "possible_mislabel" else ""
        rows.append(dict(
            recording_id=r.recording_id, key=key, bucket=bucket,
            rms=round(rms, 4), speech_frac=round(sfrac, 3), speech_sec=round(ssec, 2),
            n_phon=len(phon), label_cost=round(float(lcost), 3),
            best_key=bkey, best_cost=round(float(bcost), 3), suggest=suggest,
            final_label=getattr(r, "final_label", "") if not pd.isna(getattr(r, "final_label", np.nan)) else "",
            path=r.path, duration=r.duration, reciter_id=r.reciter_id))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(man)}", flush=True)

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")

    print(f"\nwrote {args.out.relative_to(REPO)}  ({len(out)} clips)")
    print("\nbucket distribution:")
    for b, n in out["bucket"].value_counts().items():
        print(f"  {b:20} {n:5}  ({n / len(out):5.1%})")
    # cross-check auto-flags against the pre-existing human final_label
    lab = out[out["final_label"] != ""]
    if len(lab):
        print(f"\ncross-check vs existing final_label ({len(lab)} human-labeled):")
        good = {"correct", "in_correct"}   # in_correct = mispronounced but right ayah -> keep
        auto_bad = ~out["bucket"].isin(["ok", "borderline"])
        for lv in sorted(lab["final_label"].unique()):
            sub = lab[lab["final_label"] == lv]
            flagged = (~sub["bucket"].isin(["ok", "borderline"])).mean()
            tag = "(should KEEP)" if lv in good else "(should DROP)"
            print(f"  {lv:20} {len(sub):4}  auto-flagged-bad {flagged:5.1%}  {tag}")


if __name__ == "__main__":
    main()
