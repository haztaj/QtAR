#!/usr/bin/env python3
"""
Full-corpus waqf segmentation: references + per-clip aligned segment spans.

For every corpus ayah with >= 2 waqf segments (see segment_waqf.py; audition-approved
2026-07-06), this produces:

  data/lang/segment_phonemes.json     "s:a#NN" -> {phonemes, text, n_segments}
      Per-segment phoneme references (pausal form at segment ends). The segment-level
      matcher experiments build their tries from this.

  data/raw/segments/segment_spans.csv recording_id, key, seg_idx, n_segments,
                                      start_sample, end_sample, score
      Per-clip aligned segment boundaries (16 kHz sample offsets) via CTC forced
      alignment with our own model. Enables on-the-fly segment-clip training and
      segment-level eval WITHOUT writing thousands of wav files.

Idempotent: already-aligned recording_ids are skipped on re-run.

  python data/build_segments.py                      # full corpus
  python data/build_segments.py --checkpoint training/exp/best_s123.pt --max-clip-sec 420
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA_DIR = Path(__file__).parent
REPO = DATA_DIR.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(DATA_DIR))

from data import load_wav_16k, logmel_16k, load_tokens   # noqa: E402
from model import EmformerCTC                             # noqa: E402
from segment_waqf import waqf_segments, segment_phonemes, ENC_FRAME_SEC, SR  # noqa: E402

# Batching guards (forward-only, but the same WDDM paging cliff applies — see training/data.py).
LINEAR_BUDGET = 24000     # sum-of-max mel frames per batch
QUAD_BUDGET = 1.0e8       # batch * max_mel_frames^2
TAIL_PAD_FRAMES = 8       # ~0.3 s tail after the last phoneme of each segment


def align_batch(model, device, wavs: list[torch.Tensor], ref_ids: list[int]):
    """Batched forward + per-clip forced alignment. Returns list of (token_spans, score)
    (None where alignment failed)."""
    import torchaudio.functional as F
    feats = [logmel_16k(w) for w in wavs]
    T = max(f.shape[0] for f in feats)
    batch = torch.zeros(len(feats), T, feats[0].shape[1])
    lens = torch.zeros(len(feats), dtype=torch.long)
    for i, f in enumerate(feats):
        batch[i, :f.shape[0]] = f
        lens[i] = f.shape[0]
    with torch.no_grad():
        # fp16 autocast: a ~390 s clip's attention is ~2.4 GB/layer in fp32 — enough to
        # exceed VRAM (with desktop ambient) and freeze/kill the process on 2:282.
        if device == "cuda":
            torch.cuda.empty_cache()
            with torch.amp.autocast("cuda"):
                log_probs, out_lens = model(batch.to(device), lens.to(device))
        else:
            log_probs, out_lens = model(batch.to(device), lens.to(device))
    log_probs = log_probs.float().cpu()
    out = []
    tgt = torch.tensor([ref_ids], dtype=torch.int32)
    for i in range(len(feats)):
        Ti = int(out_lens[i])
        if Ti < len(ref_ids):
            out.append(None)
            continue
        try:
            aligned, scores = F.forced_align(log_probs[i:i+1, :Ti], tgt, blank=0)
            spans = F.merge_tokens(aligned[0], scores[0])
            if len(spans) != len(ref_ids):
                out.append(None)
                continue
            out.append(([(s.start, s.end) for s in spans],
                        float(np.mean([s.score for s in spans]))))
        except Exception:
            out.append(None)
    return out


def seg_bounds(token_spans, seg_lens: list[int], n_samples: int) -> list[tuple[int, int]]:
    """Per-segment (start_sample, end_sample): midpoint cuts between adjacent segments."""
    bounds = [0]
    pos = 0
    for L in seg_lens[:-1]:
        pos += L
        bounds.append((token_spans[pos - 1][1] + token_spans[pos][0]) // 2)
    bounds.append(token_spans[-1][1] + TAIL_PAD_FRAMES)
    spans = []
    for i in range(len(seg_lens)):
        s0 = int(bounds[i] * ENC_FRAME_SEC * SR)
        s1 = min(int(bounds[i + 1] * ENC_FRAME_SEC * SR), n_samples)
        spans.append((s0, s1))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="training/exp/best_s123.pt")
    ap.add_argument("--max-clip-sec", type=float, default=420.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(REPO / args.checkpoint, map_location=device)
    model = EmformerCTC(num_tokens=ck["vocab"]).to(device).eval()
    model.load_state_dict(ck["model"])
    tok2id = load_tokens()

    text = json.loads((DATA_DIR / "manifests" / "ayah_text.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(DATA_DIR / "raw" / "audio" / "manifest.csv")

    # --- segment references over the whole corpus ---
    refs: dict[str, dict] = {}
    seg_info: dict[str, tuple[list[list[str]], list[int]]] = {}   # key -> (seg phoneme lists, lens)
    for key, t in text.items():
        segs = waqf_segments(t)
        if len(segs) < 2:
            continue
        ph = segment_phonemes(segs)          # NOTE: also merges tiny fragments into `segs`
        seg_info[key] = (ph, [len(p) for p in ph])
        for i, (p, words) in enumerate(zip(ph, segs), 1):
            refs[f"{key}#{i:02d}"] = {"phonemes": " ".join(p), "text": " ".join(words),
                                      "n_segments": len(ph)}
    lang = DATA_DIR / "lang" / "segment_phonemes.json"
    lang.write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"segment references: {len(refs)} segments over {len(seg_info)} ayat -> {lang}")

    # --- alignment over every clip of every segmented ayah ---
    out_dir = DATA_DIR / "raw" / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    spans_csv = out_dir / "segment_spans.csv"
    done: set[str] = set()
    if spans_csv.exists():
        done = set(pd.read_csv(spans_csv)["recording_id"].unique())
        print(f"resuming — {len(done)} recordings already aligned")

    rows: list[dict] = []
    n_clip = n_fail = n_skip = 0

    def flush():
        if not rows:
            return
        df = pd.DataFrame(rows)
        header = not spans_csv.exists()
        df.to_csv(spans_csv, mode="a", header=header, index=False)
        rows.clear()

    keys = sorted(seg_info, key=lambda k: (int(k.split(":")[0]), int(k.split(":")[1])))
    for ki, key in enumerate(keys):
        ph, seg_lens = seg_info[key]
        ref_ids = [tok2id[p] for seg in ph for p in seg]
        sid, aid = (int(x) for x in key.split(":"))
        clips = manifest[(manifest.surah_id == sid) & (manifest.ayah_id == aid)]
        clips = clips[~clips.recording_id.isin(done)]
        clips = clips.sort_values("duration")
        pend = clips.to_dict("records")

        i = 0
        while i < len(pend):
            # greedy length-bucketed batch under linear + quadratic budgets
            batch = [pend[i]]; i += 1
            while i < len(pend):
                mx = max(r["duration"] for r in batch + [pend[i]]) * 100
                if (len(batch) + 1) * mx > LINEAR_BUDGET or (len(batch) + 1) * mx * mx > QUAD_BUDGET:
                    break
                batch.append(pend[i]); i += 1
            if batch[0]["duration"] > args.max_clip_sec:
                n_skip += len(batch)
                continue
            wavs = [load_wav_16k(r["path"]) for r in batch]
            results = align_batch(model, device, wavs, ref_ids)
            for r, wav, res in zip(batch, wavs, results):
                n_clip += 1
                if res is None:
                    n_fail += 1
                    continue
                token_spans, score = res
                for si, (s0, s1) in enumerate(seg_bounds(token_spans, seg_lens, len(wav)), 1):
                    rows.append({"recording_id": r["recording_id"], "key": key, "seg_idx": si,
                                 "n_segments": len(seg_lens), "start_sample": s0,
                                 "end_sample": s1, "score": round(score, 3)})
        if (ki + 1) % 20 == 0 or ki == len(keys) - 1:
            flush()
            if device == "cuda":
                torch.cuda.empty_cache()   # return the pool after long-clip peaks (2:282 etc.)
            print(f"  [{ki+1}/{len(keys)}] {key} | clips {n_clip} | fail {n_fail} | skip {n_skip}",
                  flush=True)

    flush()
    total = len(pd.read_csv(spans_csv)) if spans_csv.exists() else 0
    print(f"\ndone. {n_clip} clips aligned this run ({n_fail} failures, {n_skip} skipped >cap). "
          f"spans rows total: {total} -> {spans_csv}")


if __name__ == "__main__":
    main()
