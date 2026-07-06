#!/usr/bin/env python3
"""
Ablation: whole-ayah index (A) vs waqf-segment index (B) as the detection unit.

Evaluated on segment-cut clips of the TEST reciters (never trained on), cut on the fly
from the aligned spans (data/raw/segments/segment_spans.csv). Streams are decoded once
(greedy CTC with per-phoneme frame times -> TTD in seconds) and cached; the matcher
arms then run CPU-only.

  Arm A: PhonemeTrie over ayah_phonemes.json          (1,057 whole-ayah units)
  Arm B: PhonemeTrie over segment_phonemes.json       (1,029 segments)
         + the unsegmented ayat as single units       (712 units) -> realistic index

Report splits seg_idx==1 (ayah-start audio: both arms should work) from seg_idx>=2
(mid-ayah cold start: the pause-resume case — arm A is structurally blind here).

  python research/segment_ablation.py
  python research/segment_ablation.py --limit 500          # quick pass
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "matcher"))

TEST_RECITERS = {"warsh_husary", "warsh_yassin", "yasser_ad_dussary"}
ENC_FRAME_SEC = 0.04
SR = 16000
LINEAR_BUDGET = 24000
QUAD_BUDGET = 1.0e8
CACHE = REPO / "data" / "raw" / "segments" / "test_streams.pkl"


# ---------------------------------------------------------------------------
# Stream cache: decode segment-cut test clips once (greedy + frame times)
# ---------------------------------------------------------------------------

def build_cache() -> list[dict]:
    from data import load_wav_16k, logmel_16k, load_tokens
    from model import EmformerCTC

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(REPO / "training/exp/best_s123.pt", map_location=device)
    model = EmformerCTC(num_tokens=ck["vocab"]).to(device).eval()
    model.load_state_dict(ck["model"])
    id2tok = {v: k for k, v in load_tokens().items()}

    manifest = pd.read_csv(REPO / "data/raw/audio/manifest.csv")
    manifest = manifest[manifest.reciter_id.isin(TEST_RECITERS)]
    spans = pd.read_csv(REPO / "data/raw/segments/segment_spans.csv")
    spans = spans.merge(manifest[["recording_id", "path"]], on="recording_id", how="inner")
    spans["dur"] = (spans.end_sample - spans.start_sample) / SR
    print(f"test-reciter segment clips: {len(spans)}")

    def greedy_with_times(lp: torch.Tensor, T: int):
        ids = lp[:T].argmax(-1).tolist()
        phons, times, prev = [], [], -1
        for f, s in enumerate(ids):
            if s != prev and s != 0:
                phons.append(id2tok[s])
                times.append(f * ENC_FRAME_SEC)
            prev = s
        return phons, times

    # group by recording so each source wav is loaded once
    out: list[dict] = []
    by_rec = list(spans.groupby("recording_id"))
    batch_feats, batch_meta = [], []

    def flush_batch():
        nonlocal batch_feats, batch_meta
        if not batch_feats:
            return
        from torch.nn.utils.rnn import pad_sequence
        lens = torch.tensor([f.shape[0] for f in batch_feats])
        feats = pad_sequence(batch_feats, batch_first=True)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
            lp, ol = model(feats.to(device), lens.to(device))
        lp = lp.float().cpu()
        for i, meta in enumerate(batch_meta):
            phons, times = greedy_with_times(lp[i], int(ol[i]))
            meta["phonemes"] = phons
            meta["times"] = times
            out.append(meta)
        batch_feats, batch_meta = [], []

    from data import load_wav_16k, logmel_16k  # noqa: F811
    for ri, (rec_id, gr) in enumerate(by_rec):
        wav = load_wav_16k(gr.iloc[0]["path"])
        for r in gr.itertuples():
            seg_wav = wav[r.start_sample:r.end_sample]
            if len(seg_wav) < SR // 4:
                continue
            feats = logmel_16k(seg_wav)
            # batch under the same linear+quad budgets (paging cliff)
            cur_max = max([f.shape[0] for f in batch_feats] + [feats.shape[0]])
            if batch_feats and ((len(batch_feats) + 1) * cur_max > LINEAR_BUDGET
                                or (len(batch_feats) + 1) * cur_max * cur_max > QUAD_BUDGET):
                flush_batch()
            batch_feats.append(feats)
            batch_meta.append({"recording_id": rec_id, "key": r.key, "seg_idx": int(r.seg_idx),
                               "n_segments": int(r.n_segments), "dur": float(r.dur)})
        if (ri + 1) % 200 == 0:
            flush_batch()
            print(f"  decoded {ri+1}/{len(by_rec)} recordings ({len(out)} segment streams)", flush=True)
    flush_batch()
    CACHE.write_bytes(pickle.dumps(out))
    print(f"cached {len(out)} streams -> {CACHE}")
    return out


# ---------------------------------------------------------------------------
# Matcher arms
# ---------------------------------------------------------------------------

def run_arm(streams, refs: dict[str, list[str]], truth_of, limit=0):
    """Match each stream against a trie of `refs`. truth_of(meta) -> the true unit key.
    Returns per-stream dicts with rank-1 unit, correctness, and TTD (s)."""
    from phoneme_matcher import PhonemeTrie, PhonemeMatcher
    trie = PhonemeTrie.from_ayah_phonemes(refs)
    results = []
    for meta in (streams[:limit] if limit else streams):
        true_key = truth_of(meta)
        if true_key is None:
            continue
        m = PhonemeMatcher(trie, allow_restart=False)
        ttd = None
        for i, p in enumerate(meta["phonemes"]):
            cands = m.step(p)
            if ttd is None and cands and cands[0].key == true_key:
                ttd = meta["times"][i]
        cands = m.candidates(k=3)
        keys = [c.key for c in cands]
        results.append({
            "seg_idx": meta["seg_idx"], "key": meta["key"], "dur": meta["dur"],
            "top1": bool(keys) and keys[0] == true_key,
            "top1_parent": bool(keys) and keys[0].split("#")[0] == true_key.split("#")[0],
            "top3_parent": any(k.split("#")[0] == true_key.split("#")[0] for k in keys[:3]),
            "ttd": ttd,
        })
    return results


def report(name: str, rows: list[dict]):
    def agg(rs):
        if not rs:
            return "        —"
        n = len(rs)
        p = sum(r["top1_parent"] for r in rs) / n
        u = sum(r["top1"] for r in rs) / n
        t3 = sum(r["top3_parent"] for r in rs) / n
        ttds = [r["ttd"] for r in rs if r["ttd"] is not None]
        det = len(ttds) / n
        ttd = np.median(ttds) if ttds else float("nan")
        return f"unit {u:6.1%} | parent {p:6.1%} | top3par {t3:6.1%} | det {det:5.1%} | medTTD {ttd:5.1f}s | n={n}"
    first = [r for r in rows if r["seg_idx"] == 1]
    later = [r for r in rows if r["seg_idx"] > 1]
    print(f"\n== {name} ==")
    print(f"  seg 1 (ayah start) : {agg(first)}")
    print(f"  seg >=2 (mid-ayah) : {agg(later)}")
    print(f"  all                : {agg(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    if CACHE.exists() and not args.rebuild_cache:
        streams = pickle.loads(CACHE.read_bytes())
        print(f"loaded {len(streams)} cached streams")
    else:
        streams = build_cache()

    import json
    ayah_ph = {k: v.split() for k, v in json.loads(
        (REPO / "data/lang/ayah_phonemes.json").read_text(encoding="utf-8")).items()}
    seg_raw = json.loads((REPO / "data/lang/segment_phonemes.json").read_text(encoding="utf-8"))
    seg_ph = {k: v["phonemes"].split() for k, v in seg_raw.items()}
    segmented_parents = {k.split("#")[0] for k in seg_ph}
    # Arm B index: segments + unsegmented ayat as single units
    b_refs = dict(seg_ph)
    b_refs.update({k: v for k, v in ayah_ph.items() if k not in segmented_parents})
    print(f"arm A index: {len(ayah_ph)} ayat | arm B index: {len(b_refs)} units "
          f"({len(seg_ph)} segments + {len(b_refs) - len(seg_ph)} whole ayat)")

    a = run_arm(streams, ayah_ph, truth_of=lambda m: m["key"], limit=args.limit)
    report("Arm A — whole-ayah index (parent = truth; unit == parent)", a)

    b = run_arm(streams, b_refs, truth_of=lambda m: f"{m['key']}#{m['seg_idx']:02d}", limit=args.limit)
    report("Arm B — segment index", b)

    # B's confusion structure on misses (research: formulaic-phrase collisions)
    miss = [r for r in b if not r["top1_parent"]]
    print(f"\narm B parent-misses: {len(miss)}/{len(b)}")


if __name__ == "__main__":
    main()
