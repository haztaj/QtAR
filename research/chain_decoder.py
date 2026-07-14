#!/usr/bin/env python3
"""
Chained segment decoding over FULL continuous clips — the streaming-realistic condition.

A full recording of a segmented ayah contains its waqf segments in order. The decoder
consumes the phoneme stream and emits the segment chain: match against the segment
index; when the leading unit is COMPLETE (matcher ayah_progress) and has held the
margin (T, K persistence — segment-scale, i.e. the Juz-Amma-like regime), commit it,
reset the matcher (anchor advances), and bias the EXPECTED successor (segment n -> n+1)
with a context bonus.

The experiment is the context ablation: --bonus 0.22 (ayah-level constant) vs --bonus 0.
The miss dissection says 74.5% of cold-matching errors are exact twins; if sequential
context works, twin-position accuracy should jump when the bonus is on — validating the
97.9% ceiling empirically.

  python research/chain_decoder.py               # both arms (bonus 0.22 and 0)
  python research/chain_decoder.py --limit 200   # quick pass
"""

from __future__ import annotations

import argparse
import json
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
LINEAR_BUDGET = 24000
QUAD_BUDGET = 1.0e8
CACHE = REPO / "data" / "raw" / "segments" / "full_streams_test.pkl"

# Segment-scale commit policy (segments are Juz-Amma-sized units).
COMMIT_T = 0.15
COMMIT_K = 5
COMPLETE_COST = 0.45
MIN_INPUT_FRAC = 0.6
FINALIZE_PROGRESS = 0.5   # end-of-stream: flush the top partial if this far through


# ---------------------------------------------------------------------------
# Full-clip stream cache
# ---------------------------------------------------------------------------

def build_cache() -> list[dict]:
    from data import load_wav_16k, logmel_16k, load_tokens
    from model import EmformerCTC
    from torch.nn.utils.rnn import pad_sequence

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(REPO / "training/exp/best_s123.pt", map_location=device)
    model = EmformerCTC(num_tokens=ck["vocab"]).to(device).eval()
    model.load_state_dict(ck["model"])
    id2tok = {v: k for k, v in load_tokens().items()}

    manifest = pd.read_csv(REPO / "data/raw/audio/manifest.csv")
    manifest = manifest[manifest.reciter_id.isin(TEST_RECITERS)]
    spans = pd.read_csv(REPO / "data/raw/segments/segment_spans.csv")
    n_seg = spans.groupby("key")["seg_idx"].max().to_dict()
    keys = set(n_seg)
    rows = manifest[[f"{s}:{a}" in keys for s, a in zip(manifest.surah_id, manifest.ayah_id)]]
    rows = rows.sort_values("duration").to_dict("records")
    print(f"full test clips of segmented ayat: {len(rows)}")

    from chain_sliding import greedy_with_alts   # Phase 0: also store posterior alternatives

    out: list[dict] = []
    i = 0
    while i < len(rows):
        batch = [rows[i]]; i += 1
        while i < len(rows):
            mx = max(r["duration"] for r in batch + [rows[i]]) * 100
            if (len(batch) + 1) * mx > LINEAR_BUDGET or (len(batch) + 1) * mx * mx > QUAD_BUDGET:
                break
            batch.append(rows[i]); i += 1
        feats = [logmel_16k(load_wav_16k(r["path"])) for r in batch]
        lens = torch.tensor([f.shape[0] for f in feats])
        padded = pad_sequence(feats, batch_first=True)
        with torch.no_grad():
            if device == "cuda":
                torch.cuda.empty_cache()
                with torch.amp.autocast("cuda"):
                    lp, ol = model(padded.to(device), lens.to(device))
            else:
                lp, ol = model(padded.to(device), lens.to(device))
        lp = lp.float().cpu()
        for b, r in enumerate(batch):
            phons, times, alts = greedy_with_alts(lp[b], int(ol[b]), id2tok, ENC_FRAME_SEC)
            key = f"{r['surah_id']}:{r['ayah_id']}"
            out.append({"recording_id": r["recording_id"], "key": key,
                        "n_segments": n_seg[key], "dur": r["duration"],
                        "phonemes": phons, "times": times, "alts": alts})
        if len(out) % 200 < len(batch):
            print(f"  decoded {len(out)}/{len(rows)}", flush=True)
    CACHE.write_bytes(pickle.dumps(out))
    print(f"cached {len(out)} full-clip streams -> {CACHE}")
    return out


# ---------------------------------------------------------------------------
# Chained decoder
# ---------------------------------------------------------------------------

def successor(key: str, refs) -> str | None:
    if "#" not in key:            # whole-ayah unit (unsegmented) — no within-ayah successor
        return None
    parent, idx = key.split("#")
    nxt = f"{parent}#{int(idx) + 1:02d}"
    return nxt if nxt in refs else None


def decode_chain(stream, trie, refs, bonus: float):
    """Streaming chained decode. Returns list of emitted unit keys."""
    from phoneme_matcher import PhonemeMatcher
    emitted: list[str] = []
    expected: str | None = None
    m = PhonemeMatcher(trie, allow_restart=False)
    leader, run = None, 0

    def adjusted(cands):
        adj = [(c.norm_cost - (bonus if c.key == expected else 0.0), c) for c in cands]
        adj.sort(key=lambda t: t[0])
        return adj

    for p in stream["phonemes"]:
        m.step(p)
        cands = m.partial_candidates(k=8, min_progress=0.15)
        if not cands:
            continue
        adj = adjusted(cands)
        top = adj[0][1]
        margin = (adj[1][0] - adj[0][0]) if len(adj) > 1 else float("inf")
        if top.key == leader and margin >= COMMIT_T:
            run += 1
        elif margin >= COMMIT_T:
            leader, run = top.key, 1
        else:
            leader, run = None, 0
        if run >= COMMIT_K:
            prog, term, complete = m.ayah_progress(top.key, complete_cost=COMPLETE_COST,
                                                   min_input_frac=MIN_INPUT_FRAC)
            if complete:
                emitted.append(top.key)
                expected = successor(top.key, refs)
                m = PhonemeMatcher(trie, allow_restart=False)
                leader, run = None, 0
    # end-of-stream finalize: flush the leading partial if far enough through
    cands = m.partial_candidates(k=8, min_progress=FINALIZE_PROGRESS)
    if cands:
        adj = adjusted(cands)
        emitted.append(adj[0][1].key)
    return emitted


def edit_seq(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--bonus", type=float, default=None, help="run a single bonus instead of both")
    args = ap.parse_args()

    from phoneme_matcher import PhonemeTrie

    if CACHE.exists() and not args.rebuild_cache:
        streams = pickle.loads(CACHE.read_bytes())
        print(f"loaded {len(streams)} cached full-clip streams")
    else:
        streams = build_cache()
    if args.limit:
        streams = streams[:args.limit]

    ayah_ph = {k: v.split() for k, v in json.loads(
        (REPO / "data/lang/ayah_phonemes.json").read_text(encoding="utf-8")).items()}
    seg_raw = json.loads((REPO / "data/lang/segment_phonemes.json").read_text(encoding="utf-8"))
    refs = {k: v["phonemes"].split() for k, v in seg_raw.items()}
    segmented = {k.split("#")[0] for k in refs}
    refs.update({k: v for k, v in ayah_ph.items() if k not in segmented})
    trie = PhonemeTrie.from_ayah_phonemes(refs)

    # twin classes: identical phoneme references
    by_ref = defaultdict(list)
    for k, v in refs.items():
        by_ref[" ".join(v)].append(k)
    twins = {k for ks in by_ref.values() if len(ks) > 1 for k in ks}
    print(f"index: {len(refs)} units | twin units: {len(twins)}")

    for bonus in ([args.bonus] if args.bonus is not None else [0.22, 0.0]):
        ser_num = ser_den = 0
        pos_ok = pos_n = twin_ok = twin_n = 0
        exact = 0
        for st in streams:
            truth = [f"{st['key']}#{i:02d}" for i in range(1, st["n_segments"] + 1)]
            emitted = decode_chain(st, trie, refs, bonus)
            ser_num += edit_seq(emitted, truth)
            ser_den += len(truth)
            exact += emitted == truth
            for i, t in enumerate(truth):
                ok = i < len(emitted) and emitted[i] == t
                pos_ok += ok; pos_n += 1
                if t in twins:
                    twin_ok += ok; twin_n += 1
        print(f"\n== chained decode, context bonus = {bonus} ==")
        print(f"  segment error rate (SER)   : {ser_num/ser_den:6.1%}")
        print(f"  exact-chain clips          : {exact/len(streams):6.1%}  (n={len(streams)})")
        print(f"  positional segment accuracy: {pos_ok/pos_n:6.1%}  ({pos_n} positions)")
        print(f"  twin-position accuracy     : {twin_ok/max(1,twin_n):6.1%}  ({twin_n} twin positions)")


if __name__ == "__main__":
    main()
