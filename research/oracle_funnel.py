#!/usr/bin/env python3
"""
Oracle funnel for the sliding-chain detection layer (uses the CURRENT window_best).

For every true segment (aligned span known from segment_spans.csv), across the windows
covering its span center at every scale:

  A retrieved   3-gram shortlist surfaced the truth in >= 1 covering window
  B gated       ...and it passed the length gate at some scale
  C cheap       ...and its infix cost <= FIRE_COST somewhere
  D wins        window_best returned the truth in >= 1 covering window

Split by reference length bucket (the scale-coverage question), plus competitor stats
on D-losses (what beat the truth: a twin? shorter? longer?).

  python research/oracle_funnel.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "matcher"))
sys.path.insert(0, str(REPO / "research"))

CACHE = REPO / "data" / "raw" / "segments" / "full_streams_test.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=1.5)
    args = ap.parse_args()

    from chain_sliding import (window_best, build_ngram_index, _infix_norm,
                               FIRE_COST, SHORTLIST)

    streams = pickle.loads(CACHE.read_bytes())[:args.limit]
    ayah_ph = {k: v.split() for k, v in json.loads(
        (REPO / "data/lang/ayah_phonemes.json").read_text(encoding="utf-8")).items()}
    seg_raw = json.loads((REPO / "data/lang/segment_phonemes.json").read_text(encoding="utf-8"))
    refs = {k: v["phonemes"].split() for k, v in seg_raw.items()}
    segd = {k.split("#")[0] for k in refs}
    refs.update({k: v for k, v in ayah_ph.items() if k not in segd})
    ref_lens = {k: len(v) for k, v in refs.items()}
    idx = build_ngram_index(refs)
    spans = pd.read_csv(REPO / "data/raw/segments/segment_spans.csv").set_index(
        ["recording_id", "seg_idx"])
    by_ref = defaultdict(list)
    for k, v in refs.items():
        by_ref[" ".join(v)].append(k)
    twin_of = {k: set(ks) - {k} for ks in by_ref.values() if len(ks) > 1 for k in ks}

    def bucket(L):
        for hi, name in ((12, "<12"), (25, "12-25"), (45, "25-45"), (80, "45-80")):
            if L < hi:
                return name
        return ">=80"

    scales = (0.7, 1.0, 1.5, 2.2)   # matched filter bank (mirror chain_sliding defaults)
    funnel = defaultdict(Counter)
    beat = Counter()
    for st in streams:
        phons, times = st["phonemes"], st["times"]
        for si in range(1, st["n_segments"] + 1):
            tkey = f"{st['key']}#{si:02d}"
            try:
                row = spans.loc[(st["recording_id"], si)]
            except KeyError:
                continue
            s0, s1 = row.start_sample / 16000, row.end_sample / 16000
            mid = (s0 + s1) / 2
            L = ref_lens[tkey]
            b = bucket(L)
            funnel[b]["n"] += 1
            got = dict(A=0, B=0, C=0, D=0)
            best_winner = None
            for sc in scales:
                W = args.window * sc
                t = max(0.0, mid - W)
                while t <= mid:
                    w0, w1 = t, t + W
                    t += args.hop
                    if not (w0 <= mid <= w1):
                        continue
                    win = [p for p, tt in zip(phons, times) if w0 <= tt < w1]
                    n = len(win)
                    if n < 4:
                        continue
                    from collections import Counter as C2
                    c = C2()
                    for i in range(n - 2):
                        for key in idx.get(tuple(win[i:i + 3]), ()):
                            c[key] += 1
                    short = [k for k, _ in c.most_common(SHORTLIST)]
                    in_short = tkey in short
                    got["A"] |= in_short
                    gate = 0.5 * n <= L <= 1.3 * n   # tight band (mirror window_best)
                    got["B"] |= (in_short and gate)
                    if in_short and gate:
                        got["C"] |= (_infix_norm(refs[tkey], win) <= FIRE_COST)
                    bk, bc = window_best(win, idx, refs, ref_lens)
                    if bk == tkey and bc <= FIRE_COST:
                        got["D"] = 1
                    elif bk is not None and bc <= FIRE_COST and best_winner is None:
                        best_winner = bk
            for k, v in got.items():
                funnel[b][k] += v
            if not got["D"] and best_winner is not None:
                if best_winner in twin_of.get(tkey, ()):
                    beat["exact twin"] += 1
                elif ref_lens[best_winner] > L:
                    beat["longer ref (munch overshoot)"] += 1
                else:
                    beat["shorter/other ref"] += 1
            elif not got["D"]:
                beat["no window fired at all"] += 1

    print(f"{'bucket':>8} {'n':>5} {'A_short':>8} {'B_gate':>8} {'C_cheap':>8} {'D_wins':>8}")
    tot = Counter()
    for b in ("<12", "12-25", "25-45", "45-80", ">=80"):
        f = funnel[b]
        if not f["n"]:
            continue
        tot.update(f)
        print(f"{b:>8} {f['n']:>5} " + " ".join(f"{f[k]/f['n']:>8.1%}" for k in "ABCD"))
    print(f"{'ALL':>8} {tot['n']:>5} " + " ".join(f"{tot[k]/tot['n']:>8.1%}" for k in "ABCD"))
    print("\nD-losses — what beat the truth:")
    n_loss = sum(beat.values())
    for k, v in beat.most_common():
        print(f"  {k:32} {v:>4}  ({v/n_loss:.1%})")


if __name__ == "__main__":
    main()
