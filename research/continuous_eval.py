#!/usr/bin/env python3
"""
Continuous-recitation evaluation — the deployment-realistic condition.

Composes multi-ayah streams (runs of 4 consecutive ayat per test reciter, concatenated
phoneme streams from the per-clip caches) and runs the v8 sliding chain decoder with
FULL sequential context: within-ayah segment succession AND cross-ayah handoff (last
unit of ayah N -> first unit of ayah N+1). This gives position-1-of-ayah twins the
context they structurally lack in the per-clip eval — the condition that decides the
methodology's effective accuracy.

Also reports the derived AYAH chain (consecutive-parent dedup of the emitted units):
"would the mushaf highlight track correctly?"

  python research/continuous_eval.py
  python research/continuous_eval.py --limit 100 --run-len 4
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "matcher"))
sys.path.insert(0, str(REPO / "research"))

TEST_RECITERS = {"warsh_husary", "warsh_yassin", "yasser_ad_dussary"}
ENC_FRAME_SEC = 0.04
LINEAR_BUDGET = 24000
QUAD_BUDGET = 1.0e8
SEG_CACHE = REPO / "data" / "raw" / "segments" / "full_streams_test.pkl"
UNSEG_CACHE = REPO / "data" / "raw" / "segments" / "unseg_streams_test.pkl"


def build_unseg_cache() -> list[dict]:
    """Decode full clips of UNSEGMENTED corpus ayat for the test reciters (the segment
    cache only covers segmented ayat)."""
    from data import load_wav_16k, logmel_16k, load_tokens
    from model import EmformerCTC
    from torch.nn.utils.rnn import pad_sequence

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(REPO / "training/exp/best_s123.pt", map_location=device)
    model = EmformerCTC(num_tokens=ck["vocab"]).to(device).eval()
    model.load_state_dict(ck["model"])
    id2tok = {v: k for k, v in load_tokens().items()}

    seg_keys = set(pd.read_csv(REPO / "data/raw/segments/segment_spans.csv")["key"].unique())
    manifest = pd.read_csv(REPO / "data/raw/audio/manifest.csv")
    manifest = manifest[manifest.reciter_id.isin(TEST_RECITERS)]
    rows = manifest[[f"{s}:{a}" not in seg_keys for s, a in zip(manifest.surah_id, manifest.ayah_id)]]
    rows = rows.sort_values("duration").to_dict("records")
    print(f"unsegmented test clips to decode: {len(rows)}")

    def greedy_with_times(lp, T):
        ids = lp[:T].argmax(-1).tolist()
        phons, times, prev = [], [], -1
        for f, s in enumerate(ids):
            if s != prev and s != 0:
                phons.append(id2tok[s]); times.append(f * ENC_FRAME_SEC)
            prev = s
        return phons, times

    out = []
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
            phons, times = greedy_with_times(lp[b], int(ol[b]))
            out.append({"recording_id": r["recording_id"], "reciter": r["reciter_id"],
                        "key": f"{r['surah_id']}:{r['ayah_id']}", "dur": r["duration"],
                        "phonemes": phons, "times": times})
        if len(out) % 400 < len(batch):
            print(f"  decoded {len(out)}/{len(rows)}", flush=True)
    UNSEG_CACHE.write_bytes(pickle.dumps(out))
    print(f"cached {len(out)} unsegmented streams -> {UNSEG_CACHE}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max sequences")
    ap.add_argument("--run-len", type=int, default=4, help="consecutive ayat per stream")
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=1.5)
    ap.add_argument("--cost", type=float, default=0.30)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    from chain_sliding import decode_sliding, build_ngram_index
    from phoneme_matcher import PhonemeTrie  # noqa: F401  (parity with other scripts)

    # --- per-clip stream lookup: (reciter, ayah-key) -> stream ---
    seg_streams = pickle.loads(SEG_CACHE.read_bytes())
    if UNSEG_CACHE.exists() and not args.rebuild_cache:
        unseg_streams = pickle.loads(UNSEG_CACHE.read_bytes())
    else:
        unseg_streams = build_unseg_cache()
    lookup: dict[tuple[str, str], dict] = {}
    for st in seg_streams:
        rec = st["recording_id"].rsplit("_s", 1)[0]
        lookup[(rec, st["key"])] = st
    for st in unseg_streams:
        lookup[(st["reciter"], st["key"])] = st

    # --- references / index ---
    ayah_ph = {k: v.split() for k, v in json.loads(
        (REPO / "data/lang/ayah_phonemes.json").read_text(encoding="utf-8")).items()}
    seg_raw = json.loads((REPO / "data/lang/segment_phonemes.json").read_text(encoding="utf-8"))
    refs = {k: v["phonemes"].split() for k, v in seg_raw.items()}
    n_segs = {k.split("#")[0]: v["n_segments"] for k, v in seg_raw.items()}
    segmented = set(n_segs)
    refs.update({k: v for k, v in ayah_ph.items() if k not in segmented})
    ref_lens = {k: len(v) for k, v in refs.items()}
    ngram_idx = build_ngram_index(refs)

    def units_of(ayah: str) -> list[str]:
        return ([f"{ayah}#{i:02d}" for i in range(1, n_segs[ayah] + 1)]
                if ayah in segmented else [ayah])

    def first_unit(ayah: str) -> str:
        return f"{ayah}#01" if ayah in segmented else ayah

    # cross-ayah successor: within ayah -> next segment; last unit -> next ayah's first
    def succ_full(key: str) -> str | None:
        parent = key.split("#")[0]
        if "#" in key:
            idx = int(key.split("#")[1])
            if idx < n_segs.get(parent, 0):
                return f"{parent}#{idx + 1:02d}"
        s, a = (int(x) for x in parent.split(":"))
        nxt = f"{s}:{a + 1}"
        return first_unit(nxt) if (nxt in segmented or nxt in refs) else None

    # twin units (identical refs)
    by_ref = defaultdict(list)
    for k, v in refs.items():
        by_ref[" ".join(v)].append(k)
    twins = {k for ks in by_ref.values() if len(ks) > 1 for k in ks}

    # NOTE: extending twin substitution to NEAR-twins via the segment ambiguity map
    # (data/lang/ambiguous_units.json -> decode_sliding(confusable=...)) measured
    # NEUTRAL on the full 747 run (SER 14.5% = 14.5%) — exact-twin substitution
    # already captures the resolvable mass; the 65 near-twin units are too rare.

    # --- compose sequences: non-overlapping runs of consecutive ayat per reciter ---
    seqs = []
    ayat_by_surah = defaultdict(list)
    for k in refs:
        parent = k.split("#")[0]
        s, a = (int(x) for x in parent.split(":"))
        ayat_by_surah[s].append(a)
    for s in ayat_by_surah:
        ayat_by_surah[s] = sorted(set(ayat_by_surah[s]))
    for reciter in sorted(TEST_RECITERS):
        for s, ayat in sorted(ayat_by_surah.items()):
            for i in range(0, len(ayat) - args.run_len + 1, args.run_len):
                run = ayat[i:i + args.run_len]
                if run != list(range(run[0], run[0] + len(run))):
                    continue                      # must be consecutive
                keys = [f"{s}:{a}" for a in run]
                sts = [lookup.get((reciter, k)) for k in keys]
                if any(x is None for x in sts):
                    continue
                phons, times, t0 = [], [], 0.0
                for st in sts:
                    phons.extend(st["phonemes"])
                    times.extend(t + t0 for t in st["times"])
                    t0 += st["dur"]
                truth = [u for k in keys for u in units_of(k)]
                seqs.append({"reciter": reciter, "keys": keys, "truth": truth,
                             "phonemes": phons, "times": times, "dur": t0})
    if args.limit:
        seqs = seqs[:args.limit]
    n_units = sum(len(q["truth"]) for q in seqs)
    print(f"sequences: {len(seqs)} (run-len {args.run_len}) | truth units: {n_units} | "
          f"mean dur {sum(q['dur'] for q in seqs)/len(seqs):.0f}s")

    def edit_seq(a, b):
        m, n = len(a), len(b)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            cur = [i] + [0] * n
            for j in range(1, n + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            prev = cur
        return prev[n]

    def aligned_truth_hits(emitted, truth):
        """Edit-distance traceback -> for each TRUTH position, was it matched exactly?
        (Prefix indexing shifts after one insertion; alignment scoring does not.)"""
        m, n = len(emitted), len(truth)
        D = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): D[i][0] = i
        for j in range(n + 1): D[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                D[i][j] = min(D[i-1][j] + 1, D[i][j-1] + 1,
                              D[i-1][j-1] + (emitted[i-1] != truth[j-1]))
        hits = [False] * n
        i, j = m, n
        while i > 0 and j > 0:
            if D[i][j] == D[i-1][j-1] + (emitted[i-1] != truth[j-1]):
                hits[j-1] = emitted[i-1] == truth[j-1]
                i, j = i - 1, j - 1
            elif D[i][j] == D[i-1][j] + 1:
                i -= 1
            else:
                j -= 1
        return hits

    def assemble(emitted):
        """Deferral-grade chain assembly (HighlightController logic at unit level):
        - the EXPECTED successor of the last confirmed unit confirms immediately;
        - a unit continuing the current parent forward confirms;
        - an unexpected jump is DEFERRED (up to TWO pending): confirmed retroactively
          when a later emission supports it (its successor or same-parent forward).
          The 2-deep buffer gives JUNK TOLERANCE 1 — one interloper between a true
          unit and its supporter no longer kills the true unit (diagnostic 2026-07-07:
          that single mechanism accounted for ~110/128 assembly-lost hits, dominated
          by cold starts where the chain never got seeded);
        - unsupported pendings age out (interloper — the twin-error signature);
        - backward/repeat emissions within the current parent are dropped (re-fires)."""
        confirmed: list[str] = []
        pending: list[str] = []                    # oldest first, len <= 2

        def parent(u): return u.split("#")[0]
        def idx(u): return int(u.split("#")[1]) if "#" in u else 0

        def supports(p, u):
            return u == succ_full(p) or (parent(u) == parent(p) and idx(u) > idx(p))

        for u in emitted:
            sup = next((k for k in range(len(pending) - 1, -1, -1)
                        if supports(pending[k], u)), None)
            if sup is not None:                    # retro-confirm the supported pending
                confirmed.append(pending[sup])     # (junk between/after it is dropped)
                confirmed.append(u)
                pending = []
                continue
            if confirmed:
                last = confirmed[-1]
                if u == succ_full(last):
                    confirmed.append(u)            # expected successor: confirm now
                    pending = []
                    continue
                if parent(u) == parent(last):
                    if idx(u) > idx(last):
                        confirmed.append(u)        # forward skip within the parent
                        pending = []
                    continue                       # backward/repeat: drop (re-fire)
            pending.append(u)                      # unexpected jump: await support
            if len(pending) > 2:
                pending.pop(0)                     # oldest ages out (interloper)
        for p in pending:                          # end of stream: flush chainable tail
            if confirmed and (p == succ_full(confirmed[-1])
                              or (parent(p) == parent(confirmed[-1])
                                  and idx(p) > idx(confirmed[-1]))):
                confirmed.append(p)
        if not confirmed and pending:
            confirmed.append(pending[0])           # lone emissions — keep the first
        return confirmed

    def parents_dedup(units, smooth: bool = False):
        seq = [u.split("#")[0] for u in units]
        if smooth:
            # Drop single-unit parent islands sandwiched inside another parent's run —
            # the interleaved-twin-error case the highlight layer defers/ignores.
            keep = []
            for i, p in enumerate(seq):
                if 0 < i < len(seq) - 1 and seq[i-1] == seq[i+1] != p:
                    continue
                keep.append(p)
            seq = keep
        out = []
        for p in seq:
            if not out or out[-1] != p:
                out.append(p)
        return out

    for name, vn, vj, tw, asm, conf in (
            ("context+twin+ASSEMBLY", 1, 2, True, True, None),
            ("context+twin", 1, 2, True, False, None),
            ("blind", 2, 2, False, False, None)):
        ser_n = ser_d = exact = 0
        pos_ok = pos_n = twin_ok = twin_n = 0
        aser_n = aser_d = 0
        for q in seqs:
            emitted = decode_sliding(q, ngram_idx, refs, args.window, args.hop, args.cost,
                                     vn, vj, ref_lens=ref_lens, use_twin_sub=tw,
                                     succ_fn=succ_full, confusable=conf)
            if asm:
                emitted = assemble(emitted)
            truth = q["truth"]
            ser_n += edit_seq(emitted, truth); ser_d += len(truth)
            exact += emitted == truth
            hits = aligned_truth_hits(emitted, truth)
            for tkey, ok in zip(truth, hits):
                pos_ok += ok; pos_n += 1
                if tkey in twins:
                    twin_ok += ok; twin_n += 1
            ta = parents_dedup(truth)
            ea = parents_dedup(emitted, smooth=not asm)   # assembly needs no smoothing
            aser_n += edit_seq(ea, ta); aser_d += len(ta)
        print(f"\n== continuous, {name} ==")
        print(f"  unit SER {ser_n/ser_d:6.1%} | exact seqs {exact/len(seqs):5.1%} | "
              f"aligned-hit {pos_ok/pos_n:6.1%} | twins {twin_ok/max(1,twin_n):6.1%} (n={twin_n})")
        print(f"  AYAH-chain SER {aser_n/aser_d:6.1%}   <- would the mushaf highlight track?")


if __name__ == "__main__":
    main()
