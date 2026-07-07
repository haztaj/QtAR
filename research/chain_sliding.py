#!/usr/bin/env python3
"""
Chained segment decoding v2: SLIDING-WINDOW chaining (production-proven architecture).

v1 (chain_decoder.py, commit-and-reset anchoring) failed with a desync cascade:
pos-1 accuracy 83.6%, pos-2 10.1% — the reset discards the next segment's consumed
prefix, so nothing downstream ever completes. This v2 uses the architecture already
validated live at ayah level on short units (demo/sliding.py -> sdk segmenter.cpp):
stateless windows over the phoneme stream, each matched whole-window against the
segment index; a vote state machine assembles the chain. Windows re-anchor at every
hop — desync is impossible by construction.

Context enters as in the production segmenter: committing the EXPECTED successor
(segment n -> n+1) needs `votes_next` consecutive agreeing windows; anything else
needs `votes_jump`. The ablation compares (votes_next=1, votes_jump=2) against a
context-blind (2,2).

  python research/chain_sliding.py
  python research/chain_sliding.py --window 10 --hop 2.5 --cost 0.35
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "matcher"))

CACHE = REPO / "data" / "raw" / "segments" / "full_streams_test.pkl"


def successor(key: str, refs) -> str | None:
    if "#" not in key:
        return None
    parent, idx = key.split("#")
    nxt = f"{parent}#{int(idx) + 1:02d}"
    return nxt if nxt in refs else None


def make_succ_full(refs):
    """Cross-ayah successor over the unit index: within an ayah -> next segment;
    last unit -> the next ayah's first unit. (The deployment-condition succ_fn.)"""
    n_segs: dict[str, int] = {}
    for k in refs:
        if "#" in k:
            parent, idx = k.split("#")
            n_segs[parent] = max(n_segs.get(parent, 0), int(idx))
    segmented = set(n_segs)

    def first_unit(ayah: str) -> str:
        return f"{ayah}#01" if ayah in segmented else ayah

    def succ_full(key: str) -> str | None:
        parent = key.split("#")[0]
        if "#" in key:
            idx = int(key.split("#")[1])
            if idx < n_segs.get(parent, 0):
                return f"{parent}#{idx + 1:02d}"
        s, a = (int(x) for x in parent.split(":"))
        nxt = f"{s}:{a + 1}"
        return first_unit(nxt) if (nxt in segmented or nxt in refs) else None

    return succ_full


def assemble(emitted, succ_fn):
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
        return u == succ_fn(p) or (parent(u) == parent(p) and idx(u) > idx(p))

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
            if u == succ_fn(last):
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
        if confirmed and (p == succ_fn(confirmed[-1])
                          or (parent(p) == parent(confirmed[-1])
                              and idx(p) > idx(confirmed[-1]))):
            confirmed.append(p)
    if not confirmed and pending:
        confirmed.append(pending[0])           # lone emissions — keep the first
    return confirmed


def _edit_norm(a: list, b: list) -> float:
    """Normalized edit distance (0=identical) — the production window score (demo/sliding.py)."""
    if len(b) < len(a):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for cb in b:
        cur = prev[0] + 1
        prevj = prev[0]
        row = [cur]
        for i, ca in enumerate(a, 1):
            cur = min(prev[i] + 1, cur + 1, prevj + (ca != cb))
            prevj = prev[i]
            row.append(cur)
        prev = row
    return prev[-1] / max(1, max(len(a), len(b)))


SHORTLIST = 60     # n-gram-shortlisted candidates per window (speed; production scores all)
FIRE_COST = 0.30   # windows firing at/below this enter blended selection
COVER_BONUS = 0.15 # selection = cost - COVER_BONUS*coverage (anti-snippet AND anti-overshoot)
STRONG_COST = 0.15 # near-certain fire (truth median 0.08): commits with a single vote
MIN_ADVANCE = 2.0  # a new emission needs its window to extend this far past the last commit
REPEAT_SUPPRESS = 20.0  # suppress re-emitting the same unit within this many seconds


def build_ngram_index(refs, n: int = 3):
    """Inverted index phoneme-3gram -> ref keys. Alignment-free candidate retrieval:
    the trie shortlist (root-anchored, even with restart) failed to surface refs whose
    match starts mid-window; shared-3gram counting has no anchoring assumptions.
    Values are SORTED TUPLES, not sets: set iteration order is hash-randomized per
    process, which made Counter tie-breaks (exactly the twin cases) nondeterministic
    across runs (~±5 pts on the twin metric)."""
    from collections import defaultdict as dd
    idx = dd(set)
    for key, ph in refs.items():
        for i in range(len(ph) - n + 1):
            idx[tuple(ph[i:i + n])].add(key)
    return {g: tuple(sorted(ks, key=_key_sort)) for g, ks in idx.items()}


def _key_sort(u: str):
    sa, _, seg = u.partition("#")
    s, a = sa.split(":")
    return int(s), int(a), int(seg) if seg else 0


def _infix_norm(ref: list, win: list) -> float:
    """Infix-normalized edit distance: best alignment of `ref` as a SUBSTRING of `win`
    (free leading/trailing window gaps), / len(ref). Windows start/end at arbitrary
    offsets relative to segment boundaries — whole-window distance punishes the edge
    junk as errors; infix does not."""
    m = len(ref)
    prev = [0] * (len(win) + 1)              # free leading skips in the window
    for i in range(1, m + 1):
        cur = [i] + [0] * len(win)
        ri = ref[i - 1]
        for j in range(1, len(win) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ri != win[j - 1]))
        prev = cur
    return min(prev) / max(1, m)             # free trailing skips


def _prefix_norm(ref, win, min_i, end_slack: int = 2):
    """Best PREFIX of `ref` aligned to the END of `win` (free leading window skips,
    up to `end_slack` trailing window positions of slack for CTC timing noise).
    Returns (norm_cost, prefix_len) minimizing cost/len over len >= min_i — the
    early-detection score: 'is the reciter currently this far into this unit?'."""
    m, n = len(ref), len(win)
    prev = [0] * (n + 1)                     # free leading skips
    best, best_i = 1e9, 0
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ri = ref[i - 1]
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ri != win[j - 1]))
        if i >= min_i:
            c = min(cur[max(0, n - end_slack):]) / i
            if c < best:
                best, best_i = c, i
        prev = cur
    return best, best_i


def window_best(win, ngram_idx, refs, ref_lens, fire_cost=FIRE_COST):
    """Best (key, cost) for one window: 3-gram shortlist -> infix edit-norm.
    Length gate is TIGHT (0.5n..1.3n): with the multi-scale filter bank each window
    size only fires refs of its own length class — small windows can't be swallowed
    by long refs, big windows can't fire snippets. A loose gate (0.3..1.6) plus small
    scales regressed end-to-end: recall rose but noisy small-window fires flooded the
    vote machine (raw unit SER 32.6% -> 41.6%)."""
    import heapq
    from collections import Counter
    c = Counter()
    for i in range(len(win) - 2):
        for key in ngram_idx.get(tuple(win[i:i + 3]), ()):
            c[key] += 1
    n = len(win)
    # Shortlist: top by raw shared-3gram count UNION top by length-normalized count.
    # The normalized pass runs over the FULL counter: a 5-ph ref shares at most 3
    # 3-grams, ranking ~230th by raw count even when decoded perfectly (diagnostic
    # 2026-07-07) — restricting normalization to the raw top-180 never sees it.
    short = [k for k, _ in c.most_common(SHORTLIST)]
    short += [k for k, _ in heapq.nlargest(20, c.items(),
                                           key=lambda kv: kv[1] / ref_lens[kv[0]])]
    # Blended selection: cost - COVER_BONUS * coverage. Pure-cost lets short formulaic
    # snippets embed at ~0 cost; pure-longest (maximal munch) swallows short truths with
    # longer refs (oracle: 84.8% of losses). Coverage-blended cost handles both.
    best_key, best_cost = None, 1e9
    best_sel = 1e9
    for key in dict.fromkeys(short):
        L = ref_lens[key]
        if not (0.5 * n <= L <= 1.3 * n):   # tight band: each scale serves its size class
            continue
        cost = _infix_norm(refs[key], win)
        sel = cost - COVER_BONUS * min(L, n) / n
        if cost <= fire_cost and sel < best_sel:
            best_sel, best_key, best_cost = sel, key, cost
        elif best_sel == 1e9 and cost < best_cost:
            best_key, best_cost = key, cost
    return best_key, best_cost


def decode_sliding(stream, ngram_idx, refs, window_s, hop_s, cost_thresh,
                   votes_next: int, votes_jump: int, ref_lens=None,
                   scales=(0.2, 0.7, 1.0, 1.5, 2.2),
                   use_twin_sub: bool = True, succ_fn=None, confusable=None,
                   early_prefix: float | None = None):
    """Multi-scale sliding windows; per window the production whole-window edit-norm
    (trie-shortlisted); vote state machine emits the chain.

    early_prefix (e.g. 0.5): context-gated EARLY detection — at each largest-scale
    window, if the window TAIL matches >= this fraction of the EXPECTED successor's
    prefix (cost <= cost_thresh), fire the expected unit without waiting for it to
    complete. Only ever fires the unit context already predicts (low risk); addresses
    whole-unit matching's inherent commit-at-unit-end latency."""
    if ref_lens is None:
        ref_lens = {k: len(v) for k, v in refs.items()}
    if succ_fn is None:
        succ_fn = lambda k: successor(k, refs)   # within-ayah only (per-clip eval)
    phons, times = stream["phonemes"], stream["times"]
    if not phons:
        return []
    # collect window fires (t, key, cost) across scales, then vote in time order;
    # kind 0 = prefix-check event (largest scale only), kind 1 = whole-unit fire
    fires = []
    t_end = times[-1]
    for si, sc in enumerate(scales):
        w = window_s * sc
        largest = si == len(scales) - 1
        t = 0.0
        j0 = 0
        while t <= t_end + 1e-6:
            w0, w1 = t, t + w
            while j0 < len(times) and times[j0] < w0:
                j0 += 1
            j1 = j0
            while j1 < len(times) and times[j1] < w1:
                j1 += 1
            win = phons[j0:j1]
            t += hop_s
            if len(win) < 4:
                continue
            if early_prefix and largest:
                fires.append((w1, 0, "", 0.0, win))
            key, cost = window_best(win, ngram_idx, refs, ref_lens, fire_cost=cost_thresh)
            if key is not None and cost <= cost_thresh:
                fires.append((w1, 1, key, cost, None))
    fires.sort(key=lambda f: (f[0], f[1], f[2], f[3]))

    emitted: list[str] = []
    emit_t: list[float] = []
    expected: str | None = None
    pending: str | None = None
    votes = 0
    consumed = -1e9                        # soft time anchor: emission gate only —
    for w1, kind, top, cost, win in fires:  # matching stays stateless (no cascade risk)
        if kind == 0:                      # prefix-check event: early-fire the EXPECTED unit
            if expected is None:
                continue
            L = ref_lens[expected]
            min_i = max(6, int(early_prefix * L + 0.999999))
            if L < min_i:
                continue
            pcost, _ = _prefix_norm(refs[expected], win, min_i)
            if pcost > cost_thresh:
                continue
            top, cost = expected, pcost    # falls through the normal gates below
        if w1 < consumed + MIN_ADVANCE:
            continue                       # this window is mostly inside the last commit
        if emitted and top == emitted[-1]:
            continue                       # same unit still in view
        if top in emitted and w1 - emit_t[emitted.index(top)] < REPEAT_SUPPRESS:
            continue                       # late re-fire of an already-emitted unit
        # Twin substitution: exact twins (identical refs) tie on cost AND length — the
        # matcher cannot pick between them. If the fire is a twin of the EXPECTED unit,
        # context resolves it: emit the expected one. (The dissection's core claim.)
        # With `confusable` (the ambiguity map), extend to NEAR-twins: within tau the
        # acoustic evidence can't reliably separate them either, and the sequential
        # prior favors the expected unit.
        if (use_twin_sub and expected is not None and top != expected
                and (refs[top] == refs[expected]
                     or (confusable is not None and expected in confusable.get(top, ())))):
            top = expected
        need = votes_next if top == expected else votes_jump
        if cost <= STRONG_COST:
            need = min(need, 1)            # confidence-scaled: strong fires commit alone
        if top == pending:
            votes += 1
        else:
            pending, votes = top, 1
        if votes >= need:
            emitted.append(top)
            emit_t.append(w1)
            consumed = w1 - 2.0            # keep some overlap for the next unit's window
            expected = succ_fn(top)
            pending, votes = None, 0
    return emitted


def edit_seq(a, b):
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
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=1.5)
    ap.add_argument("--cost", type=float, default=0.30)
    args = ap.parse_args()

    from phoneme_matcher import PhonemeTrie

    streams = pickle.loads(CACHE.read_bytes())
    if args.limit:
        streams = streams[:args.limit]
    print(f"{len(streams)} full-clip streams | window {args.window}s hop {args.hop}s "
          f"cost<={args.cost}")

    ayah_ph = {k: v.split() for k, v in json.loads(
        (REPO / "data/lang/ayah_phonemes.json").read_text(encoding="utf-8")).items()}
    seg_raw = json.loads((REPO / "data/lang/segment_phonemes.json").read_text(encoding="utf-8"))
    refs = {k: v["phonemes"].split() for k, v in seg_raw.items()}
    segmented = {k.split("#")[0] for k in refs}
    refs.update({k: v for k, v in ayah_ph.items() if k not in segmented})
    ngram_idx = build_ngram_index(refs)

    ref_lens = {k: len(v) for k, v in refs.items()}
    by_ref = defaultdict(list)
    for k, v in refs.items():
        by_ref[" ".join(v)].append(k)
    twins = {k for ks in by_ref.values() if len(ks) > 1 for k in ks}

    for name, vn, vj in (("context (next=1, jump=2)", 1, 2),
                         ("blind   (next=2, jump=2)", 2, 2)):
        ser_num = ser_den = exact = 0
        pos = defaultdict(lambda: [0, 0])
        twin_ok = twin_n = 0
        for st in streams:
            truth = [f"{st['key']}#{i:02d}" for i in range(1, st["n_segments"] + 1)]
            emitted = decode_sliding(st, ngram_idx, refs, args.window, args.hop, args.cost, vn, vj,
                                     ref_lens=ref_lens, use_twin_sub=(vn != vj))
            ser_num += edit_seq(emitted, truth)
            ser_den += len(truth)
            exact += emitted == truth
            for i, tkey in enumerate(truth):
                ok = i < len(emitted) and emitted[i] == tkey
                pos[min(i, 3)][0] += ok
                pos[min(i, 3)][1] += 1
                if tkey in twins:
                    twin_ok += ok; twin_n += 1
        n_pos = sum(v[1] for v in pos.values())
        n_ok = sum(v[0] for v in pos.values())
        print(f"\n== sliding chain, {name} ==")
        print(f"  SER {ser_num/ser_den:6.1%} | exact chains {exact/len(streams):5.1%} | "
              f"positional {n_ok/n_pos:6.1%} | twins {twin_ok/max(1,twin_n):6.1%} (n={twin_n})")
        print("  by position: " + " | ".join(
            f"pos{p+1}{'+' if p == 3 else ''} {v[0]/v[1]:5.1%} (n={v[1]})"
            for p, v in sorted(pos.items())))


if __name__ == "__main__":
    main()
