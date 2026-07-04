#!/usr/bin/env python3
"""
Prefix-anchored streaming detection (`--mode stream`).

The sliding-window segmenter matches each fixed window as a WHOLE against WHOLE ayat, so a
long ayah (e.g. 78:40, 105 phonemes) is length-pruned out of every 4 s window and never
detected. This detector instead accumulates the decode of the CURRENT ayah's audio and
scores each ayah by PREFIX ALIGNMENT — the min cost to turn the input into a *prefix* of the
ayah (input fully consumed, ayah free to end anywhere). So an ayah of ANY length surfaces as
soon as its prefix is discriminative (early, before it finishes), while a short ayah falls
away once the input outgrows it (the extra input becomes insertion cost). "Commit on
divergence."

Why prefix alignment and not the matcher's `partial_candidates`: the latter takes the min
cost over the ayah's nodes and does NOT penalize a short ayah when the input runs past its
end, so short ayat match a tiny early decode at cost ~0 and produce false early commits.
Prefix alignment consumes the whole input, so those short ayat accrue cost and drop out.

Driving (see `live_detect.py --mode stream`): a growing audio buffer is decoded each hop and
fed here; commits use PER-HOP persistence (the same top-1 must hold margin >= T for K hops)
to filter the first-second transients. On ayah COMPLETION the caller resets the buffer to a
short tail so the next ayah decodes cleanly (single-ayah audio — avoids the growing-buffer
multi-ayah under-decode that the legacy `buffer` mode suffers). Model-independent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "matcher"))
from phoneme_matcher import CommitTracker, SequentialContext  # noqa: E402


def prefix_align(inp: list, ay: list) -> tuple[float, int]:
    """(normalized cost, matched ayah-prefix length) to turn `inp` into a PREFIX of `ay`.

    Full edit DP; the answer is min over k of edit(inp, ay[:k]) — the input is fully consumed,
    the ayah may end anywhere (unused suffix is free). Cost normalized by len(inp), so it's
    comparable across ayat and rises for an ayah shorter than the input."""
    n, L = len(inp), len(ay)
    if n == 0:
        return 0.0, 0
    prev = list(range(L + 1))                      # D[0][k] = k
    for i in range(1, n + 1):
        ci = inp[i - 1]
        cur = [i]
        for k in range(1, L + 1):
            cur.append(min(prev[k] + 1, cur[k - 1] + 1, prev[k - 1] + (ci != ay[k - 1])))
        prev = cur
    best_k = min(range(L + 1), key=lambda k: prev[k])
    return prev[best_k] / n, best_k


class StreamDetector:
    """Prefix-anchored streaming matcher over the current ayah's growing decode.

    `feed(phonemes)` takes the FULL greedy decode of the current ayah's audio so far and
    returns a status dict:
        ranked       : [(key, cost, progress)]      running top-k (context-biased)
        detected     : key | None                   committed if confident else top-1
        committed    : key | None                   the persistence-committed ayah
        progress     : 0..1                          how far through `detected`
        commit_event : None | {event, ayah, ...}     set on a NEW commit -> announce it
        boundary     : bool                          True once `detected` COMPLETES ->
                                                      caller resets the audio buffer
    """

    def __init__(self, trie, seq: SequentialContext, ayah_phonemes: dict, *,
                 threshold: float = 0.15, persistence: int = 2, revise: int = 4,
                 min_progress: float = 0.2, complete_cost: float = 0.30, len_tol: float = 0.6,
                 commit_cost_max: float = 0.35):
        self.seq = seq
        self._keys = list(ayah_phonemes)
        self._ph = [ayah_phonemes[k] for k in self._keys]
        self._len = [len(p) for p in self._ph]
        self.min_progress = min_progress
        self.complete_cost = complete_cost
        self.len_tol = len_tol
        self.commit_cost_max = commit_cost_max        # never commit an ambiguous/high-cost lead
        self._commit_cfg = (threshold, persistence, revise)
        self._new_ayah()

    def _new_ayah(self) -> None:
        self.tracker = CommitTracker(*self._commit_cfg)
        self._shown: str | None = None
        self._done: str | None = None

    def reset(self) -> None:
        """Full reset (new session): forget the committed ayah AND the context streak."""
        self.seq.set_current(None)
        self._new_ayah()

    def _kind(self, key: str) -> str:
        if self.seq.current is None:
            return "detect"
        i = self.seq._idx.get(self.seq.current)
        nxt = self.seq._order[i + 1] if i is not None and i + 1 < len(self.seq._order) else None
        return "advance" if key == nxt else "jump"

    def feed(self, phonemes: list[str]) -> dict:
        n = len(phonemes)
        scored = []                                   # (cost, key, progress)
        for key, ph, L in zip(self._keys, self._ph, self._len):
            if L < self.len_tol * n:                  # too short to explain the input as a prefix
                continue
            cost, bk = prefix_align(phonemes, ph)
            scored.append((cost - self.seq.bonus_for(key), key, bk / L))
        if not scored:
            return {"ranked": [], "detected": None, "committed": None,
                    "progress": 0.0, "commit_event": None, "boundary": False}
        scored.sort(key=lambda s: s[0])
        ranked = [(k, c, pr) for c, k, pr in scored[:3]]
        top, top_cost, top_prog = scored[0][1], scored[0][0], scored[0][2]
        margin = (scored[1][0] - top_cost) if len(scored) > 1 else 1.0

        # per-hop persistence: same top must hold margin>=T for K hops (filters transients),
        # gated to a plausible early candidate — enough of the ayah heard (min_progress) AND a
        # low absolute cost (a high-cost lead is an ambiguous fragment, not a real detection).
        eligible = top if (top_prog >= self.min_progress and top_cost <= self.commit_cost_max) else None
        self.tracker.update(eligible, margin)
        committed = self.tracker.committed
        detected = committed or eligible
        prog = next((pr for c, k, pr in scored if k == detected), 0.0) if detected else 0.0
        d_cost = next((c for c, k, pr in scored if k == detected), 1.0) if detected else 1.0

        commit_event = None
        if detected and detected != self._shown:
            self._shown = detected
            commit_event = {"event": self._kind(detected), "ayah": detected,
                            "committed": committed is not None, "cost": round(d_cost, 3)}

        boundary = False
        if detected and prog >= 0.9 and d_cost <= self.complete_cost and self._done != detected:
            self._done = detected
            self.seq.set_current(detected)            # next ayah expected = detected + 1
            self._new_ayah()
            boundary = True

        return {"ranked": ranked, "detected": detected, "committed": committed,
                "progress": prog, "commit_event": commit_event, "boundary": boundary}


def run_offline(audio, sr, decode_fn, trie, seq, ayah_phonemes, *,
                hop_s: float = 1.0, reset_tail_s: float = 0.3, min_speech_s: float = 0.5,
                on_status=None, **kw):
    """Drive a StreamDetector over a whole recording (for testing/analysis).

    `decode_fn(audio_buffer) -> list[phoneme]`. Grows a buffer, decodes it every `hop_s`, and
    on an ayah boundary resets the buffer to the last `reset_tail_s`. Returns the committed
    detect/advance/jump events with the time + progress at which each fired."""
    det = StreamDetector(trie, seq, ayah_phonemes, **kw)
    H, tail = int(hop_s * sr), int(reset_tail_s * sr)
    start, pos, events = 0, 0, []
    while pos < len(audio):
        pos = min(pos + H, len(audio))
        buf = audio[start:pos]
        if len(buf) < int(min_speech_s * sr):
            continue
        st = det.feed(decode_fn(buf))
        if on_status:
            on_status(pos / sr, buf, st)
        if st["commit_event"]:
            events.append({**st["commit_event"], "t": round(pos / sr, 2),
                           "progress": round(st["progress"], 2)})
        if st["boundary"]:
            start = max(start, pos - tail)
    return events
