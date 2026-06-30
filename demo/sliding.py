#!/usr/bin/env python3
"""
Sliding-window segmentation for continuous (no-pause) recitation.

Instead of one growing buffer (which the single-ayah model under-decodes past the
first ayah), slide a fixed window across the stream and decode each window — the model
decodes each ayah well within its own window. A small state machine assembles the
per-window detections into an ayah sequence, finding boundaries by CONTENT (which ayah
dominates the window) rather than by pauses. Bounded per-window cost; ~0.02 RTF.

`SlidingWindowSegmenter` is model-independent (feed it decoded phonemes per window);
`run_offline` drives it over a whole recording for testing/analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "matcher"))
from phoneme_matcher import SequentialContext  # noqa: E402


def _edit_norm(a: list, b: list) -> float:
    """Normalized edit distance between two phoneme lists (0=identical)."""
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


class SlidingWindowSegmenter:
    """Assemble per-window ayah detections into a sequence.

    Each window is classified by WHOLE-window normalized edit distance to each ayah
    (which ayah the window *contains*, robust to short-ayah fragments). A window only
    fires when its best ayah is confident (cost < `max_cost`); straddling/ambiguous
    windows (cost ~0.4-0.6) are ignored. The sequential context gives the expected next
    ayah a bonus so continuations win ties. State machine: confident window for current
    -> stay; for current+1 -> advance; for anything else -> jump (needs `jump_votes`).
    """

    def __init__(self, trie, seq: SequentialContext, ayah_phonemes: dict,
                 max_cost: float = 0.30, jump_votes: int = 2, len_tol: float = 0.6):
        self.seq = seq
        self.max_cost = max_cost
        self.jump_votes = jump_votes
        self.len_tol = len_tol
        self._keys = list(ayah_phonemes)
        self._ph = [ayah_phonemes[k] for k in self._keys]
        self._len = [len(p) for p in self._ph]
        self.current: str | None = None
        self._pending: str | None = None
        self._votes = 0

    def _window_best(self, phonemes):
        """Best (key, cost) by context-biased whole-window edit distance, length-pruned."""
        n = len(phonemes)
        best_key, best_cost = None, 1e9
        for key, ph, L in zip(self._keys, self._ph, self._len):
            if not (self.len_tol * n <= L <= n / self.len_tol):   # prune by length
                continue
            c = _edit_norm(phonemes, ph) - self.seq.bonus_for(key)  # context bonus
            if c < best_cost:
                best_cost, best_key = c, key
        return (best_key, best_cost) if best_key else None

    def _expected_next(self):
        i = self.seq._idx.get(self.current)
        if i is not None and i + 1 < len(self.seq._order):
            return self.seq._order[i + 1]
        return None

    def _vote(self, key, need):
        if key == self._pending:
            self._votes += 1
        else:
            self._pending, self._votes = key, 1
        return self._votes >= need

    def process(self, phonemes, t: float):
        """Feed one window's phonemes + its center time. Returns an event dict on a
        detection/advance/jump, else None."""
        if len(phonemes) < 3:
            return None
        best = self._window_best(phonemes)
        if best is None or best[1] > self.max_cost:
            return None
        key, cost = best

        def commit(kind):
            self.current = key
            self.seq.set_current(key)
            self._pending, self._votes = None, 0
            return {"event": kind, "ayah": key, "t": round(t, 2), "cost": round(cost, 2)}

        if self.current is None:                          # cold start: confident = detect
            return commit("detect")
        if key == self.current:                           # still on current ayah
            self._pending, self._votes = None, 0
            return None
        if key == self._expected_next():                  # confident continuation -> advance
            return commit("advance")
        if self._vote(key, self.jump_votes):              # unexpected -> jump (needs votes)
            return commit("jump")
        return None


def run_offline(audio, sr, decode_fn, ayah_phonemes, seq, window_s=4.0, hop_s=1.0, **kw):
    """Slide over a whole recording; returns the list of segmentation events.
    decode_fn(window_audio) -> list[phoneme]."""
    seg = SlidingWindowSegmenter(None, seq, ayah_phonemes, **kw)
    W, H = int(window_s * sr), int(hop_s * sr)
    events = []
    start = 0
    while start < len(audio):
        win = audio[start:start + W]
        if len(win) >= int(0.5 * sr):
            ev = seg.process(decode_fn(win), (start + len(win) / 2) / sr)
            if ev:
                events.append(ev)
        start += H
    return events
