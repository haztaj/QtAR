#!/usr/bin/env python3
"""
Prefix-anchored streaming detection (`--mode stream`).

The sliding-window segmenter matches each fixed window as a WHOLE against WHOLE ayat, so a
long ayah (e.g. 78:40, 105 phonemes) is length-pruned out of every 4 s window and never
detected. This detector instead accumulates the decode of the recitation's audio and scores
each ayah by PREFIX ALIGNMENT -- the min cost to turn the input into a *prefix* of the ayah
(input fully consumed, ayah free to end anywhere). So an ayah of ANY length surfaces as soon
as its prefix is discriminative (early, before it finishes), while a short ayah falls away
once the input outgrows it (the extra input becomes insertion cost). "Commit on divergence."

Commit policy is RANK PERSISTENCE, not absolute cost: on a quiet mic the cost is high (~0.4-
0.6) with small margins to confusables, yet the correct ayah holds #1 for many hops. As the
buffer grows the top-1 hands off A -> A+1 -> A+2, so a continuous recitation is committed
ayah by ayah without resetting the buffer. See StreamDetector. Model-independent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "matcher"))
from phoneme_matcher import SequentialContext  # noqa: E402


def prefix_align(inp: list, ay: list) -> tuple[float, int]:
    """(normalized cost, matched ayah-prefix length) to turn `inp` into a PREFIX of `ay`.

    Full edit DP; the answer is min over k of edit(inp, ay[:k]) -- the input is fully consumed,
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


def _key_pair(sa: str) -> tuple[int, int]:
    s, a = sa.split(":")
    return int(s), int(a)


class StreamDetector:
    """Prefix-anchored streaming matcher over the growing decode of a recitation.

    `feed(phonemes)` takes the FULL greedy decode of the audio so far and returns a status
    dict: ranked [(key,cost,progress)], detected, committed, progress, commit_event, boundary.

    Commit policy -- RANK PERSISTENCE, not absolute cost. On a quiet mic the prefix-align cost
    is high (~0.4-0.6) with small margins to confusables, yet the correct ayah holds #1 for
    many hops; the stable rank is the reliable signal. So an ayah commits when it holds the top
    spot for K consecutive hops (with a plausible `min_progress` heard, and below a loose
    `commit_cost_max` to reject garbage). As the buffer grows the top-1 naturally hands off
    A -> A+1 -> A+2, so a continuous recitation is committed ayah by ayah WITHOUT resetting the
    buffer (resetting to a short tail decodes garbage). A **jump** (non-continuation) needs more
    persistence (`jump_persistence`); a **backward** step to an earlier ayah of the same surah
    is suppressed -- reciters go forward, so a backward lead is decode noise (kills the trailing
    resurgence after a surah ends). Sequential context biases the expected next ayah, further
    resisting those backward flickers.
    """

    def __init__(self, trie, seq: SequentialContext, ayah_phonemes: dict, *,
                 persistence: int = 3, jump_persistence: int = 5, min_progress: float = 0.15,
                 commit_cost_max: float = 0.55, len_tol: float = 0.6, **_ignored):
        self.seq = seq
        self._keys = list(ayah_phonemes)
        self._ph = [ayah_phonemes[k] for k in self._keys]
        self._len = [len(p) for p in self._ph]
        self.persistence = persistence
        self.jump_persistence = jump_persistence
        self.min_progress = min_progress
        self.commit_cost_max = commit_cost_max
        self.len_tol = len_tol
        self.reset()

    def reset(self) -> None:
        """New session: forget the committed ayah, the rank run, and the context streak."""
        self.seq.set_current(None)
        self._leader: str | None = None
        self._run = 0
        self._committed: str | None = None

    def _relation(self, key: str) -> str:
        cur = self.seq.current
        if cur is None:
            return "cold"
        if key == cur:
            return "current"
        i = self.seq._idx.get(cur)
        nxt = self.seq._order[i + 1] if i is not None and i + 1 < len(self.seq._order) else None
        if key == nxt:
            return "continuation"
        (cs, ca), (ks, ka) = _key_pair(cur), _key_pair(key)
        if ks == cs and ka < ca:
            return "backward"
        return "jump"

    def feed(self, phonemes: list[str]) -> dict:
        n = len(phonemes)
        scored = []                                   # (cost, key, progress)
        for key, ph, L in zip(self._keys, self._ph, self._len):
            if L < self.len_tol * n:                  # too short to explain the input as a prefix
                continue
            cost, bk = prefix_align(phonemes, ph)
            scored.append((cost - self.seq.bonus_for(key), key, bk / L))
        if not scored:
            return {"ranked": [], "detected": None, "committed": self._committed,
                    "progress": 0.0, "commit_event": None, "boundary": False}
        scored.sort(key=lambda s: s[0])
        ranked = [(k, c, pr) for c, k, pr in scored[:3]]
        top, top_cost, top_prog = scored[0][1], scored[0][0], scored[0][2]

        self._run = self._run + 1 if top == self._leader else 1     # rank persistence
        self._leader = top
        rel = self._relation(top)
        need = self.persistence if rel in ("cold", "current", "continuation") else self.jump_persistence
        eligible = (top_prog >= self.min_progress and top_cost <= self.commit_cost_max
                    and rel != "backward")

        commit_event = None
        if eligible and self._run >= need and top != self._committed:
            self._committed = top
            self.seq.set_current(top)                 # advance context (streak grows on continuation)
            self._run = 0
            kind = "detect" if rel == "cold" else ("advance" if rel == "continuation" else "jump")
            commit_event = {"event": kind, "ayah": top, "committed": True, "cost": round(top_cost, 3)}

        detected = self._committed or (top if top_prog >= self.min_progress else None)
        prog = next((pr for c, k, pr in scored if k == detected), 0.0) if detected else 0.0
        return {"ranked": ranked, "detected": detected, "committed": self._committed,
                "progress": prog, "commit_event": commit_event, "boundary": False}


def run_offline(audio, sr, decode_fn, trie, seq, ayah_phonemes, *,
                hop_s: float = 1.0, min_speech_s: float = 0.5, on_status=None, **kw):
    """Drive a StreamDetector over a whole recording (for testing/analysis).

    `decode_fn(audio_buffer) -> list[phoneme]`. Grows a buffer from the start and decodes it
    every `hop_s`. Returns the committed detect/advance/jump events with the time + progress at
    which each fired. (Live capture caps the buffer and can reset on a VAD pause; see
    live_detect.py.)"""
    det = StreamDetector(trie, seq, ayah_phonemes, **kw)
    H, pos, events = int(hop_s * sr), 0, []
    while pos < len(audio):
        pos = min(pos + H, len(audio))
        buf = audio[:pos]
        if len(buf) < int(min_speech_s * sr):
            continue
        st = det.feed(decode_fn(buf))
        if on_status:
            on_status(pos / sr, buf, st)
        if st["commit_event"]:
            events.append({**st["commit_event"], "t": round(pos / sr, 2),
                           "progress": round(st["progress"], 2)})
    return events
