#!/usr/bin/env python3
"""
Unified detection (`--mode auto`, the default).

Sliding and stream are two matchers, each reliable in a different regime and — crucially —
**silent in the other's**:

- `sliding` (whole-window edit distance over a fixed 4 s window) gets short back-to-back ayat
  and is *silent* on long ayat (a 4 s window of a 100+-phoneme ayah is length-pruned, never
  confident — it fires nothing, not garbage).
- `stream` (prefix-align over an anchored buffer + rank persistence) gets long/individual ayat
  and a partial, *agreeing* subset on short ones.

So they never conflict, and the union of their commits covers every ayah length. `AutoDetector`
runs both per hop (each on its own window + its own sequential context) and merges their commit
events into one deduplicated, ordered stream — no `--mode` choice needed. Validated on the
regression fixtures: 78:40 (long), 78:38→40 (long continuous), 85:12→16 (short continuous),
114:1→3 (short continuous) all come out right in this one mode.

Why not a single merged *scorer* (min of the two costs over one buffer): the difference is
buffer discipline, not the metric — short rapid ayat need stateless fixed windows, long ayat
need an anchored buffer, and one buffer can't be both (prototyped; it stayed stuck). Running
both keeps each discipline.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "matcher"))
from phoneme_matcher import SequentialContext  # noqa: E402
from sliding import SlidingWindowSegmenter      # noqa: E402
from streaming import StreamDetector            # noqa: E402

DEFAULT_CONTEXT = dict(bonus=0.22, window=2, surah_bonus=0.10, streak_bonus=0.05)


class AutoDetector:
    """Runs sliding + stream (independent contexts) and merges their commit events.

    `feed(slide_ph, t, stream_ph)` takes the decode of the fixed sliding window AND the decode
    of the stream's anchored buffer for this hop, and returns:
        commit  : None | {event, ayah, source}   the merged detection to announce
        refocus : None | float                    stream's window-tail signal (driver clips buf)
        ranked  : [(key, cost, progress)]         stream's running top-k (for display)
    """

    def __init__(self, trie, ayah_phonemes: dict, *, window_cost: float = 0.30,
                 recent: int = 6, context: dict | None = None, **stream_kw):
        ctx = context or DEFAULT_CONTEXT
        keys = list(trie.key_to_node.keys())
        self._order = sorted(keys, key=lambda k: tuple(int(x) for x in k.split(":")))
        self._idx = {k: i for i, k in enumerate(self._order)}
        self.slider = SlidingWindowSegmenter(None, SequentialContext(keys, **ctx),
                                             ayah_phonemes, max_cost=window_cost)
        self.stream = StreamDetector(trie, SequentialContext(keys, **ctx), ayah_phonemes,
                                     **stream_kw)
        self._recent_n = recent
        self.reset()

    def reset(self) -> None:
        self.slider.reset()
        self.stream.reset()
        self.emitted: list[str] = []
        self._recent: deque[str] = deque(maxlen=self._recent_n)

    def _kind(self, ayah: str) -> str:
        if not self.emitted:
            return "detect"
        last = self.emitted[-1]
        i = self._idx.get(last)
        nxt = self._order[i + 1] if i is not None and i + 1 < len(self._order) else None
        return "advance" if ayah == nxt else "jump"

    def _reconcile(self, cands: list[tuple[str, str, float]]):
        """cands: (source, ayah, cost). Emit a new ayah once; tiebreak simultaneous ones."""
        new = [c for c in cands if c[1] not in self._recent]   # dedup repeats + belated dupes
        if not new:
            return None
        if len(new) > 1:
            last = self.emitted[-1] if self.emitted else None
            i = self._idx.get(last) if last else None
            nxt = self._order[i + 1] if i is not None and i + 1 < len(self._order) else None
            conts = [c for c in new if c[1] == nxt]            # prefer the continuation
            pick = conts[0] if conts else min(new, key=lambda c: c[2])   # else lower cost
        else:
            pick = new[0]
        source, ayah, _ = pick
        ev = {"event": self._kind(ayah), "ayah": ayah, "source": source}
        self.emitted.append(ayah)
        self._recent.append(ayah)
        return ev

    def feed(self, slide_ph: list[str], t: float, stream_ph: list[str]) -> dict:
        sev = self.slider.process(slide_ph, t) if len(slide_ph) >= 3 else None
        sst = self.stream.feed(stream_ph)
        cands = []
        if sev:
            cands.append(("sliding", sev["ayah"], sev.get("cost", 1.0)))
        if sst["commit_event"]:
            ce = sst["commit_event"]
            cands.append(("stream", ce["ayah"], ce.get("cost", 1.0)))
        return {"commit": self._reconcile(cands), "refocus": sst["refocus"],
                "ranked": sst["ranked"]}


def run_offline(audio, sr, decode_fn, trie, ayah_phonemes, *,
                window_s: float = 4.0, hop_s: float = 1.0, min_speech_s: float = 0.5,
                on_status=None, **kw):
    """Drive an AutoDetector over a whole recording (for testing/analysis).

    Per hop, decodes BOTH the fixed `window_s` sliding window and the stream's anchored buffer,
    and advances the stream buffer start on `refocus`. Returns the merged commit events."""
    auto = AutoDetector(trie, ayah_phonemes, **kw)
    W, H = int(window_s * sr), int(hop_s * sr)
    stream_start, pos, events = 0, 0, []
    while pos < len(audio):
        pos = min(pos + H, len(audio))
        stream_buf = audio[stream_start:pos]
        if len(stream_buf) < int(min_speech_s * sr):
            continue
        slide_ph = decode_fn(audio[max(0, pos - W):pos])
        st = auto.feed(slide_ph, pos / sr, decode_fn(stream_buf))
        if on_status:
            on_status(pos / sr, st)
        if st["commit"]:
            events.append({**st["commit"], "t": round(pos / sr, 2)})
        if st["refocus"]:
            stream_start = max(stream_start, pos - int(st["refocus"] * sr))
    return events
