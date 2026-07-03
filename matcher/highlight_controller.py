#!/usr/bin/env python3
"""
HighlightController — platform-agnostic ayah-highlight state machine (Stage-3, post-commit).

Sits ON TOP of the commit layer (matcher `CommitTracker` output). It consumes a stream of
*committed* ayah detections and produces render-ready `HighlightState` snapshots. Every
platform / SDK version just draws the snapshot — the deferral, the ambiguity handling and
the retroactive resolution live here once, not re-coded per UI.

Two jobs:

1. **Centralized highlight state.** One immutable snapshot per change:
       confirmed : [ayah keys, in confirm order]         -> highlighted, settled
       pending   : {ayah, options[], reason} | None       -> awaiting disambiguation
       active    : ayah key | None                        -> the ayah to emphasize now
   The UI renders this wholesale; no per-platform logic.

2. **Ambiguity deferral.** When the committed ayah belongs to a confusable class
   (from `data/lang/ambiguous_ayat.json`), we do NOT highlight a guess. Instead:
     - predecessor pins it (the prior confirmed ayah is unique to one option) -> confirm now;
     - else if a future ayah can pin it (successors are distinct) -> hold PENDING
       (reason='await_successor'); the next detection resolves it RETROACTIVELY;
     - else context can't help (e.g. 99:8, which ends its surah) -> PENDING
       (reason='needs_choice'); surface `options` for a manual/UI pick via `choose()`.

Corpus-agnostic: the confusable map comes from `find_ambiguous.py`, which runs on Juz Amma
now and the full Quran later. This controller reads that map and needs no code change.

    from matcher.highlight_controller import HighlightController
    hc = HighlightController()             # loads data/lang/ambiguous_ayat.json
    snap = hc.detect("78:1")               # -> HighlightState
    snap = hc.choose("99:8")               # resolve a needs_choice pending
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_AMBIG = REPO / "data" / "lang" / "ambiguous_ayat.json"


def _key(sa: str) -> tuple[int, int]:
    s, a = sa.split(":")
    return int(s), int(a)


# --------------------------------------------------------------------------- #
# The render-ready snapshot — the SDK's public output contract.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pending:
    ayah: str | None            # the resolved ayah once known, else None while deferred
    options: tuple[str, ...]    # the confusable set the UI would choose among
    reason: str                 # 'await_successor' | 'needs_choice'

    def to_dict(self) -> dict:
        return {"ayah": self.ayah, "options": list(self.options), "reason": self.reason}


@dataclass(frozen=True)
class HighlightState:
    confirmed: tuple[str, ...] = ()     # settled, highlighted — in confirm order
    pending: Pending | None = None      # awaiting disambiguation (deferred highlight)
    active: str | None = None           # the ayah to emphasize right now

    def to_dict(self) -> dict:
        return {
            "confirmed": list(self.confirmed),
            "pending": self.pending.to_dict() if self.pending else None,
            "active": self.active,
        }


class HighlightController:
    """Deterministic reference for the C++ `sdk/core` port (conformance-covered)."""

    def __init__(self, ambiguous_path: Path = DEFAULT_AMBIG):
        data = json.loads(Path(ambiguous_path).read_text(encoding="utf-8"))
        amb = data["ambiguous"]
        # class membership: detected key -> the full confusable set (incl. itself), sorted.
        self._class: dict[str, list[str]] = {}
        # per-member reciting-order neighbours (within surah), from the finder.
        self._pred: dict[str, str | None] = {}
        self._succ: dict[str, str | None] = {}
        for sa, info in amb.items():
            members = sorted({sa, *info["confusable_with"]}, key=_key)
            self._class[sa] = members
            self._pred[sa] = info["predecessor"]
            self._succ[sa] = info["successor"]
        self.reset()

    # -- lifecycle ----------------------------------------------------------- #
    def reset(self) -> None:
        self._confirmed: list[str] = []
        self._pending: Pending | None = None
        self._active: str | None = None

    def state(self) -> HighlightState:
        return HighlightState(tuple(self._confirmed), self._pending, self._active)

    def is_ambiguous(self, key: str) -> bool:
        return key in self._class

    # -- inputs -------------------------------------------------------------- #
    def detect(self, key: str, confidence: float = 1.0) -> HighlightState:
        """Feed a *committed* ayah detection. Returns the new snapshot."""
        # (1) A pending await_successor resolves the moment its successor is detected.
        if self._pending is not None and self._pending.reason == "await_successor":
            resolved = self._resolve_by_successor(key)
            if resolved is not None:
                self._confirm(resolved)             # retroactively settle the deferred ayah
            else:
                # reciter didn't continue as expected — the tie can't be broken by context.
                self._pending = Pending(None, self._pending.options, "needs_choice")

        # (2) Handle the newly detected ayah.
        if not self.is_ambiguous(key):
            self._confirm(key)
            return self.state()

        options = tuple(self._class[key])
        last = self._confirmed[-1] if self._confirmed else None
        pinned = self._resolve_by_predecessor(options, last)
        if pinned is not None:
            self._confirm(pinned)                    # predecessor uniquely identifies it
        elif self._successors_distinct(options):
            self._pending = Pending(None, options, "await_successor")   # defer — no guess
        else:
            self._pending = Pending(None, options, "needs_choice")      # manual fallback
        return self.state()

    def choose(self, key: str) -> HighlightState:
        """Manually resolve a `needs_choice` pending (host/UI picked an option)."""
        if self._pending is None or key not in self._pending.options:
            return self.state()
        self._confirm(key)
        return self.state()

    # -- internals ----------------------------------------------------------- #
    def _confirm(self, key: str) -> None:
        self._pending = None
        self._confirmed.append(key)
        self._active = key

    def _resolve_by_predecessor(self, options, last) -> str | None:
        if last is None:
            return None
        hits = [m for m in options if self._pred.get(m) == last]
        return hits[0] if len(hits) == 1 else None

    def _resolve_by_successor(self, detected) -> str | None:
        assert self._pending is not None
        hits = [m for m in self._pending.options if self._succ.get(m) == detected]
        return hits[0] if len(hits) == 1 else None

    def _successors_distinct(self, options) -> bool:
        """A future detection can disambiguate iff every option has a successor and they
        are all distinct (so exactly one option will match whatever comes next)."""
        succs = [self._succ.get(m) for m in options]
        return all(s is not None for s in succs) and len(set(succs)) == len(succs)


# --------------------------------------------------------------------------- #
# Self-test — drives the three resolution paths from the live ambiguity map.
# --------------------------------------------------------------------------- #
def _demo():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ambiguous", type=Path, default=DEFAULT_AMBIG)
    args = ap.parse_args()

    data = json.loads(args.ambiguous.read_text(encoding="utf-8"))
    amb = data["ambiguous"]
    hc = HighlightController(args.ambiguous)

    # pick one representative ayah per resolvable_by bucket
    picks: dict[str, str] = {}
    for sa, info in sorted(amb.items(), key=lambda kv: _key(kv[0])):
        picks.setdefault(info["resolvable_by"], sa)

    def show(label, snap: HighlightState):
        p = snap.pending
        ps = "None" if p is None else f"{{options={list(p.options)}, reason={p.reason}}}"
        print(f"  {label:<28} confirmed={list(snap.confirmed)}  pending={ps}  active={snap.active}")

    passed = True

    # 1) predecessor: confirm the prior ayah, then the ambiguous one -> immediate confirm.
    for bucket in ("predecessor", "both"):
        sa = picks.get(bucket)
        if not sa:
            continue
        pred = amb[sa]["predecessor"]
        hc.reset()
        hc.detect(pred)
        snap = hc.detect(sa)
        ok = snap.pending is None and snap.confirmed[-1] == sa
        passed &= ok
        print(f"\n[{bucket}]  detect {pred} then {sa}  -> {'PASS' if ok else 'FAIL'}")
        show(f"after {sa}", snap)

    # 2) successor: detect ambiguous first -> deferred; detect its successor -> retro-confirm.
    sa = picks.get("successor")
    if sa:
        succ = amb[sa]["successor"]
        hc.reset()
        s1 = hc.detect(sa)
        s2 = hc.detect(succ)
        ok = (s1.pending is not None and s1.pending.reason == "await_successor"
              and s1.active is None and sa in s2.confirmed and succ in s2.confirmed)
        passed &= ok
        print(f"\n[successor]  detect {sa} (defer) then {succ}  -> {'PASS' if ok else 'FAIL'}")
        show(f"after {sa}", s1)
        show(f"after {succ}", s2)

    # 3) none: unresolvable by context -> needs_choice; manual choose settles it.
    sa = picks.get("none")
    if sa:
        hc.reset()
        s1 = hc.detect(sa)
        pick = s1.pending.options[0] if s1.pending else sa
        s2 = hc.choose(pick)
        ok = (s1.pending is not None and s1.pending.reason == "needs_choice"
              and pick in s2.confirmed and s2.pending is None)
        passed &= ok
        print(f"\n[none]  detect {sa} -> needs_choice; choose {pick}  -> {'PASS' if ok else 'FAIL'}")
        show(f"after {sa}", s1)
        show(f"after choose", s2)

    print(f"\n{'ALL PASS' if passed else 'FAILURES'}  "
          f"({len(hc._class)} ambiguous ayat loaded from {args.ambiguous.name})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(_demo())
