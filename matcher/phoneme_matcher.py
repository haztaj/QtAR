#!/usr/bin/env python3
"""
Stage-2 incremental fuzzy matcher.

A trie over all MVP ayat in phoneme space + a streaming approximate-match beam
with insertion / deletion / substitution penalties (learner tolerance). Emits
ranked surah:ayah candidates as phonemes stream in.

Design notes
------------
- The trie merges shared prefixes, so prefix-overlapping ayat surface as one
  candidate set and only separate as the recitation diverges (early detection).
- A global entry beam (root re-seeded to cost 0 each step) lets a new ayah start
  at any input position -> restart / jump detection.
- Streaming approximate string matching: frontier maps trie-node -> best edit
  cost aligning the observed phonemes so far to the path reaching that node.
    insertion  = observed has an extra phoneme  -> stay at node, consume input
    substitution = observed != reference phoneme -> move to child, consume input
    deletion   = reference has a phoneme skipped -> move to child, no input
  Deletions are an epsilon-closure, bounded to MAX_DEL consecutive per step.

Independent of the acoustic model: feed it phoneme tokens (ground-truth from the
G2P, or later the greedy/posterior output of Stage 1).
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path

INF = float("inf")
REPO = Path(__file__).resolve().parent.parent
AYAH_PHONEMES = REPO / "data" / "lang" / "ayah_phonemes.json"


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------

class PhonemeTrie:
    """Array-backed trie. Node i: children[i]: {phoneme -> j}, keys[i], depth[i]."""

    def __init__(self):
        self.children: list[dict[str, int]] = [{}]
        self.keys: list[list[str]] = [[]]
        self.depth: list[int] = [0]
        self.key_to_node: dict[str, int] = {}   # ayah key -> its terminal node
        self.key_to_path: dict[str, list[int]] = {}  # ayah key -> nodes along its path
        self.root = 0

    def _new_node(self, depth: int) -> int:
        self.children.append({})
        self.keys.append([])
        self.depth.append(depth)
        return len(self.children) - 1

    def add(self, key: str, phonemes: list[str]) -> None:
        n = self.root
        path = []
        for d, p in enumerate(phonemes, 1):
            nxt = self.children[n].get(p)
            if nxt is None:
                nxt = self._new_node(d)
                self.children[n][p] = nxt
            n = nxt
            path.append(n)
        self.keys[n].append(key)
        self.key_to_node[key] = n
        self.key_to_path[key] = path

    @classmethod
    def from_ayah_phonemes(cls, mapping: dict[str, list[str]]) -> "PhonemeTrie":
        t = cls()
        for key, phons in mapping.items():
            t.add(key, phons)
        return t

    def num_nodes(self) -> int:
        return len(self.children)


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    key: str
    cost: float          # raw edit cost
    norm_cost: float     # cost / matched length
    ref_len: int
    progress: float = 1.0  # fraction of the ayah matched (1.0 for completed/terminal)


class PhonemeMatcher:
    def __init__(
        self,
        trie: PhonemeTrie,
        sub_penalty: float = 1.0,
        ins_penalty: float = 1.0,
        del_penalty: float = 1.0,
        beam_width: int = 600,
        max_del: int = 2,
        allow_restart: bool = True,
    ):
        self.t = trie
        self.SUB = sub_penalty
        self.INS = ins_penalty
        self.DEL = del_penalty
        self.beam = beam_width
        self.max_del = max_del
        self.allow_restart = allow_restart
        self.reset()

    def reset(self) -> None:
        # frontier: node -> best cost
        self.frontier: dict[int, float] = {self.t.root: 0.0}
        self.n_steps = 0   # input phonemes consumed (for completion sanity)

    @staticmethod
    def _relax(d: dict[int, float], k: int, v: float) -> None:
        if v < d.get(k, INF):
            d[k] = v

    def step(self, x: str) -> list[Candidate]:
        self.n_steps += 1
        children = self.t.children
        cur = self.frontier
        if self.allow_restart:
            # Global entry beam: a fresh ayah may start at this input position.
            if cur.get(self.t.root, INF) > 0.0:
                cur[self.t.root] = 0.0

        new: dict[int, float] = {}
        for idx, c in cur.items():
            # insertion: consume x, stay at node
            self._relax(new, idx, c + self.INS)
            # match / substitution: consume x, advance to child
            for p, ch in children[idx].items():
                self._relax(new, ch, c + (0.0 if p == x else self.SUB))

        # deletion epsilon-closure (advance in trie without consuming x), bounded
        for _ in range(self.max_del):
            for idx, c in list(new.items()):
                for p, ch in children[idx].items():
                    self._relax(new, ch, c + self.DEL)

        # prune to beam
        if len(new) > self.beam:
            keep = heapq.nsmallest(self.beam, new.items(), key=lambda kv: kv[1])
            new = dict(keep)
        self.frontier = new
        return self.candidates()

    def step_many(self, phonemes: list[str]) -> list[Candidate]:
        out: list[Candidate] = []
        for p in phonemes:
            out = self.step(p)
        return out

    def candidates(self, k: int = 5) -> list[Candidate]:
        cands: list[Candidate] = []
        for idx, c in self.frontier.items():
            for key in self.t.keys[idx]:
                ref_len = self.t.depth[idx]
                cands.append(Candidate(key, c, c / max(1, ref_len), ref_len))
        cands.sort(key=lambda x: (x.norm_cost, x.cost))
        return cands[:k]

    def partial_candidates(self, k: int = 5, min_progress: float = 0.0) -> list[Candidate]:
        """Rank ALL ayat by best partial match, not just completed ones.

        For each ayah, takes its deepest currently-active frontier node (= how far the
        input has matched into that ayah) and scores cost / matched-depth. Enables
        early detection: a long ayah ranks high from its first words, well before its
        terminal is reached. (The frontier self-filters — a shallow node after many
        input phonemes carries high insertion cost and gets beam-pruned — so deepest
        active node ≈ input position with low cost for plausible ayat only.)
        """
        out: list[Candidate] = []
        for key, path in self.t.key_to_path.items():
            deepest = 0
            best_norm = INF
            best_cost = INF
            for d, node in enumerate(path, 1):
                c = self.frontier.get(node)
                if c is not None:
                    deepest = d                       # progress = furthest matched
                    nc = c / d
                    if nc < best_norm:                # rank by BEST partial alignment,
                        best_norm, best_cost = nc, c  # not the deepest (a mispronounced
            if deepest == 0:                          # ending shouldn't sink the ayah)
                continue
            total = len(path)
            prog = deepest / total
            if prog < min_progress:
                continue
            out.append(Candidate(key, best_cost, best_norm, total, prog))
        out.sort(key=lambda c: (round(c.norm_cost, 6), -c.progress))
        return out[:k]

    def candidate_for(self, key: str) -> Candidate | None:
        """Current cost of a specific ayah (its terminal node in the frontier), or
        None if that node isn't currently active. Lets a context prior score the
        expected continuation even when it's outside the natural top-k."""
        node = self.t.key_to_node.get(key)
        if node is None:
            return None
        c = self.frontier.get(node)
        if c is None:
            return None
        ref_len = self.t.depth[node]
        return Candidate(key, c, c / max(1, ref_len), ref_len)

    def partial_for(self, key: str, min_progress: float = 0.0) -> Candidate | None:
        """Partial-match candidate for a specific ayah (its deepest active frontier
        node), or None. The partial analogue of candidate_for — lets the context prior
        score the expected continuation early, before its terminal is reached."""
        path = self.t.key_to_path.get(key)
        if not path:
            return None
        deepest = 0
        best_norm = INF
        best_cost = INF
        for d, node in enumerate(path, 1):
            c = self.frontier.get(node)
            if c is not None:
                deepest = d
                nc = c / d
                if nc < best_norm:
                    best_norm, best_cost = nc, c
        if deepest == 0:
            return None
        total = len(path)
        prog = deepest / total
        if prog < min_progress:
            return None
        return Candidate(key, best_cost, best_norm, total, prog)

    def ayah_progress(self, key: str, complete_cost: float = 0.40, min_input_frac: float = 0.7):
        """How far the recitation is through a specific ayah.

        Walks the ayah's trie path and finds the deepest node currently active in the
        frontier (≈ the recitation's position in that ayah). Returns
        (progress 0..1, terminal_norm_cost or None, complete bool). `complete` is True
        once the ayah's TERMINAL node is reached with norm-cost ≤ complete_cost AND the
        input has produced at least `min_input_frac` of the ayah's phonemes. The input
        guard prevents a sparse/quiet decode from "completing" a long ayah via deletions
        (the failure mode on a quiet mic). Content-based, independent of pauses.
        """
        path = self.t.key_to_path.get(key)
        if not path:
            return 0.0, None, False
        total = len(path)
        best_depth = 0
        for depth, node in enumerate(path, 1):
            if node in self.frontier:
                best_depth = depth
        terminal_cost = self.frontier.get(path[-1])
        term_norm = terminal_cost / total if terminal_cost is not None else None
        enough_input = self.n_steps >= min_input_frac * total
        complete = term_norm is not None and term_norm <= complete_cost and enough_input
        return best_depth / total, term_norm, complete

    def commit_margin(self) -> tuple[Candidate | None, float]:
        """Top candidate + margin (norm-cost gap to runner-up). Larger = more confident."""
        cands = self.candidates(k=2)
        if not cands:
            return None, 0.0
        if len(cands) == 1:
            return cands[0], INF
        return cands[0], cands[1].norm_cost - cands[0].norm_cost


class CommitTracker:
    """Revisable persistence-based commit policy over a matcher's streaming output.

    A single margin crossing is unreliable — early in a recitation a transient
    wrong candidate can briefly lead with a large margin. Requiring the SAME
    top-1 to hold margin >= threshold for `persistence` consecutive phonemes
    suppresses those transients. Tuned defaults (eval/tune_commit.py): T=0.15, K=5
    → ~86% commit accuracy on clean.

    Revisable: a commit can be changed if a DIFFERENT candidate holds the lead for
    `revise_persistence` steps (> persistence, hysteresis). So it locks on easily the
    first time but only switches an established commit on stronger/longer evidence —
    flexible early, stable once a sequence is going.
    """

    def __init__(self, threshold: float = 0.15, persistence: int = 5,
                 revise_persistence: int | None = None):
        self.threshold = threshold
        self.persistence = persistence
        self.revise_persistence = revise_persistence or (persistence + 3)
        self.reset()

    def reset(self) -> None:
        self._run = 0
        self._key: str | None = None
        self.committed: str | None = None

    def update(self, top, margin: float) -> str | None:
        """Feed the matcher's current (top, margin). `top` may be a Candidate, a plain
        ayah-key string, or None. Returns the committed key the step it (re)commits,
        else None. `self.committed` always holds the latest."""
        key = top.key if hasattr(top, "key") else top
        if key is not None and margin >= self.threshold:
            self._run = self._run + 1 if key == self._key else 1
            self._key = key
        else:
            self._run = 0
            self._key = None
            return None

        if self.committed is None:
            need = self.persistence
        elif self._key == self.committed:
            return None                      # already committed to this key
        else:
            need = self.revise_persistence   # changing an existing commit: higher bar

        if self._run >= need:
            self.committed = self._key
            return self.committed
        return None


class SequentialContext:
    """Sticky continuation prior across ayat. After ayah X is committed:

    - the next `window` ayat (canonical order — surah boundaries handled) get a strong
      cost bonus, decaying with distance;
    - all ayat of the CURRENT surah get a smaller bonus (resist jumping out of the surah);
    - a STREAK boost grows the bonus with each confirmed continuation, so once a few
      ayat have been recited in sequence the prior is hard to dislodge.

    Still soft — strong acoustic evidence can override and the revisable commit corrects
    a genuine jump (after which the streak resets).
    """

    def __init__(self, ayah_keys, bonus: float = 0.22, window: int = 2,
                 surah_bonus: float = 0.10, streak_bonus: float = 0.05, streak_cap: int = 4):
        self.bonus = bonus
        self.window = window
        self.surah_bonus = surah_bonus
        self.streak_bonus = streak_bonus
        self.streak_cap = streak_cap
        self.current: str | None = None
        self.streak = 0
        self._order = sorted(ayah_keys, key=lambda k: tuple(int(x) for x in k.split(":")))
        self._idx = {k: i for i, k in enumerate(self._order)}

    def set_current(self, key: str | None) -> None:
        if key is None:
            self.current, self.streak = None, 0
            return
        # streak grows only when this is the expected continuation of the old current
        if self.current is not None and self.current in self._idx:
            nxt_i = self._idx[self.current] + 1
            expected = self._order[nxt_i] if nxt_i < len(self._order) else None
            self.streak = min(self.streak + 1, self.streak_cap) if key == expected else 0
        self.current = key

    def _surah(self, key: str) -> int:
        return int(key.split(":")[0])

    def bonus_for(self, key: str) -> float:
        """Total prior bonus (subtracted from norm-cost) for a candidate ayah."""
        if self.current is None or self.current not in self._idx:
            return 0.0
        eff = self.bonus + self.streak_bonus * self.streak     # streak-boosted
        i0 = self._idx[self.current]
        b = 0.0
        for j in range(1, self.window + 1):                    # next-ayah window
            i = i0 + j
            if i < len(self._order) and self._order[i] == key:
                b = max(b, eff * (1 - (j - 1) / (self.window + 1)))
        if self._surah(key) == self._surah(self.current):      # same-surah stickiness
            b = max(b, self.surah_bonus)
        return b

    def expected_keys(self) -> list[str]:
        """Next-window keys to score explicitly even if outside the natural top-k."""
        if self.current is None or self.current not in self._idx:
            return []
        i0 = self._idx[self.current]
        return [self._order[i0 + j] for j in range(1, self.window + 1)
                if i0 + j < len(self._order)]

    def rerank(self, matcher: "PhonemeMatcher", k: int = 5, min_progress: float = 0.2):
        """Apply the sticky prior to PARTIAL candidates (early detection) and re-rank.
        Returns (ranked [(key, adj_cost, progress)], top_key, adj_margin)."""
        cands = {c.key: c for c in matcher.partial_candidates(k=max(k, 8), min_progress=min_progress)}
        for key in self.expected_keys():
            if key not in cands:
                c = matcher.partial_for(key, min_progress)
                if c is not None:
                    cands[key] = c
        if not cands:
            return [], None, 0.0
        adjusted = sorted((c.norm_cost - self.bonus_for(key), key, c.progress)
                          for key, c in cands.items())
        top_key = adjusted[0][1]
        margin = (adjusted[1][0] - adjusted[0][0]) if len(adjusted) > 1 else INF
        return [(key, ac, prog) for ac, key, prog in adjusted[:k]], top_key, margin


# ---------------------------------------------------------------------------
# Self-test / evaluation on ground-truth phonemes
# ---------------------------------------------------------------------------

def _load() -> dict[str, list[str]]:
    raw = json.loads(AYAH_PHONEMES.read_text(encoding="utf-8"))
    return {k: v.split() for k, v in raw.items()}


def _corrupt(phonemes: list[str], rate: float, vocab: list[str], rng) -> list[str]:
    """Simulate learner errors: random sub/ins/del at the given per-token rate."""
    out: list[str] = []
    for p in phonemes:
        r = rng.random()
        if r < rate / 3:                 # substitution
            out.append(rng.choice(vocab))
        elif r < 2 * rate / 3:           # deletion (skip)
            continue
        elif r < rate:                   # insertion
            out.append(rng.choice(vocab))
            out.append(p)
        else:
            out.append(p)
    return out


if __name__ == "__main__":
    import random

    ayah_ph = _load()
    trie = PhonemeTrie.from_ayah_phonemes(ayah_ph)
    print(f"Trie: {len(ayah_ph)} ayat -> {trie.num_nodes()} nodes")
    vocab = sorted({p for ph in ayah_ph.values() for p in ph})

    # 1) Exact match: every ayah should rank #1 at cost 0.
    exact_ok = 0
    for key, phons in ayah_ph.items():
        m = PhonemeMatcher(trie, allow_restart=False)
        cands = m.step_many(phons)
        if cands and cands[0].key == key and cands[0].cost == 0.0:
            exact_ok += 1
    print(f"Exact-match top-1: {exact_ok}/{len(ayah_ph)} "
          f"({exact_ok/len(ayah_ph):.1%})")

    # 2) Learner-error robustness (sampled).
    rng = random.Random(0)
    sample = rng.sample(list(ayah_ph), k=150)
    for rate in (0.10, 0.20, 0.30):
        t1 = t3 = 0
        for key in sample:
            corr = _corrupt(ayah_ph[key], rate, vocab, rng)
            m = PhonemeMatcher(trie, allow_restart=False)
            cands = m.step_many(corr)
            keys = [c.key for c in cands]
            t1 += key == (keys[0] if keys else None)
            t3 += key in keys[:3]
        print(f"  err={rate:.0%}: top1={t1/len(sample):.1%}  top3={t3/len(sample):.1%}")

    # 3) Early detection: how many phonemes until the true ayah enters top-3?
    rng = random.Random(1)
    sample = [k for k in rng.sample(list(ayah_ph), k=100) if len(ayah_ph[k]) >= 12]
    fracs = []
    for key in sample:
        phons = ayah_ph[key]
        m = PhonemeMatcher(trie, allow_restart=False)
        hit = None
        for i, p in enumerate(phons, 1):
            cands = m.step(p)
            if key in [c.key for c in cands[:3]]:
                hit = i
                break
        if hit:
            fracs.append(hit / len(phons))
    if fracs:
        print(f"\nEarly detection (clean): true ayah in top-3 after "
              f"{sum(fracs)/len(fracs):.0%} of phonemes on average "
              f"({len(fracs)}/{len(sample)} detected before end)")
