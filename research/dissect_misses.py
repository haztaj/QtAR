#!/usr/bin/env python3
"""
Dissect arm-B (segment index) parent-misses from the detection-unit ablation.

For every stream where the top-1 unit's parent != the true parent, categorize WHY:

  exact-twin   top-1 unit's phoneme reference is IDENTICAL to the truth's — the matcher
               cannot distinguish them by construction; sequential context / ambiguity
               deferral resolves these (same class as the known 82:13<->83:22 twins).
  near-twin    normalized edit distance between the two references < 0.20 — formulaic
               variants; context resolves most, margin policies must not over-commit.
  truth-close  truth ranked in top-3 with a small cost gap — margin-level confusion.
  other        truth ranked low / references dissimilar — decode-quality or alignment
               issues (the true residual error).

The effective context-resolved ceiling of the segment index is
  1 - (truth-low misses) / total.

  python research/dissect_misses.py
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "matcher"))

CACHE = REPO / "data" / "raw" / "segments" / "test_streams.pkl"


def edit(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def main():
    from phoneme_matcher import PhonemeTrie, PhonemeMatcher

    streams = pickle.loads(CACHE.read_bytes())
    ayah_ph = {k: v.split() for k, v in json.loads(
        (REPO / "data/lang/ayah_phonemes.json").read_text(encoding="utf-8")).items()}
    seg_raw = json.loads((REPO / "data/lang/segment_phonemes.json").read_text(encoding="utf-8"))
    seg_ph = {k: v["phonemes"].split() for k, v in seg_raw.items()}
    seg_text = {k: v["text"] for k, v in seg_raw.items()}
    segmented = {k.split("#")[0] for k in seg_ph}
    refs = dict(seg_ph)
    refs.update({k: v for k, v in ayah_ph.items() if k not in segmented})
    trie = PhonemeTrie.from_ayah_phonemes(refs)

    misses = []
    for meta in streams:
        true_key = f"{meta['key']}#{meta['seg_idx']:02d}"
        m = PhonemeMatcher(trie, allow_restart=False)
        for p in meta["phonemes"]:
            m.step(p)
        cands = m.candidates(k=5)
        if not cands or cands[0].key.split("#")[0] == meta["key"]:
            continue
        truth_rank = next((i for i, c in enumerate(cands) if c.key == true_key), None)
        misses.append({"meta": meta, "true_key": true_key, "cands": cands, "truth_rank": truth_rank})

    print(f"parent-misses: {len(misses)} / {len(streams)}")

    cats = Counter()
    examples = defaultdict(list)
    for x in misses:
        true_ref = refs[x["true_key"]]
        top = x["cands"][0]
        top_ref = refs[top.key]
        d = edit(top_ref, true_ref) / max(len(true_ref), 1)
        if d == 0.0:
            cat = "exact-twin"
        elif d < 0.20:
            cat = "near-twin"
        elif x["truth_rank"] is not None and x["truth_rank"] <= 2:
            cat = "truth-close (rank<=3)"
        else:
            cat = "other (truth low)"
        cats[cat] += 1
        if len(examples[cat]) < 4:
            examples[cat].append(
                f"    true {x['true_key']:>12} vs top1 {top.key:>12} "
                f"(refdist {d:.2f}, truth_rank {x['truth_rank']}, dur {x['meta']['dur']:.1f}s)\n"
                f"      text: {seg_text.get(x['true_key'], ayah_ph.get(x['true_key'], ''))[:60] if isinstance(seg_text.get(x['true_key']), str) else ''}")

    print("\ncategory breakdown:")
    for cat, n in cats.most_common():
        print(f"  {cat:24} {n:4}  ({n/len(misses):5.1%} of misses, {n/len(streams):5.2%} of all)")
    resolvable = cats["exact-twin"] + cats["near-twin"] + cats["truth-close (rank<=3)"]
    print(f"\ncontext/margin-resolvable: {resolvable}/{len(misses)} of misses")
    print(f"effective ceiling with context resolution: "
          f"{1 - cats['other (truth low)']/len(streams):.1%}")

    # duration correlation: are short segments the weak spot?
    import numpy as np
    miss_durs = [x["meta"]["dur"] for x in misses]
    all_durs = [m["dur"] for m in streams]
    print(f"\nmiss median duration {np.median(miss_durs):.1f}s vs corpus median {np.median(all_durs):.1f}s")
    short = sum(1 for d in miss_durs if d < 4)
    print(f"misses with dur < 4s: {short}/{len(misses)} ({short/len(misses):.1%})")

    print("\nexamples per category:")
    for cat in cats:
        print(f"  {cat}:")
        for e in examples[cat]:
            print(e)


if __name__ == "__main__":
    main()
