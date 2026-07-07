#!/usr/bin/env python3
"""
Find ambiguous ayat — groups of ayat too phoneme-similar to reliably tell apart on their
own — and classify whether sequential context can disambiguate each.

Corpus-agnostic: run it on the Juz Amma lexicon now and the full-Quran lexicon later.
It reuses the matcher's edit metric (plain normalized Levenshtein, 1/1/1) so "ambiguous"
matches what the runtime matcher actually confuses.

  python matcher/find_ambiguous.py                              # ayah level, tau 0.15
  python matcher/find_ambiguous.py --lexicon <path> --tau 0.12 --out <path>
  python matcher/find_ambiguous.py --units      # segment-level unit index (waqf segments
                                                # + unsegmented ayat) -> ambiguous_units.json

Output JSON — for every ambiguous ayah:
  confusable_with : the candidate set the highlighter would have to choose among
  predecessor / successor : neighbours in reciting order (within the surah)
  resolvable_by   : 'predecessor' | 'successor' | 'both' | 'none'
      * predecessor -> knowing the previous (confidently-detected) ayah pins this one
      * successor   -> WAIT for the next ayah, then it pins this one (retroactive highlight)
      * none        -> context can't break the tie; needs a manual/option fallback

This is the data the deferral + highlight logic consumes: when a detected ayah is
ambiguous, hold it, surface `confusable_with` as options, and resolve via the neighbour
that `resolvable_by` names.

Unit mode (--units) additionally emits `cross_parent` per ambiguous unit: whether any
confusable peer belongs to a DIFFERENT ayah. Within-parent confusions are harmless for
mushaf highlighting (the highlighted ayah is the same either way); only cross-parent
ones need deferral.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_LEXICON = REPO / "data" / "lang" / "ayah_phonemes.json"
DEFAULT_OUT = REPO / "data" / "lang" / "ambiguous_ayat.json"
SEGMENT_LEXICON = REPO / "data" / "lang" / "segment_phonemes.json"
UNITS_OUT = REPO / "data" / "lang" / "ambiguous_units.json"

try:
    from rapidfuzz.distance import Levenshtein as _RF
except ImportError:                                    # pragma: no cover
    _RF = None


def _lev(a: str, b: str, cutoff: int) -> int:
    """Levenshtein distance, early-exiting once it exceeds `cutoff` (returns cutoff+1)."""
    if _RF is not None:
        return _RF.distance(a, b, score_cutoff=cutoff)
    # Banded DP fallback (rapidfuzz not installed) — fine for small corpora.
    la, lb = len(a), len(b)
    if abs(la - lb) > cutoff:
        return cutoff + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = cur[0]
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ai != b[j - 1]))
            row_min = min(row_min, cur[j])
        if row_min > cutoff:
            return cutoff + 1
        prev = cur
    return prev[lb]


def _key(u: str) -> tuple[int, int, int]:
    """Sort key for 's:a' (ayah) or 's:a#NN' (waqf-segment unit) keys."""
    sa, _, seg = u.partition("#")
    s, a = sa.split(":")
    return int(s), int(a), int(seg) if seg else 0


def _parent(u: str) -> str:
    return u.partition("#")[0]


def _encode_all(raw: dict[str, str]) -> dict[str, str]:
    """Phonemes -> single chars so edit distance is character-level (fast, and matches
    the matcher's per-token metric)."""
    vocab: dict[str, str] = {}

    def encode(seq: str) -> str:
        out = []
        for ph in seq.split():
            c = vocab.get(ph)
            if c is None:
                c = chr(0x100 + len(vocab))       # private code point per phoneme
                vocab[ph] = c
            out.append(c)
        return "".join(out)

    return {sa: encode(seq) for sa, seq in raw.items()}


def load_lexicon(path: Path) -> dict[str, str]:
    return _encode_all(json.loads(path.read_text(encoding="utf-8")))


def load_units(seg_path: Path, ayah_path: Path) -> dict[str, str]:
    """The chain decoder's unit index: waqf segments + unsegmented ayat as single units."""
    seg_raw = json.loads(seg_path.read_text(encoding="utf-8"))
    raw = {k: v["phonemes"] for k, v in seg_raw.items()}
    parents = {_parent(k) for k in raw}
    ayah_raw = json.loads(ayah_path.read_text(encoding="utf-8"))
    raw.update({k: v for k, v in ayah_raw.items() if k not in parents})
    return _encode_all(raw)


def chain_neighbors(keys):
    """pred/succ along the recitation chain of units, within each surah: segment n-1/n+1
    inside an ayah, last/first segment across ayah boundaries. Reduces to ayah±1 for
    plain 's:a' keys (contiguous corpora)."""
    seq = sorted(keys, key=_key)
    pred: dict[str, str | None] = {}
    succ: dict[str, str | None] = {}
    for i, u in enumerate(seq):
        p = seq[i - 1] if i > 0 and _key(seq[i - 1])[0] == _key(u)[0] else None
        n = seq[i + 1] if i + 1 < len(seq) and _key(seq[i + 1])[0] == _key(u)[0] else None
        pred[u], succ[u] = p, n
    return pred, succ


def find_confusable(enc: dict[str, str], tau: float) -> dict[str, dict[str, float]]:
    """Each ayah -> {other_id: normalized_distance} for all others within `tau`.

    Length pruning: editNorm = dist / max(len) >= |len_a - len_b| / max(len), so any pair
    with too-different lengths can't be within tau. Sort by length and only compare within
    the feasible length window -> scales to the full corpus."""
    items = sorted(enc.items(), key=lambda kv: len(kv[1]))
    ids = [k for k, _ in items]
    strs = [v for _, v in items]
    lens = [len(v) for v in strs]
    n = len(ids)
    neigh: dict[str, dict[str, float]] = {k: {} for k in ids}

    for i in range(n):
        li = lens[i]
        # partners j>i have len_j >= li; feasible only while li/len_j >= 1-tau  (li>=1)
        max_lj = li / (1 - tau) if tau < 1 else float("inf")
        for j in range(i + 1, n):
            lj = lens[j]
            if lj > max_lj:
                break                                  # lengths only grow -> done with i
            m = lj                                     # lj >= li here
            cutoff = int(tau * m)
            d = _lev(strs[i], strs[j], cutoff)
            if d <= cutoff:
                nd = d / m
                neigh[ids[i]][ids[j]] = nd
                neigh[ids[j]][ids[i]] = nd
    return {k: v for k, v in neigh.items() if v}


def classify(neigh, pred_map, succ_map, unit_mode: bool = False):
    """For each ambiguous unit, find its chain neighbours and whether predecessor /
    successor uniquely pins it out of its confusable set (and is itself unambiguous)."""
    ambiguous = set(neigh)

    def reliable(u):                                   # a neighbour we can confidently detect
        return u is not None and u not in ambiguous

    out = {}
    for sa, others in neigh.items():
        cset = list(others)
        p, s = pred_map[sa], succ_map[sa]
        # predecessor resolves iff it exists, is unambiguous, and no confusable peer shares it
        by_pred = reliable(p) and all(pred_map.get(o) != p for o in cset)
        by_succ = reliable(s) and all(succ_map.get(o) != s for o in cset)
        resolvable = ("both" if by_pred and by_succ else
                      "predecessor" if by_pred else
                      "successor" if by_succ else "none")
        out[sa] = {
            "confusable_with": sorted(cset, key=_key),
            "max_norm_dist": round(max(others.values()), 4),
            "predecessor": p,
            "successor": s,
            "resolvable_by": resolvable,
        }
        if unit_mode:
            # within-parent confusions don't change the highlighted ayah — only
            # cross-parent ones need deferral at the highlight layer
            out[sa]["cross_parent"] = any(_parent(o) != _parent(sa) for o in cset)
    return out


def clusters(neigh):
    """Connected components of the confusability graph (for the summary)."""
    seen, comps = set(), []
    for start in neigh:
        if start in seen:
            continue
        stack, comp = [start], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(neigh.get(u, {}))
        comps.append(sorted(comp, key=_key))
    return sorted(comps, key=lambda c: (-len(c), _key(c[0])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    ap.add_argument("--tau", type=float, default=0.15,
                    help="max normalized edit distance to call two ayat confusable "
                         "(default 0.15 ~ the matcher's commit margin)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--units", action="store_true",
                    help="segment-level unit index (waqf segments + unsegmented ayat)")
    args = ap.parse_args()
    out_path = args.out or (UNITS_OUT if args.units else DEFAULT_OUT)

    if args.units:
        enc = load_units(SEGMENT_LEXICON, args.lexicon)
        lex_desc = f"{SEGMENT_LEXICON.relative_to(REPO)} + {args.lexicon.relative_to(REPO)}"
    else:
        enc = load_lexicon(args.lexicon)
        lex_desc = (str(args.lexicon.relative_to(REPO)) if args.lexicon.is_relative_to(REPO)
                    else str(args.lexicon))
    pred_map, succ_map = chain_neighbors(enc)
    neigh = find_confusable(enc, args.tau)
    ayat = classify(neigh, pred_map, succ_map, unit_mode=args.units)
    comps = clusters(neigh)

    by_res = {"both": 0, "predecessor": 0, "successor": 0, "none": 0}
    for v in ayat.values():
        by_res[v["resolvable_by"]] += 1
    n_exact = sum(1 for v in ayat.values() if v["max_norm_dist"] == 0.0)

    report = {
        "meta": {
            "lexicon": lex_desc,
            "tau": args.tau,
            "metric": "normalized Levenshtein (1/1/1), matcher-consistent",
            "rapidfuzz": _RF is not None,
            ("n_units" if args.units else "n_ayat"): len(enc),
        },
        "summary": {
            "n_ambiguous": len(ayat),
            "n_classes": len(comps),
            "n_exact_duplicate": n_exact,
            "resolvable_by": by_res,
        },
        "classes": [{"members": c, "size": len(c)} for c in comps],
        "ambiguous": {k: ayat[k] for k in sorted(ayat, key=_key)},
    }
    if args.units:
        n_cross = sum(1 for v in ayat.values() if v["cross_parent"])
        report["summary"]["n_cross_parent"] = n_cross
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    unit_word = "units" if args.units else "ayat"
    print(f"corpus       : {len(enc)} {unit_word}  ({lex_desc}, tau={args.tau}, "
          f"rapidfuzz={'yes' if _RF else 'no (DP fallback)'})")
    print(f"ambiguous    : {len(ayat)} {unit_word} in {len(comps)} classes "
          f"({n_exact} in exact-duplicate pairs)")
    print(f"resolvable by: predecessor {by_res['predecessor']}  successor {by_res['successor']}"
          f"  both {by_res['both']}  NONE {by_res['none']}")
    if args.units:
        print(f"cross-parent : {n_cross} / {len(ayat)} "
              f"(within-parent confusions don't move the highlighted ayah)")
    if by_res["none"]:
        n_none_cross = [k for k, v in ayat.items()
                        if v["resolvable_by"] == "none" and v.get("cross_parent", True)]
        print(f"  ⚠ context-insensitive (need option fallback / deeper N-back): "
              f"{by_res['none']}" + (f" ({len(n_none_cross)} cross-parent)" if args.units else ""))
        show = n_none_cross if args.units else \
            [k for k, v in ayat.items() if v["resolvable_by"] == "none"]
        for k in sorted(show, key=_key)[:20]:
            print(f"      {k}  <-> {', '.join(ayat[k]['confusable_with'][:6])}")
        if len(show) > 20:
            print(f"      ... and {len(show) - 20} more")
    print(f"wrote        : {out_path.relative_to(REPO) if out_path.is_relative_to(REPO) else out_path}")


if __name__ == "__main__":
    main()
