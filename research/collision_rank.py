#!/usr/bin/env python3
"""
Corpus collision ranking — the real basis for a misdetection blacklist.

For every decoder unit U (waqf segment or unsegmented ayah), count how many DIFFERENT ayah
contexts would spuriously fire it: i.e. contexts V (V's parent ayah != U's) where U is in V's
3-gram shortlist AND infix_norm(U, V) <= fire-cost — exactly the retrieve-then-score the decoder
does per window. Plus the BASMALA (recited before ~113 surahs, not a numbered ayah), which is the
single biggest multiplier (it is why 55:1 الرحمن misfires everywhere).

High collision count = misdetection magnet. This is the axis a length/word-count blacklist misses:
55:1 is 9 phonemes ("safe" by length) but collides across hundreds of contexts + the basmala.

Output: ranked list (worst first) to research/collision_rank.csv + top offenders to stdout.
Run: research/collision_rank.py   (CPU; a few minutes over the full corpus)
"""
import sys, json, csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "research"))
from chain_sliding import build_ngram_index, window_counts, _infix_norm  # noqa: E402

TH = 0.45            # fire-cost (phone regime)
LMAX = 30            # only units this short can realistically embed elsewhere (firing set)
RAW_TOP, NORM_TOP = 60, 20   # shortlist sizes (mirror windowBest)
BASMALA_SURAHS = 113         # surahs that open with a recited basmala (all but surah 9)


def main():
    units = {k: v.split() for k, v in
             json.load(open(REPO / "conformance/assets/unit_phonemes.json")).items()}
    ayat = {k: v.split() for k, v in
            json.load(open(REPO / "data/lang/ayah_phonemes.json")).items()}
    refs = {k: p for k, p in units.items() if len(p) <= LMAX}      # firing set
    idx = build_ngram_index(refs)
    print(f"units {len(units)} | firing set (<= {LMAX} ph) {len(refs)} | contexts {len(ayat)} + basmala")

    coll = defaultdict(int)                 # # distinct ayah contexts that misfire U
    surahs = defaultdict(set)               # distinct surahs collided into (breadth)
    examples = defaultdict(list)            # a few (context, cost) for interpretability

    def shortlist(cnt):
        byraw = [k for k, _ in cnt.most_common(RAW_TOP)]
        bynorm = sorted(cnt.keys(), key=lambda k: cnt[k] / len(refs[k]), reverse=True)[:NORM_TOP]
        return list(dict.fromkeys(byraw + bynorm))

    for i, (vkey, vph) in enumerate(ayat.items()):
        vsur = int(vkey.split(":")[0])
        cnt = window_counts(vph, idx)
        if not cnt:
            continue
        for u in shortlist(cnt):
            if u.split("#")[0] == vkey:                # same ayah -> not a misdetection
                continue
            c = _infix_norm(refs[u], vph)
            if c <= TH:
                coll[u] += 1
                surahs[u].add(vsur)
                if len(examples[u]) < 4:
                    examples[u].append((vkey, round(c, 2)))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(ayat)} contexts", flush=True)

    # basmala context (special: fires ~113x in real recitation)
    basmala = ayat["1:1"]
    bcnt = window_counts(basmala, idx)
    basmala_hit = {}
    for u in shortlist(bcnt):
        c = _infix_norm(refs[u], basmala)
        if c <= TH:
            basmala_hit[u] = round(c, 2)

    # rank: effective misfire contexts = distinct ayat + basmala multiplier
    def eff(u):
        return coll.get(u, 0) + (BASMALA_SURAHS if u in basmala_hit else 0)

    ranked = sorted(refs.keys(), key=lambda u: (eff(u), coll.get(u, 0)), reverse=True)

    # write full ranked CSV
    out = REPO / "research/collision_rank.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "phonemes", "n_ph", "ayah_collisions", "distinct_surahs",
                    "basmala_cost", "eff_misfire_contexts", "example_hits"])
        for u in ranked:
            if eff(u) == 0:
                continue
            w.writerow([u, " ".join(refs[u]), len(refs[u]), coll.get(u, 0), len(surahs.get(u, ())),
                        basmala_hit.get(u, ""), eff(u),
                        "; ".join(f"{k}={c}" for k, c in examples.get(u, []))])
    n_flagged = sum(1 for u in ranked if eff(u) > 0)
    print(f"\nwrote {out}  ({n_flagged} units with >=1 collision)")

    print(f"\n=== TOP 45 misdetection magnets (eff = distinct ayat + {BASMALA_SURAHS} if basmala) ===")
    print(f"{'unit':<12}{'ph':>3}{'ayat':>6}{'surahs':>7}{'basml':>7}{'eff':>6}   text/example")
    for u in ranked[:45]:
        bt = basmala_hit.get(u, "")
        ex = examples.get(u, [])[:2]
        print(f"{u:<12}{len(refs[u]):>3}{coll.get(u,0):>6}{len(surahs.get(u,())):>7}"
              f"{str(bt):>7}{eff(u):>6}   {' '.join(refs[u])[:22]:<22}  {ex}")

    # where does 55:1 land?
    if "55:1" in refs:
        rank55 = ranked.index("55:1")
        print(f"\n55:1 rank: #{rank55 + 1}/{n_flagged}  "
              f"(ayat {coll.get('55:1',0)}, surahs {len(surahs.get('55:1',()))}, "
              f"basmala {basmala_hit.get('55:1','-')}, eff {eff('55:1')})")


if __name__ == "__main__":
    main()
