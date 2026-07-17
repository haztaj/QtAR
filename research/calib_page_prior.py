#!/usr/bin/env python3
"""
Validate the page-context prior (Config::chainPageBonus, off-page penalty) end-to-end on the
labeled real phone sessions. For each session, simulate "the user is viewing the page holding
these ayat" by setting the page context to the truth surah within +-5 ayat of the truth range
(a realistic page window — far-off twins like 2:275 while reciting 2:255 fall OFF-page). Compare
no-prior vs +prior: correct in-order truth units (should not drop) and spurious extra emissions
(should drop). Not a page->ayah map (that lives in the app); this proxies it for the harness.
"""
import sys, os, subprocess, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "sdk/build/cmake-linux/test_detector"
MODEL = REPO / "export/onnx/model_full_tu_22s.int8.onnx"
SUFFIX = REPO / "export/onnx/model_full_tu_5s.int8.onnx"
CONF = REPO / "conformance"
SESSDIR = REPO / "data/raw/audio_bench/real/sessions"
BONUS = 0.08
PAGE_PAD = 5


def expand(s):
    out = []
    for part in str(s).split():
        if ":" not in part:
            continue
        sur, rng = part.split(":")
        if "-" in rng:
            a, b = rng.split("-"); out += [(int(sur), i) for i in range(int(a), int(b) + 1)]
        else:
            out.append((int(sur), int(rng)))
    return out


def page_for(truth_units):
    """Realistic page window: truth surah, ayat within +-PAD of the truth span."""
    sur = truth_units[0][0]
    ays = [a for s, a in truth_units if s == sur]
    lo, hi = min(ays) - PAGE_PAD, max(ays) + PAGE_PAD
    return ",".join(f"{sur}:{a}" for a in range(max(1, lo), hi + 1))


def score(detected, truth_keys):
    d = list(detected); n = 0
    for t in truth_keys:
        if t in d:
            d = d[d.index(t) + 1:]; n += 1
    return n


def run(wav, page):
    e = dict(os.environ)
    e.update({"QR_COST": "0.45", "QR_NORMRMS": "0.15", "QR_SUBMIN": "0.0", "QR_SUFFIX": str(SUFFIX)})
    if page:
        e["QR_PAGE"] = page; e["QR_PAGEBONUS"] = str(BONUS)
    r = subprocess.run([str(BIN), str(MODEL), str(CONF), str(wav), "--chain"],
                       capture_output=True, text=True, env=e, timeout=300)
    m = re.search(r"detected sequence:(.*)", r.stdout)
    return m.group(1).split() if m else []


def main():
    import pandas as pd
    labels = pd.read_csv(REPO / "data/raw/audio_bench/real/labels.csv")
    jobs = []
    for _, row in labels.iterrows():
        wav = SESSDIR / row["file"]
        if wav.exists():
            tu = expand(row["truth"])
            jobs.append((row["file"], wav, tu, [f"{s}:{a}" for s, a in tu], page_for(tu)))

    def work(j):
        f, wav, tu, tkeys, page = j
        base = run(wav, None)
        prio = run(wav, page)
        return dict(f=f, n=len(tkeys), page=page,
                    base_seq=base, prio_seq=prio,
                    base_hit=score(base, tkeys), prio_hit=score(prio, tkeys),
                    base_extra=len(base) - score(base, tkeys), prio_extra=len(prio) - score(prio, tkeys))

    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(work, jobs))

    hdr = f"{'session':<26}{'n':>3}{'hit b/p':>9}{'extra b/p':>11}   note"
    print(hdr); print("-" * (len(hdr) + 20))
    tb = tp = eb = ep = N = 0
    for x in res:
        N += x["n"]; tb += x["base_hit"]; tp += x["prio_hit"]; eb += x["base_extra"]; ep += x["prio_extra"]
        note = ""
        if x["prio_hit"] < x["base_hit"]:
            note = f"REGRESS  base={x['base_seq']} prio={x['prio_seq']}"
        elif x["prio_extra"] < x["base_extra"]:
            note = f"junk-- ({x['base_extra']}->{x['prio_extra']})"
        print(f"{x['f']:<26}{x['n']:>3}{x['base_hit']:>4}/{x['prio_hit']:<4}"
              f"{x['base_extra']:>6}/{x['prio_extra']:<4}   {note}")
    print("-" * (len(hdr) + 20))
    print(f"{'TOTAL':<26}{N:>3}{tb:>4}/{tp:<4}{eb:>6}/{ep:<4}")
    print(f"\ntrue-unit hits: {tb} -> {tp}  ({'no change' if tb==tp else ('REGRESSION' if tp<tb else 'GAIN')})")
    print(f"spurious extra emissions: {eb} -> {ep}  (page prior should reduce)")


if __name__ == "__main__":
    main()
