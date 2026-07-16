#!/usr/bin/env python3
"""
Faithful confirmation of the calib probe: sweep chainCost per session through the REAL
rolling-window Detector (test_detector), not the optimistic whole-stream infix. For each
labeled session, find the per-session best cost (most correct in-order truth units) and check
whether it clusters near the global 0.45 or genuinely diverges (per-user headroom).
"""
import sys, subprocess, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "sdk/build/cmake-linux/test_detector"
MODEL = REPO / "export/onnx/model_full_tu_22s.int8.onnx"
SUFFIX = REPO / "export/onnx/model_full_tu_5s.int8.onnx"
CONF = REPO / "conformance"
SESSDIR = REPO / "data/raw/audio_bench/real/sessions"
COSTS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]


def expand(s):
    out = []
    for part in str(s).split():
        if ":" not in part:
            continue
        sur, rng = part.split(":")
        if "-" in rng:
            a, b = rng.split("-"); out += [f"{sur}:{i}" for i in range(int(a), int(b) + 1)]
        else:
            out.append(f"{sur}:{rng}")
    return out


def score(detected, truth):
    """Correct in-order units (LCS-free greedy: walk truth, consume matching detections)."""
    d = list(detected); n = 0
    for t in truth:
        if t in d:
            d = d[d.index(t) + 1:]; n += 1
    return n


def run(wav, cost):
    env = {"QR_COST": str(cost), "QR_NORMRMS": "0.15", "QR_SUBMIN": "0.0",
           "QR_SUFFIX": str(SUFFIX), "PATH": "/usr/bin:/bin"}
    import os
    e = dict(os.environ); e.update(env)
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
            jobs.append((row["file"], wav, expand(row["truth"])))

    def work(j):
        f, wav, truth = j
        res = {}
        for c in COSTS:
            res[c] = score(run(wav, c), truth)
        return f, len(truth), res

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(work, jobs))

    hdr = f"{'session':<26}{'n':>3}  " + "".join(f"{c:>6}" for c in COSTS) + f"{'best':>7}"
    print(hdr); print("-" * len(hdr))
    tot = {c: 0 for c in COSTS}; N = 0; best_costs = []
    for f, n, res in results:
        N += n
        for c in COSTS:
            tot[c] += res[c]
        bc = max(COSTS, key=lambda c: res[c])
        best_costs.append(bc)
        cells = "".join(f"{res[c]:>6}" for c in COSTS)
        print(f"{f:<26}{n:>3}  {cells}{bc:>7}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<26}{N:>3}  " + "".join(f"{tot[c]:>6}" for c in COSTS))
    print(f"\nglobal-best single cost: {max(COSTS, key=lambda c: tot[c])} "
          f"({max(tot.values())}/{N})   |  cost 0.45 -> {tot[0.45]}/{N}")
    from collections import Counter
    print(f"per-session best-cost distribution: {dict(sorted(Counter(best_costs).items()))}")
    oracle = sum(max(res[c] for c in COSTS) for _, _, res in results)
    print(f"per-session ORACLE (best cost each): {oracle}/{N}  "
          f"vs best global {max(tot.values())}/{N}  -> per-user headroom {oracle - max(tot.values())} units")


if __name__ == "__main__":
    main()
