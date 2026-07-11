"""Cut the aligned continuous recitations into <=28 s multi-ayah TRAINING WINDOWS — the
phase-3 concatenation-training examples (real continuous audio; the repetition-suppression
root fix — see research/CLAUDE.md).

Input:  data/raw/continuous/alignments/<reciter>/sNNN.csv  (align_continuous.py)
Output: data/raw/phase3/windows_train.csv + windows_eval.csv (eval-only reciters, e.g. the
        held-out test reciter yasser_ad_dussary — see sources/*.json _meta.eval_only)

Columns: path,start_s,end_s,duration,surah,ayah_from,ayah_to,n_ayat,n_ph,phonemes
`path` points at the .pcm sibling (extract_continuous_pcm.py — int16 mono 16 kHz); the
Dataset slices it via np.memmap at train time. No window WAVs are written.

Policy:
- Boundary reconciliation: adjacent refined bounds can overlap slightly (wasl joins) —
  the cut between ayah k and k+1 is the midpoint of (end_k, start_{k+1}).
- A window is a run of CONSECUTIVE ayat whose reconciled span fits MAX_S; slide the start
  by one ayah per emission (overlapping windows are distinct examples — different left
  contexts are exactly what phase-3 trains).
- Windows never include FLAGged ayat (LOW_SCORE / ROUGH_ONLY).
- Single-ayah windows are emitted only when the ayah alone exceeds PAIR_S (long ayat are
  already covered by the per-ayah corpus; here we want multi-ayah continuity).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "data/raw/continuous"
ALIGN = ROOT / "alignments"
OUTDIR = REPO / "data/raw/phase3"
MAX_S = 28.0        # window cap (training ceiling 30 s minus label/pad slack)
PAD_S = 0.15        # audio pad at window edges
PAIR_S = 20.0       # a lone ayah this long leaves no room for a partner -> skip singles


def eval_only_reciters() -> set[str]:
    out = set()
    for spec in (ROOT / "sources").glob("*.json"):
        meta = json.loads(spec.read_text(encoding="utf-8")).get("_meta", {})
        if meta.get("eval_only"):
            out.add(spec.stem)
    return out


def main():
    ayah_ph = json.loads((REPO / "data/lang/ayah_phonemes.json").read_text(encoding="utf-8"))
    evals = eval_only_reciters()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    writers = {}
    for name in ("train", "eval"):
        f = open(OUTDIR / f"windows_{name}.csv", "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(["path", "start_s", "end_s", "duration", "surah", "ayah_from", "ayah_to",
                    "n_ayat", "n_ph", "phonemes"])
        writers[name] = (f, w)

    stats = {"train": [0, 0.0], "eval": [0, 0.0]}
    for rdir in sorted(d for d in ALIGN.iterdir() if d.is_dir()):
        which = "eval" if rdir.name in evals else "train"
        for csv_path in sorted(rdir.glob("s*.csv")):
            surah = int(csv_path.stem[1:])
            pcm = ROOT / rdir.name / f"{csv_path.stem}.pcm"
            if not pcm.exists():
                continue                          # run extract_continuous_pcm.py first
            rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
            n = len(rows)
            ok = [r["flag"] == "OK" for r in rows]
            start = [float(r["start_s"]) for r in rows]
            end = [float(r["end_s"]) for r in rows]
            # reconciled cut points: cut[k] separates ayah k and k+1 (0-based)
            cut = [(end[k] + start[k + 1]) / 2 for k in range(n - 1)]
            w0 = lambda k: (start[k] if k == 0 else cut[k - 1])
            w1 = lambda k: (end[k] if k == n - 1 else cut[k])
            for i in range(n):
                if not ok[i]:
                    continue
                j = i
                while (j + 1 < n and ok[j + 1] and w1(j + 1) - w0(i) <= MAX_S):
                    j += 1
                span = w1(j) - w0(i)
                if j == i and (span < PAIR_S or span > MAX_S):
                    continue    # lone ayah: too short = no continuity value; too long = over cap
                ayat = [int(rows[k]["ayah"]) for k in range(i, j + 1)]
                ph = " ".join(ayah_ph[f"{surah}:{a}"] for a in ayat)
                a_s, b_s = max(0.0, w0(i) - PAD_S), w1(j) + PAD_S
                writers[which][1].writerow([
                    str(pcm), f"{a_s:.2f}", f"{b_s:.2f}", f"{b_s - a_s:.2f}",
                    surah, ayat[0], ayat[-1], len(ayat), len(ph.split()), ph])
                stats[which][0] += 1
                stats[which][1] += span
    for name, (f, _) in writers.items():
        f.close()
        cnt, sec = stats[name]
        print(f"windows_{name}.csv: {cnt} windows, {sec/3600:.2f} h")


if __name__ == "__main__":
    main()
