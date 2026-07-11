"""Semi-automatic truth labeling for pulled phone-session WAVs (audio_bench corpus B3).

For each data/raw/audio_bench/real/sessions/*.wav not yet in labels.csv, runs the detector
in TWO loose configurations (windowed base + windowed VAD-reset, cost 0.50 — looser than the
shipped 0.45 so weak true units surface), unions the detections, and proposes a truth as
CONTIGUOUS per-surah ayah ranges (recitation practice is contiguous; range expansion covers
detector misses INSIDE a range). A duration sanity check (median reference recitation length
for the proposed range vs actual WAV length) flags under-coverage — the detector missing the
range's edges — for by-ear confirmation.

NOT circular by construction: the proposal is a coherent-range hypothesis verified by the
USER (who recited these), not the detector output verbatim; edge expansion is explicitly
flagged, never silently invented.

  python research/label_sessions.py            # writes real/labels_proposed.csv
  # user reviews/edits truths, then: confirmed rows -> real/labels.csv (file,truth)
"""
import json, os, subprocess, sys, wave
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "sdk/build/cmake/test_detector.exe"
MODEL = REPO / "export/onnx/model_s123_mic_clean_22s.int8.onnx"
CONF = REPO / "conformance"
VAD = CONF / "assets/silero_vad.onnx"
REALDIR = REPO / "data/raw/audio_bench/real"
SESS = REALDIR / "sessions"

def repair_headerless(p):
    """Interrupted session recordings miss the RIFF header (written on stop) but hold raw
    PCM16 mono 16k. Write a repaired sibling (.fixed.wav) once and label that instead."""
    out = p.with_suffix(".fixed.wav")
    if not out.exists():
        data = p.read_bytes()
        with wave.open(str(out), "w") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
            f.writeframes(data)
    return out

def wav_dur(p):
    with wave.open(str(p)) as f: return f.getnframes() / f.getframerate()

def detect(wav, vad, cost="0.50"):
    env = dict(os.environ, QR_COST=cost)
    if vad: env["QR_VAD"] = str(VAD)
    r = subprocess.run([str(BIN), str(MODEL), str(CONF), str(wav), "--chain"],
                       capture_output=True, text=True, env=env, timeout=600)
    for line in r.stdout.splitlines():
        if line.startswith("detected sequence:"):
            return [u.split("#")[0] for u in line.split(":", 1)[1].split()]
    return []

def main():
    manifest = pd.read_csv(REPO / "data/raw/audio/manifest.csv")
    # median professional recitation length per ayah (phone recitation is same order)
    med_dur = manifest.groupby(["surah_id", "ayah_id"]).duration.median()
    last_ayah = {}  # surah -> last ayah in corpus
    for k in json.load(open(REPO / "data/lang/ayah_phonemes.json", encoding="utf-8")):
        s, a = map(int, k.split(":")); last_ayah[s] = max(last_ayah.get(s, 0), a)

    done = set()
    labels = REALDIR / "labels.csv"
    if labels.exists(): done = set(pd.read_csv(labels).file)

    rows = []
    for wav in sorted(SESS.glob("*.wav")):
        if wav.name in done or wav.name.endswith(".fixed.wav"): continue
        if wav.read_bytes()[:4] != b"RIFF":
            wav = repair_headerless(wav)
            if wav.name in done: continue
        dur = wav_dur(wav)
        if dur < 3.0:
            rows.append(dict(file=wav.name, truth="", wav_s=round(dur, 1), ref_s=0,
                             flag="TOO_SHORT", detections=""))
            print(f"{wav.name}: {dur:.1f}s TOO_SHORT", flush=True); continue
        seen, order = set(), []
        for v in (False, True):
            for u in detect(wav, v):
                if u not in seen: seen.add(u); order.append(u)
        # contiguous per-surah ranges over the union
        by_surah = {}
        for u in order:
            s, a = map(int, u.split(":")); by_surah.setdefault(s, []).append(a)
        ranges, ref_s = [], 0.0
        for s, ayat in by_surah.items():
            a0, a1 = min(ayat), max(ayat)
            ranges.append(f"{s}:{a0}-{a1}" if a1 > a0 else f"{s}:{a0}")
            for a in range(a0, a1 + 1): ref_s += float(med_dur.get((s, a), 6.0))
        truth = " ".join(ranges)
        flag = "OK"
        if not ranges: flag = "NO_DETECTIONS"
        elif ref_s < 0.55 * dur: flag = "UNDER_COVERED"   # edges likely missed — expand by ear
        elif ref_s > 2.2 * dur: flag = "OVER_COVERED"     # junk detections — trim by ear
        rows.append(dict(file=wav.name, truth=truth, wav_s=round(dur, 1),
                         ref_s=round(ref_s, 1), flag=flag, detections=" ".join(order)))
        print(f"{wav.name}: {dur:.1f}s -> {truth or '(none)'} [{flag}]", flush=True)

    out = REALDIR / "labels_proposed.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out} ({len(rows)} sessions) — review truths, move confirmed rows to labels.csv")

if __name__ == "__main__":
    main()
