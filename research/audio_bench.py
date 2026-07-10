"""Audio-level regression harness for the Chain detector — the test that exercises what
continuous_eval.py CANNOT: the rolling-audio buffer + VAD + decode quality, i.e. the regime
where the 22 s window CROWDS OUT short units (see research/CLAUDE.md "Rolling-window CROWDING").

It drives real audio through the full C++ Detector (`test_detector.exe`) and scores the emitted
ayah sequence against known truth. Two corpora:
  (A) COMPOSED held-out test-reciter streams (concatenated ayah clips, configurable inter-ayah
      gap + optional phone-channel augmentation) — the always-available REGRESSION net.
  (B) REAL pulled phone-session WAVs with hand-labeled truth — the FAILURE-REGIME net that
      professional reciters can't reproduce. Add each new pulled session to REAL_CASES.

Runs offline, here, repeatably. Composed WAVs are cached (gitignored). Compares detector configs
via env (QR_VAD toggles the experimental chainVadReset focused-window reset).

  python research/audio_bench.py                 # baseline vs +chainVadReset, all cases
  python research/audio_bench.py --only short     # filter by case-name substring
"""
import argparse, os, subprocess, sys, random, wave
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
for p in ("training", "matcher", "research"):
    sys.path.insert(0, str(REPO / p))

BIN = REPO / "sdk/build/cmake/test_detector.exe"
MODEL = REPO / "export/onnx/model_s123_mic_clean_22s.int8.onnx"
CONF = REPO / "conformance"
VAD = CONF / "assets/silero_vad.onnx"
BENCH = REPO / "data/raw/audio_bench"          # cached composed WAVs (gitignored)
BENCH.mkdir(parents=True, exist_ok=True)
RECITER = "warsh_husary"                        # held-out test reciter

# (A) COMPOSED cases: (name, [(surah,ayah),...], gap_s, augment)
def run_seq(surah, a0, a1): return [(surah, a) for a in range(a0, a1 + 1)]
COMPOSED = [
    ("short_112_114_cont",   run_seq(112,1,4)+run_seq(113,1,5)+run_seq(114,1,6), 0.05, True),
    ("short_112_114_paused", run_seq(112,1,4)+run_seq(113,1,5)+run_seq(114,1,6), 0.7,  True),
    ("short_105_108_paused", run_seq(105,1,5)+run_seq(106,1,4)+run_seq(107,1,7)+run_seq(108,1,3), 0.7, True),
    ("long_baqarah_1_5",     run_seq(2,1,5),  0.4, True),   # long ayat — VAD mid-ayah risk
    ("long_baqarah_253_257", run_seq(2,253,257), 0.4, True),
]
# (B) REAL pulled phone sessions (persistent gitignored corpus; add each new pulled session here):
#     (name, wav_path, truth-units)
REALDIR = BENCH / "real"
REAL_CASES = [
    ("real_112_114_paused", REALDIR / "scratch_paused.wav",
     run_seq(112,1,4)+run_seq(113,1,5)+run_seq(114,1,6)),
    ("real_112_114_cont",   REALDIR / "scratch_session.wav",
     run_seq(112,1,4)+run_seq(113,1,5)+run_seq(114,1,6)),
]
# WINDOWED decode (empty STREAM). Streaming baseline == windowed, but streaming + chainVadReset
# DIVERGES (the experimental streaming boundaryReset is not yet correct — 9/15 vs windowed 15/15
# on real_112_114_paused), so the harness uses windowed to measure the reset faithfully. Slower
# (~35 s/case; use --only to iterate). Fix streaming-reset before shipping chainVadReset.
STREAM: list = []

def truth_str(seq): return [f"{s}:{a}" for s, a in seq]

def compose(name, seq, gap_s, augment):
    out = BENCH / f"{name}.wav"
    if out.exists(): return out
    from data import load_wav_16k
    manifest = pd.read_csv(REPO / "data/raw/audio/manifest.csv")
    aug = None
    if augment:
        from augment import build_waveform_augment
        aug = build_waveform_augment(16000)
    parts = []
    for (s, a) in seq:
        row = manifest[(manifest.reciter_id == RECITER) & (manifest.surah_id == s) & (manifest.ayah_id == a)].iloc[0]
        w = load_wav_16k(row.path).numpy()
        if aug is not None:
            rid = f"{RECITER}_{s}_{a}"; random.seed(abs(hash(rid)) % (2**32)); np.random.seed(abs(hash(rid)) % (2**32))
            w = np.ascontiguousarray(aug(samples=w, sample_rate=16000))
        parts.append(w.astype(np.float32)); parts.append(np.zeros(int(gap_s * 16000), np.float32))
    x = np.clip(np.concatenate(parts), -1, 1)
    f = wave.open(str(out), "w"); f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
    f.writeframes((x * 32767).astype(np.int16).tobytes()); f.close()
    return out

def detect(wav, vad):
    env = dict(os.environ, QR_COST="0.45")
    if vad: env["QR_VAD"] = str(VAD)
    cmd = [str(BIN), str(MODEL), str(CONF), str(wav), "--chain"]
    if STREAM and all(p.exists() for p in STREAM): cmd += [str(STREAM[0]), str(STREAM[1])]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    for line in r.stdout.splitlines():
        if line.startswith("detected sequence:"):
            return [u.split("#")[0] for u in line.split(":", 1)[1].split()]
    return []

def lcs(a, b):
    m, n = len(a), len(b); dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            dp[i+1][j+1] = dp[i][j]+1 if a[i] == b[j] else max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def score(emitted, truth):
    ded = list(dict.fromkeys(emitted))
    hit = lcs(ded, truth)                       # aligned in-order recall
    tail = sum(1 for t in truth[-2:] if t in ded)  # last-2-unit recall (the crowding symptom)
    return hit, len(truth), (ded == truth), tail

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="")
    args = ap.parse_args()
    cases = []
    for (name, seq, gap, aug) in COMPOSED:
        if args.only in name: cases.append((name, compose(name, seq, gap, aug), truth_str(seq)))
    for (name, wav, seq) in REAL_CASES:
        if args.only in name and Path(wav).exists(): cases.append((name, wav, truth_str(seq)))
    print(f"{'case':24} {'truth':>5} {'baseline':>18} {'+chainVadReset':>18}")
    for name, wav, truth in cases:
        line = f"{name:24} {len(truth):>5}"
        for vad in (False, True):
            hit, n, exact, tail = score(detect(wav, vad), truth)
            line += f"   {hit}/{n} t{tail}/2 {'EXACT' if exact else '     '}"
        print(line)

if __name__ == "__main__":
    main()
