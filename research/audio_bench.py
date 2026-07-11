"""Audio-level regression harness for the Chain detector — the test that exercises what
continuous_eval.py CANNOT: the rolling-audio buffer + VAD + decode quality, i.e. the regime
where the 22 s window CROWDS OUT short units (see research/CLAUDE.md "Rolling-window CROWDING").

It drives real audio through the full C++ Detector (`test_detector.exe`) and scores the emitted
ayah sequence against known truth. Two corpora:
  (A) COMPOSED held-out test-reciter streams (concatenated ayah clips, configurable inter-ayah
      gap + optional phone-channel augmentation) — the always-available REGRESSION net.
  (B) REAL pulled phone-session WAVs with hand-labeled truth — the FAILURE-REGIME net that
      professional reciters can't reproduce. Add each new pulled session to REAL_CASES.

Runs offline, here, repeatably. Composed WAVs are cached (gitignored). Detector configs are
compared as named ARMS (env hooks in test_detector: QR_COST / QR_VAD / QR_RESET_GAP / QR_SUBMIN /
QR_EARLY; streaming acoustics as extra argv).

  python research/audio_bench.py                          # base vs +chainVadReset, all cases
  python research/audio_bench.py --arms base,stream,hardsub --only real
  python research/audio_bench.py --only fix_78             # filter by case-name substring
"""
import argparse, os, subprocess, sys, random, wave
from concurrent.futures import ThreadPoolExecutor
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
# (B2) REAL user recordings from demo/test_fixtures (quiet-mic phone, truth from *_events.jsonl):
#      long ayat + medium + short runs — the coverage the crowding fix was NOT validated on.
FIXDIR = REPO / "demo/test_fixtures"
REAL_CASES += [
    ("fix_114_quiet",     FIXDIR / "user_114_quietmic.wav",              [(114, 1)]),
    ("fix_78_38_40_cont", FIXDIR / "user_78_38to40_naba_continuous.wav",   run_seq(78, 38, 40)),
    ("fix_78_38_40_contb", FIXDIR / "user_78_38to40_naba_continuous_b.wav", run_seq(78, 38, 40)),
    ("fix_78_40_long",    FIXDIR / "user_78_40_naba_long.wav",           [(78, 40)]),
    ("fix_85_12_16_cont", FIXDIR / "user_85_12to16_buruj.wav",             run_seq(85, 12, 16)),
    ("fix_98_1_3_paused", FIXDIR / "user_98_1to3_bayyinah_paused.wav",     run_seq(98, 1, 3)),
    ("fix_98_1_4_mixed",  FIXDIR / "user_98_1to4_bayyinah_mixed.wav",      run_seq(98, 1, 4)),
]
# (B3) Rescued pulled sessions (real/sessions/) with confirmed truths in real/labels.csv:
#      columns file,truth — truth is space-separated s:a tokens, ranges allowed (112:1-4).
def parse_truth(s):
    out = []
    for tok in s.split():
        sur, a = tok.split(":")
        a0, _, a1 = a.partition("-")
        out += [(int(sur), x) for x in range(int(a0), int(a1 or a0) + 1)]
    return out

LABELS = REALDIR / "labels.csv"
if LABELS.exists():
    for _, r in pd.read_csv(LABELS).iterrows():
        REAL_CASES.append((f"sess_{Path(r.file).stem.replace('session_', '')}",
                           REALDIR / "sessions" / r.file, parse_truth(r.truth)))

# Streaming acoustics (the default-build path). chainVadReset is a designed NO-OP in streaming
# (de-crowding is a re-decode technique; see research/CLAUDE.md 2026-07-11), so the "stream" arm
# measures the deployment default and the windowed arms measure the reset faithfully.
STREAM = [REPO / "export/onnx/stream_conv.onnx", REPO / "export/onnx/stream_encoder.int8.onnx"]

# Named detector-config arms (the taint-audit matrix). Every arm inherits QR_COST=0.45 +
# subMin=0.0 (the shipped phone config) unless overridden.
ARMS = {
    "base":    dict(),                                    # windowed, shipped phone config
    "vad":     dict(vad=True),                            # + chainVadReset (gap 4.0 default)
    "vad_g3":  dict(vad=True, env={"QR_RESET_GAP": "3.0"}),
    "stream":  dict(stream=True),                         # deployment default (streaming)
    "streamvad": dict(stream=True, vad=True),             # must equal "stream" (no-op check)
    "cost035": dict(env={"QR_COST": "0.35"}),
    "cost040": dict(env={"QR_COST": "0.40"}),
    "cost050": dict(env={"QR_COST": "0.50"}),
    "hardsub": dict(env={"QR_SUBMIN": "1.0"}),            # Phase-2 soft scoring OFF
    "noearly": dict(env={"QR_EARLY": "0"}),               # v11 early-prefix OFF
    # previous model generation (pre-RetaSy-cleanup retrain) — quiet-mic regime check
    "mic":     dict(model=REPO / "export/onnx/model_s123_mic_22s.int8.onnx"),
    # candidate-config combos (taint-audit winners stacked)
    "mic050":  dict(model=REPO / "export/onnx/model_s123_mic_22s.int8.onnx",
                    env={"QR_COST": "0.50"}),
    "micvad":  dict(model=REPO / "export/onnx/model_s123_mic_22s.int8.onnx", vad=True),
    "mic050vad": dict(model=REPO / "export/onnx/model_s123_mic_22s.int8.onnx",
                      env={"QR_COST": "0.50"}, vad=True),
    "cost050vad": dict(env={"QR_COST": "0.50"}, vad=True),
    # checkpoint-selection probe: final-epoch checkpoints (selection-by-val-PER is itself
    # part of the audited taint) — compare vs the val-selected "best" exports, with vadReset
    "lastmicvad":   dict(model=REPO / "export/onnx/model_last_mic_22s.int8.onnx", vad=True),
    "lastcleanvad": dict(model=REPO / "export/onnx/model_last_clean_22s.int8.onnx", vad=True),
    # best-of-both retrain (cleaned labels + junk-noise augmentation, 2026-07-11)
    "bob":    dict(model=REPO / "export/onnx/model_s123_bob_22s.int8.onnx"),
    "bobvad": dict(model=REPO / "export/onnx/model_s123_bob_22s.int8.onnx", vad=True),
    # mic model + mic streaming graphs (the post-revert default-SDK streaming config;
    # requires export/onnx/stream_* re-exported from best_s123_mic)
    "micstream": dict(model=REPO / "export/onnx/model_s123_mic_22s.int8.onnx", stream=True),
    # phase-3 concatenation-trained model (continuous-corpus fine-tune of best_s123_mic)
    "p31suf":   dict(model=REPO / "export/onnx/model_s123_p31_22s.int8.onnx",
                     env={"QR_SUFFIX": str(REPO / "export/onnx/model_s123_p31_5s.int8.onnx")}),
    "p3":       dict(model=REPO / "export/onnx/model_s123_p3_22s.int8.onnx"),
    "p3suf":    dict(model=REPO / "export/onnx/model_s123_p3_22s.int8.onnx",
                     env={"QR_SUFFIX": str(REPO / "export/onnx/model_s123_p3_5s.int8.onnx")}),
    "p3stream": dict(model=REPO / "export/onnx/model_s123_p3_22s.int8.onnx",
                     stream=(REPO / "export/onnx/p3_stream/stream_conv.onnx",
                             REPO / "export/onnx/p3_stream/stream_encoder.int8.onnx")),
    "p3lastsuf": dict(model=REPO / "export/onnx/model_s123_p3last_22s.int8.onnx",
                      env={"QR_SUFFIX": str(REPO / "export/onnx/model_s123_p3_5s.int8.onnx")}),
    "p3sufvad": dict(model=REPO / "export/onnx/model_s123_p3_22s.int8.onnx", vad=True,
                     env={"QR_SUFFIX": str(REPO / "export/onnx/model_s123_p3_5s.int8.onnx")}),
    # v13 fresh-context suffix decode (repetition suppression fix; 5s standalone decode/hop)
    "micsufvad": dict(model=REPO / "export/onnx/model_s123_mic_22s.int8.onnx", vad=True,
                      env={"QR_SUFFIX": str(REPO / "export/onnx/model_s123_mic_5s.int8.onnx")}),
    "micsuf":    dict(model=REPO / "export/onnx/model_s123_mic_22s.int8.onnx",
                      env={"QR_SUFFIX": str(REPO / "export/onnx/model_s123_mic_5s.int8.onnx")}),
    "micsuf7":   dict(model=REPO / "export/onnx/model_s123_mic_22s.int8.onnx",
                      env={"QR_SUFFIX": str(REPO / "export/onnx/model_s123_mic_7s.int8.onnx"),
                           "QR_SUFFIX_SEC": "7"}),
}

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

def detect(wav, arm):
    spec = ARMS[arm]
    env = dict(os.environ, QR_COST="0.45")
    env.update(spec.get("env", {}))
    if spec.get("vad"): env["QR_VAD"] = str(VAD)
    cmd = [str(BIN), str(spec.get("model", MODEL)), str(CONF), str(wav), "--chain"]
    if spec.get("stream"):
        # True -> the global (deployed-model) graphs; a tuple -> arm-specific graphs
        graphs = STREAM if spec["stream"] is True else list(spec["stream"])
        assert all(p.exists() for p in graphs), f"streaming graphs missing: {graphs}"
        cmd += [str(graphs[0]), str(graphs[1])]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    seq, prov = [], None
    for line in r.stdout.splitlines():
        if line.startswith("detected sequence:"):
            seq = [u.split("#")[0] for u in line.split(":", 1)[1].split()]
        elif line.startswith("provisional:"):        # trailing cold-start ACTIVE the user saw
            prov = line.split(":", 1)[1].strip()
    if prov and (not seq or seq[-1] != prov): seq.append(prov)
    return seq

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--arms", default="base,vad", help=f"comma list of {','.join(ARMS)}")
    ap.add_argument("--jobs", type=int, default=4, help="parallel detector processes")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    for a in arms: assert a in ARMS, f"unknown arm {a}"
    cases = []
    for (name, seq, gap, aug) in COMPOSED:
        if args.only in name: cases.append((name, compose(name, seq, gap, aug), truth_str(seq)))
    for (name, wav, seq) in REAL_CASES:
        if args.only in name and Path(wav).exists(): cases.append((name, wav, truth_str(seq)))
    print(f"{'case':26} {'truth':>5}" + "".join(f" {a:>18}" for a in arms))
    tot = {a: [0, 0] for a in arms}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:   # detector runs are subprocesses
        futs = {(name, a): ex.submit(detect, wav, a)
                for name, wav, truth in cases for a in arms}
        for name, wav, truth in cases:
            line = f"{name:26} {len(truth):>5}"
            for a in arms:
                hit, n, exact, tail = score(futs[(name, a)].result(), truth)
                tot[a][0] += hit; tot[a][1] += n
                line += f"   {hit}/{n} t{tail}/2 {'EXACT' if exact else '     '}"
            print(line, flush=True)
    print(f"{'TOTAL':26} {'':>5}" + "".join(
        f"   {tot[a][0]}/{tot[a][1]} ({100.0*tot[a][0]/max(1,tot[a][1]):.0f}%)".ljust(19) for a in arms))

if __name__ == "__main__":
    main()
