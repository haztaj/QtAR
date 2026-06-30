# QtAR — Quran Ayah Detection

On-device, real-time detection of **which Quranic ayah is being recited**, from a live
mobile microphone — identified as early as possible, before the ayah finishes. Fully
offline. Target users include learners and non-native reciters on ordinary phone mics.

**MVP scope:** Juz Amma (surahs 78–114), Hafs ʿan ʿAsim. Output is a `surah:ayah` ID —
not transcription, not tajweed scoring.

> This README is the project's front door. Each subsystem has a nested `CLAUDE.md` with
> details and gotchas; `docs/` holds the SDK and mobile-audio recommendations.

## How it works — two-stage, closed-corpus

Framed as disambiguation among a finite set of known ayat (not open-vocabulary ASR), so
the neural net stays small and error-tolerant while an index "knows the text":

```
mic → VAD/energy gate → 16 kHz log-mel → streaming Emformer+CTC (int8)
    → phoneme posteriors → greedy decode → fuzzy phoneme matcher → ranked surah:ayah
```

- **Stage 1** — small streaming **Emformer + CTC** (~9.8M params, int8), 80-dim log-mel
  @ 16 kHz, outputs **phoneme** posteriors. Targets come from a deterministic Hafs G2P.
- **Stage 2** — incremental **fuzzy matcher**: a phoneme trie over all MVP ayat with
  insertion/deletion/substitution tolerance, a **sliding-window segmenter** for
  continuous (no-pause) recitation, a **sequential context** prior (after ayah X, expect
  X+1), and ayah-end detection.

> Note: the model/framework deviate from the original brief (plain PyTorch instead of
> icefall/k2; Emformer instead of Zipformer) — both user-signed-off. See `CLAUDE.md`.

## Results (current best model: `best_mic`)

End-to-end ayah-ID accuracy:

| Model | Clean test | Held-out learners |
|---|---|---|
| clean (60 ep) | 90.0% | 10.2% |
| + augmentation | 93.4% | 16.4% |
| + RetaSy learner adaptation | 95.0% | 58.4% |
| **+ RMS-norm + poor-mic aug** | **97.4%** | **66.9%** |

Continuous, no-pause recitation on a quiet mic is handled by sliding-window segmentation
(validated live). On-device footprint ≈ **16 MB** (model dominates; fixed regardless of
corpus size).

## Repository layout

| Path | What |
|---|---|
| `data/` | dataset prep: download, Hafs G2P, manifests, learner aya-ID derivation |
| `training/` | Emformer+CTC trainer (plain PyTorch), poor-mic augmentation |
| `matcher/` | phoneme trie, sliding-window segmenter, sequential context, commit policy |
| `demo/` | live mic detection (`--mode sliding`/`buffer`), session recorder + analyzer |
| `eval/` | end-to-end ayah-ID evaluation, commit-threshold tuning |
| `export/` | ONNX / int8 export, parity + RTF checks |
| `conformance/` | golden-fixture acceptance test for the native (C++) port |
| `sdk/` | C++ core + Android (`.aar` + Compose demo) scaffold |
| `docs/` | SDK architecture, mobile audio-capture recommendations |

## Setup

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130
# PyTorch 2.x + CUDA 13.0 (RTX 5080 / Blackwell). Drop the index-url for CPU/other CUDA.
```

Audio datasets, model checkpoints, and ONNX exports are **not** committed (see
`.gitignore`). To run end-to-end you regenerate the data and train (below), or supply a
checkpoint. The on-device SDK delivers the model via download-on-first-launch.

## Usage

```bash
# 1) Data pipeline (Juz Amma)  — see data/CLAUDE.md
python data/prepare.py            # download parquets + coverage report
python data/derive_aya_ids.py     # RetaSy learner clips → numeric aya IDs
python data/extract_audio.py      # parquet blobs → mp3 + manifest.csv
python data/build_manifests.py    # → lhotse manifests + ayah_text.json
python data/build_lexicon.py      # Hafs G2P → lang/ayah_phonemes.json + tokens.txt

# 2) Train  — see training/CLAUDE.md
python training/train.py --epochs 60                       # phase 1 (clean)
python data/extract_retasy.py && python data/make_phase2_splits.py
python training/train.py --epochs 30 --augment \
    --init-from training/exp/best.pt \
    --train-manifest data/raw/phase2/combined_train.csv --tag _mic   # phase 2 (learners + poor-mic)

# 3) Evaluate  — see eval/CLAUDE.md
python eval/evaluate.py --checkpoint training/exp/best_mic.pt --split test
python eval/evaluate.py --checkpoint training/exp/best_mic.pt --manifest data/raw/phase2/retasy_test.csv

# 4) Live demo (real mic; run in a terminal)  — see demo/CLAUDE.md
python demo/live_detect.py                 # sliding mode (default; continuous recitation)
python demo/live_detect.py --mode buffer   # legacy growing-buffer mode
python demo/analyze_session.py 4           # investigate the 4th detected ayah of a session

# 5) Export + conformance
python export/export_onnx.py --checkpoint training/exp/best_mic.pt
python conformance/generate.py && python conformance/verify.py
```

## Mobile SDK

Shipping as a cross-platform SDK (Android first, iOS later): a shared **C++ core** wrapped
by Kotlin / Swift APIs, with a native demo app. Design + decisions in
`docs/sdk-architecture.md`; the `conformance/` suite is the acceptance test that keeps the
C++ port numerically faithful to this Python reference.

**The C++ core is complete and validated end-to-end.** All stages — log-mel front-end, CTC
decode, fuzzy matcher + sliding segmenter, ONNX Runtime inference, and the streaming
detector orchestration — build via CMake and pass the conformance harness; fed the real
quiet-mic session in 100 ms chunks, the C++ pipeline reproduces `114:1 → 114:2 → 114:3`,
matching Python. The int8 model (15.2 MB, argmax-lossless) runs on every ORT CPU build.
Remaining: the Android `.aar` (JNI + Kotlin scaffold exists) + Compose demo, then iOS. See
`sdk/README.md`.

## Standing rules

- Hafs ʿan ʿAsim only; full Quran is out of scope until the Juz Amma MVP is solid.
- Never commit audio datasets or model checkpoints.
- Ayah IDs come from dataset fields / text joins — never hand-typed.

## Roadmap

Android `.aar` + Compose demo (C++ core done) · iOS XCFramework + SwiftUI · streaming ONNX
export · in-house learner audio for the long surahs · full-Quran scale-up (re-tune capacity
then).
