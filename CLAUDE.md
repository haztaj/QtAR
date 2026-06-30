# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

On-device, real-time detection of *which* Quranic ayah is being recited from a live mobile microphone stream. Detection must happen as early as possible — before the ayah finishes — on modest mobile hardware, fully offline.

Output: a ranked `surah:ayah` ID (e.g. `78:1`). Not transcription. Not tajweed scoring.

MVP scope: **Juz Amma only (surahs 78–114, Hafs ʿan ʿAsim)**. Full Quran is out of scope until the MVP works.

---

## Locked decisions — do not re-litigate without explicit user sign-off

| Decision | Value |
|---|---|
| Deployment | Fully on-device / offline. No server component. |
| Target speaker | App user reciting, including learners and non-native speakers. |
| Riwayah | Hafs ʿan ʿAsim only. |
| MVP scope | Juz Amma (surahs 78–114). |
| Training hardware | RTX 5080 (16 GB), Ryzen 9 9950X. |
| Architecture | Two-stage (streaming encoder + CPU fuzzy matcher). Do not swap. |
| Training framework | **Plain PyTorch** Zipformer+CTC on native Windows. ⚠️ Deviates from the original icefall/k2 plan — user-signed-off 2026-06-29 (k2 is Linux-only; avoids WSL). Export still targets sherpa-onnx / onnxruntime. |

---

## Architecture — two-stage, closed-corpus

Frame the problem as disambiguation among a **finite set of known ayat**, not open-vocabulary ASR. The neural net stays small and error-tolerant; "knowing the text" is offloaded to an index.

### Stage 1 — streaming acoustic encoder (`training/`, `export/`)
- Model: small streaming **Emformer** + CTC head (~10–30M params, int8). ⚠️ Was
  Zipformer in the original brief — switched to torchaudio Emformer, user-signed-off
  2026-06-29. Rationale: framework already left icefall (so no turnkey Zipformer
  export benefit remains), and this is an easy closed-corpus 564-ayah CTC task where
  encoder choice is not accuracy-critical. Emformer = far less code/risk, native
  torchaudio streaming (`forward` train / `infer` stream).
- Input: 80-dim log-mel spectrogram @ 16 kHz.
- Output: phoneme posteriors (token targets derived from deterministic Quranic G2P over diacritized text).
- Runtime packaging: sherpa-onnx (Android/iOS bindings).

### Stage 2 — incremental fuzzy matcher (`matcher/`)
- Trie/FST of all MVP ayat in phoneme space.
- Token-level beam with insertion/deletion/substitution penalties (learner-tolerance).
- Emits ranked `surah:ayah` candidates + confidence.
- Commits when top-candidate margin clears a threshold.
- Maintains a global entry beam for restart/jump detection.
- Early detection via prefix matching: shared-prefix ayat surface a candidate set and commit on divergence — never force a single early guess.

### Runtime pipeline
```
mic stream → Silero VAD gate → Stage 1 encoder → phoneme posteriors → Stage 2 fuzzy beam → ranked candidates → commit
```

---

## Data

### Primary (clean, professional)
- **`Buraaq/quran-md-ayahs`** (HuggingFace) — ~187k rows, 30 Hafs reciters, explicit `surah_id` + `ayah_id` fields (no derivation needed), multilingual text. 71 parquet shards (~34.9 GB total); Juz Amma lives in the final shards — script downloads backward from shard 70 until surahs 78–114 are fully covered.
  - Pre-downloaded parquets: place in `data/raw/quran-md-ayahs/` (flat) before running `prepare.py`.

### Learner / robustness
- **`RetaSy/quranic_audio_dataset`** (HuggingFace) — ~6.8k rows, 1287 reciters, 81 countries, Hafs. Has `Surah`/`Aya` fields + 6-class correctness.
  - Skews to short final surahs (Al-Ikhlas, Al-Falaq, An-Nas, Al-Kawthar, Al-Kafirun, Al-Asr) + Al-Fatiha; ~100 unique verses.
  - **Filter out non-Quran rows** (Adhan, adhkar) before any use.

### Known gap
Longer early-Juz-Amma surahs (An-Naba, An-Nazi'at, Abasa…) have **no learner audio**. Plan: ~2–3 h in-house learner collection + heavy augmentation. Validate learner-tolerance on RetaSy-covered short surahs first.

### Supplementary (verify license before use)
- `MohamedRashad/Quran-Recitations` (HF)
- OpenSLR SLR132

### Rejected datasets
- QDAT (tiny, single-verse tajweed only)
- Large Tarteel crowdsourced set (no confirmed public download)

**Never commit audio datasets to git.**

---

## Training (`training/`)

- Framework: **icefall / k2** (Zipformer recipe), output consumed by sherpa-onnx.
- Mixed precision + gradient accumulation (fits 16 GB VRAM).
- Two phases:
  1. Pretrain/fine-tune on clean professional audio.
  2. Domain-adapt with learner/non-native data + augmentation.
- Augmentation (simulate phone channel): phone-mic IRs, room RIRs, additive noise (MUSAN/household), codec round-trips (Opus/AAC/AMR), gain/clipping, SpecAugment.
  - Keep speed/tempo perturbation **mild** — madd (elongation) is phonemically meaningful.
- Optional: distill from a Quranic Whisper/wav2vec2 teacher. Whisper is **teacher/labeler only** — not the on-device model (not streaming).
- Default to **mixed precision**; target **int8 export from day one**.

---

## Export (`export/`)

- Primary: sherpa-onnx streaming Zipformer, int8 quantized, Android/iOS bindings.
- Alternatives: ONNX Runtime Mobile, TFLite with NNAPI/CoreML.

---

## Evaluation (`eval/`)

Key metrics:
- Top-1 / top-k ayah-ID accuracy.
- Time-to-detection (seconds and phoneme tokens before ayah end).
- False-commit rate (wrong ayah committed early).
- Robustness curves across SNR levels and proficiency levels.
- On-device RTF + peak memory on one low-end and one flagship device.
- Dedicated slice: mispronounced-but-correct vs genuinely-different-ayah.

---

## Standing rules

- Ayah IDs come directly from `surah_id`/`ayah_id` fields in quran-md-ayahs — never hand-typed, never derived.
- **Filter RetaSy non-Quran rows** before any training/eval use.
- Hafs ʿan ʿAsim only; do not add other riwayat until MVP is complete.
- Never commit audio datasets to git.
- Do not expand scope or swap the architecture without explicit user sign-off.

---

## Commands

Fill in as scripts are created:

```bash
# Environment setup
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130
pip install k2 --find-links https://k2-fsa.github.io/k2/cuda.html
# icefall is run directly from source — clone alongside this repo:
# git clone https://github.com/k2-fsa/icefall

# Data preparation (run order; see data/CLAUDE.md)
python data/prepare.py          # download parquets + coverage report
python data/derive_aya_ids.py   # RetaSy Arabic text -> numeric aya IDs
python data/extract_audio.py    # parquet blobs -> mp3 + manifest.csv
python data/build_manifests.py  # -> lhotse jsonl.gz + ayah_text.json
python data/build_lexicon.py    # G2P -> lang/ayah_phonemes.json + tokens.txt

# Training (see training/CLAUDE.md)
python training/train.py --smoke --num-workers 0   # verify GPU/AMP path
python training/train.py --epochs 60               # phase 1: clean audio
# phase 2: learner adaptation (augmentation + RetaSy). First build the splits:
python data/extract_retasy.py                      # RetaSy learner audio -> manifest
python data/make_phase2_splits.py                  # -> combined_train.csv + retasy_test.csv
python training/train.py --epochs 30 --lr 2e-4 --augment \
    --init-from training/exp/best.pt --train-manifest data/raw/phase2/combined_train.csv \
    --tag _phase2 --frame-budget 24000   # NOTE: --augment needs persistent_workers off (auto)

# Eval (see eval/CLAUDE.md)
python eval/evaluate.py --checkpoint training/exp/best_phase2.pt --split test
python eval/evaluate.py --checkpoint training/exp/best_phase2.pt --manifest data/raw/phase2/retasy_test.csv

# Evaluation
# TBD — see eval/CLAUDE.md

# Export (see export/CLAUDE.md)
python export/export_onnx.py --checkpoint training/exp/best_phase2.pt
#   -> export/onnx/{model.onnx, model.int8.onnx, tokens.txt}; parity + int8 + RTF

# Live mic demo (see demo/CLAUDE.md) — run in a real terminal
python demo/live_detect.py            # default mic; --list-devices to choose
```

---

## Status

- **Data pipeline: complete.** 16,920 clips / 30.8 h / 30 reciters extracted and
  manifested (24/3/3 reciter split). RetaSy learner aya IDs derived (84.8%).
  34-phoneme Hafs G2P + token table built over all 564 Juz Amma ayat. See
  `data/CLAUDE.md`.
- **Current best model: `training/exp/best_mic.pt`** (val PER 0.093). End-to-end ayah-ID:

  | model | clean test | held-out learners |
  |---|---|---|
  | clean (60ep) | 90.0% | 10.2% |
  | +augmentation | 93.4% | 16.4% |
  | +RetaSy adaptation (best_phase2) | 95.0% | 58.4% |
  | **+norm + poor-mic aug (best_mic)** | **97.4%** | **66.9%** |

  `best_mic` adds an RMS-normalized front-end (baked into `logmel_16k`) + stronger
  poor-mic augmentation. **`best_phase2.pt` is now inconsistent with the normalized
  front-end — use `best_mic.pt`** (commit threshold needs re-tuning for it; false-commit
  rose — see eval/tune_commit.py). Remaining learner gap is the known data hole (RetaSy
  only covers 13 short surahs; long early-Juz surahs have no learner audio).
- **Known issue — continuous-recitation segmentation.** With proper per-ayah windows the
  model detects even a quiet-mic continuous recitation correctly; but the streaming
  growing-buffer + completion-reset cuts boundaries wrong (completion fires before the
  ayah's audio ends on quiet input). Brief pauses between ayat work today (VAD segments
  each). Robust continuous segmentation (energy + matcher-guided, or streaming w/ CTC
  alignment) is the priority next step.
- **Commit policy tuned** (`matcher/CommitTracker`): persistence K is the lever, not
  the threshold. Default T=0.15/K=5. See matcher/CLAUDE.md.
- **ONNX export working** (`export/onnx/`): fp32 43.8 MB / int8 15.1 MB, parity 1.7e-5,
  int8 argmax lossless, CPU RTF ~0.03. Full-utterance (fixed 30 s window — Emformer's
  data-dependent masks block dynamic-T export). See export/CLAUDE.md.
- **Next options:** (1) **streaming export** (`Emformer.infer` + conv-cache — the
  real-time on-device path); (2) in-house learner collection for the long surahs
  (raises the learner ceiling); (3) the ~5 identical-phoneme ayat.
