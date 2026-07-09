# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

On-device, real-time detection of *which* Quranic ayah is being recited from a live mobile microphone stream. Detection must happen as early as possible — before the ayah finishes — on modest mobile hardware, fully offline.

Output: a ranked `surah:ayah` ID (e.g. `78:1`). Not transcription. Not tajweed scoring.

MVP scope: **Juz Amma (surahs 78–114) + Al-Fātiḥah, Al-Baqarah, Āl-ʿImrān (surahs 1–3), Hafs
ʿan ʿAsim.** ⚠️ Expanded beyond Juz-Amma-only — user-signed-off 2026-07-05. Surahs 1–3 are fully
covered by the same 30-reciter quran-md-ayahs data (14,790 clips); surah 4 is skipped (only ayat
1–34 available). The Juz-Amma-only state is preserved at git tag `juz30-v1` + `backups/juz30/`.
Full Quran remains out of scope for now.

---

## Locked decisions — do not re-litigate without explicit user sign-off

| Decision | Value |
|---|---|
| Deployment | Fully on-device / offline. No server component. |
| Target speaker | App user reciting, including learners and non-native speakers. |
| Riwayah | Hafs ʿan ʿAsim only. |
| MVP scope | Juz Amma (78–114) + surahs 1–3. ⚠️ Expanded from Juz-Amma-only, user-signed-off 2026-07-05 (revert: tag `juz30-v1`). |
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
- **ONNX export working** (`export/onnx/`): fp32 43.8 MB / int8 15.2 MB, parity 1.5e-5,
  int8 argmax lossless, CPU RTF ~0.03. int8 = weight-only dynamic, MatMul-only (avoids
  `ConvInteger`; static QDQ tanks this transformer — see export/CLAUDE.md). Full-utterance
  (fixed 30 s window — Emformer's data-dependent masks block dynamic-T export).
- **4 s windowed export done** (`export/onnx/model_4s.int8.onnx`, 11 MB): `--fixed-frames
  416 --tag _4s`. The SDK feeds 4 s sliding windows, so this right-sizes the model — **RTF
  0.002 (~15× cheaper than the 30 s export), identical detections** (Emformer masks padding).
  The demo dev-bundles it. See export/CLAUDE.md.
- **Ambiguity handling + centralized highlight — done.** (1) `matcher/find_ambiguous.py`
  precomputes the corpus-agnostic confusable-ayah map (`data/lang/ambiguous_ayat.json`;
  Juz Amma: 26 ambiguous / 13 classes, only 99:8↔99:7 context-unresolvable). (2)
  `matcher/highlight_controller.py` (Stage-3) consumes committed detections and emits
  render-ready `HighlightState` snapshots — the **centralized output contract** (state
  snapshots, user-signed-off 2026-07-03) so platforms just render; **ambiguity is deferred,
  never guessed** (predecessor pins now / successor retro-confirms / else manual choose).
  Ported to `sdk/core/src/highlight.*` (C++ byte-identical, conformance-pinned in
  `golden/highlight/`), wired through `Detector::setHighlightCallback` + JNI + Kotlin
  `Listener.onHighlightState`; `.aar` builds + bundles the map. See matcher/CLAUDE.md,
  conformance/spec.md §Stage 3.
- **Identical-phoneme ayat — resolved (2026-07-03), no longer a separate item.** The ~5
  matcher-indistinguishable pairs are exactly the exact-duplicate classes in
  `ambiguous_ayat.json` (`82:13↔83:22`, `83:9↔83:20`, `83:23↔83:35`, `84:2↔84:5`,
  `109:3↔109:5`) and **all are context-resolvable** — the HighlightController pins the
  in-order ones via sequential context and defers/retro-confirms the successor-only one
  (`83:23↔83:35`). The only context-unresolvable Juz-Amma pair is `99:8↔99:7` (near-, not
  identical-, phoneme; `99:8` ends its surah) → `needs_choice` manual fallback, by design.
- **Android app — on-device functional (2026-07-05).** The Compose demo runs live on a real
  device (Samsung foldable). Landed this session:
  - **Auto mode is the C++-core default** (`Mode::Auto`): sliding + stream matchers merged,
    handling any ayah length. `demo/streaming.py` + `demo/auto.py` ported to
    `sdk/core/src/{stream,autodet}.cpp`.
  - **Silero VAD ported to the core** (`sdk/core/src/vad.*`, `silero_vad_16k_op15.onnx`, needs
    the 64-sample context prepend). A speech-END resets the buffer + matcher so paused
    ayah-by-ayah recitation segments cleanly (parity with `demo/live_detect.py`). Bundled in
    the `.aar`; reproduced by `conformance/generate.py`.
  - **Capture fix (critical):** model inference ran on the AudioRecord read loop, so stalls
    dropped ~30% of samples (holes → garbage detection) and a race crashed on Stop. `AudioCapture`
    now decouples read from inference (reader → queue → worker) and joins threads on stop.
  - **Two-phase highlight:** the detected ayah (lighter) + its same-surah successor (darker,
    "up next") revealed once the active ayah nears completion (`doneProgress` or the successor
    leads). `HighlightSnapshot.upNext` — added at the public-snapshot layer; the
    conformance-pinned HighlightController is untouched.
  - **Mushaf reader redesign:** top strip = surah name (`surah-name.ttf` `surahNNN`) + juz
    (`quran-common.ttf` `juzNNN`); Eastern-Arabic page number, odd→right/even→left; tap toggles
    a top panel (jump + debug) and bottom panel (start/stop). Page auto-fit margin widened
    (`FIT 0.90`) — `measureText` under-measures vs Compose RTL layout, overflowing the widest
    justified line on wide/landscape screens.
  - **Font packaging:** the 604 KFGQPC page fonts (~199 MB) are downloaded ONCE into external
    files (survives app updates) instead of shipping in the APK → beta APK **205 MB → 64 MB**;
    updates never re-ship the fonts. `MushafFonts` (hosted zip + sha256 + version); `-PbundleFonts`
    keeps them in for offline dev.
  - **Runtime debug (UI-controlled):** `Detector::setDebug` / `setDebugLogging` / `setRecording`
    gate all logcat + the session-WAV recorder, toggled from the app's debug panel (persisted).
    Debug instrumentation now lives in-tree, off by default — no more strip-before-commit.
- **Research-first pivot (2026-07-06, user-directed).** Methodology findings (grounded in
  practical use) outrank shipping UI. Two results so far: (1) **negative result** — no per-clip
  commit-margin policy fixes long-ayah false commits (4 policy families swept on cached margin
  trajectories; the ambiguity is structural: long shared prefixes only diverge late; 59.7% false
  commits on s1-3 at the old T=0.15/K=5). (2) **Waqf segmentation dataset built** — ayah text
  splits at waqf marks (standalone tokens in the Uthmani text), CTC forced alignment
  (`data/build_segments.py`, torchaudio forced_align + own model) yields 1,029 segment refs /
  30,870 aligned spans over 10,350 clips with 0 failures; median segment 8.5 s = the regime the
  matcher already handles at 97%. Splits audition-approved by ear. (3) **Segment-chain decoder
  validated on continuous streams (2026-07-07)** — sliding multi-scale matched-filter windows
  (each window scale only fires refs of its own length class) + 3-gram retrieval + infix
  scoring + successor votes + twin substitution + deferral assembly (2-deep pending buffer =
  junk tolerance 1). On 747 continuous 4-ayah test sequences (v12 config: + early-prefix +
  streak gating): aligned-hit 89.3%, unit SER 11.3%, ayah-chain SER 12.4%, exact sequences
  58.8%; sequential context resolves exact-twin units 64% → 79%. Iteration history (v1–v12,
  each step measured) in research/CLAUDE.md.
  **C++ port done (2026-07-07):** `sdk/core/src/chain.{h,cpp}` + `Mode::Chain` in the Detector,
  conformance-pinned (`golden/chain/`, spec.md §Stage 2b) + exact over 200 real streams.
  **On-device (2026-07-08):** the demo app runs Mode.CHAIN live (1,057-ayah corpus, 22 s
  `model_s123_22s.int8.onnx`, versioned asset extraction). Phone-mic decodes run ~30% PER →
  fire threshold is decode-quality-dependent (0.30 clean reference / 0.45 phone,
  `Config.chainCost`); v11 early-prefix firing (fires the expected unit from a 50% prefix
  match — faster AND more accurate: clean aligned-hit 87.5→91.4, exact 45.3→56.7);
  cold-start provisional highlight kills the 10-20 s first-detection dead window. Verified
  live tracking on surahs 2 and 111.
- **Mic-adaptation retrain — done (2026-07-08).** `best_s123_mic.pt`: phase-2 recipe (poor-mic
  augmentation + RetaSy) re-applied to the expanded corpus, fine-tuned from best_s123 over
  regenerated phase-2 splits (26,855 clips / 130.3 h; 92 held-out learner reciters). Val PER
  0.130 → **0.079**; held-out learners 48.0% → 66.0% top-1 on the *dirty* test (81.9% on the
  cleaned test — see the cleanup entry below; false commits 42% → 25%); clean test 95.9%. On the pulled phone session, the strict-threshold chain
  went from 1 recovered unit to 3. Deployed: `model_s123_mic_22s.int8.onnx` (13.4 MB, int8
  argmax lossless) bundled in the demo, asset version `best_s123_mic-22s-v1`. Training infra:
  audiomentations leaks ~8.5 GB/epoch in the MAIN process (epochs slow monotonically as the
  file cache dies) → `train.py --resume` (full optimizer state) + `train_supervisor.py`
  restart every epoch — flat ~600 s epochs. See training/CLAUDE.md.
- **RetaSy learner-data cleanup — done (2026-07-08).** Tool built (`data/retasy_flag.py` →
  `retasy_review.py` → `retasy_verdicts.json` merge; see data/CLAUDE.md). User by-ear review
  applied: 2,235 → 1,678 clips (557 junk dropped, 8 relabeled); cleaned learner test 530 clips
  / 57 reciters. **Key finding: the learner number was deflated ~16 pts by unjudgeable test
  clips.** `best_s123_mic` reads **66.0% on the DIRTY test but 81.9% on the CLEANED test** —
  same model, no retraining; the 66% counted silence/noise/mislabels as misses. Retrain on
  cleaned data (`best_s123_mic_clean.pt`) adds +1.9 → **83.8%** learner (clean test 96.0%, no
  regression). RetaSy is only ~4% of training so the retrain effect is small by construction;
  the cleanup's value is the honest eval. **Learner numbers on the cleaned test set are the
  reference going forward.** `best_s123_mic_clean` is deployed on the phone.
- **Android model delivery + UI — done (2026-07-08).** (a) **Manifest-driven model download**
  (`ModelManager`): the default APK ships model-free and fetches on first launch from a hosted
  manifest (`{version, url, sha256, description}`); a differing version is detected as a new
  release and downloaded without an app update (cached in external files, survives updates,
  sha256-verified, old pruned). `-PbundleModel` ships the model in the APK for a fully offline
  build (mirrors `-PbundleFonts`). Validated on-device via a local server (adb reverse): first
  install downloads, version bump re-downloads + prunes. (b) **Update is not silent** — a
  genuine update fires `Listener.onModelUpdated(version, description)` and the demo shows a
  "what's new" dialog from the manifest `description` (`./gradlew :demo:modelManifest
  -PmodelDesc=...`). (c) **Compact demo UI** — status + Start/Stop on one line; debug controls
  moved to a ☰ dropdown top-right; Jump-to-page is a filled button. See sdk/android/README.md.
- **Waqf-segment progress exposed (2026-07-08).** `Mode::Chain` detects at the waqf-segment
  level internally; the public `HighlightSnapshot` now surfaces it as `activeSegment` /
  `activeSegmentCount` ("part N of M" within the active ayah; 0 = non-Chain/none, 1 =
  unsegmented, N = split). Added at the detector/public-snapshot layer (the conformance-pinned
  HighlightController is untouched, like `upNext`); `UnitIndex::segCountOf`; wired C++ →
  JNI JSON → Kotlin `HighlightState`; demo status shows "· part N/M". Verified on the
  2:30–2:33 smoke (2:30 → 1/3·2/3·3/3, 2:31 → 1/1, 2:32/2:33 → N/2). Coverage note: only
  segmented ayat (currently 345) report N>1 — expected to grow with full-Quran coverage.
  (4) **Segment-level ambiguity map** —
  `find_ambiguous.py --units` → `data/lang/ambiguous_units.json`: 206 ambiguous units / 84
  classes, 96% context-resolvable, 8 structural `needs_choice` cases (2:134↔2:141, 3:1↔2:1,
  99:8↔99:7); all cross-parent. Near-twin substitution in the decoder measured neutral —
  the map's value is the deferral/highlight contract.
- **Corpus expanded + model retrained (2026-07-05/06).** Surahs 1-3 added (1,057 ayat, 153.9 h);
  `best_s123.pt` (combined val PER 0.130; s1-3 0.130 / juz 0.110). End-to-end test: s1-3 96.1%
  top-1, juz 95.3% (vs 97.4% juz-only baseline, with a doubled index). Training perf: the batch
  sampler now bounds B*T^2 (quad budget) — long-clip batches previously crossed the ~8 GB WDDM
  paging cliff and collapsed throughput ~25x. Juz-30 revert: tag `juz30-v1` + `backups/juz30/`.
- **Beta hosting — done (2026-07-08).** Model + manifest uploaded to the GitHub `model` release;
  the download build fetches + sha256-verifies end-to-end on-device (see the download-delivery
  entry above). The download build works for beta testers.
- **Roadmap (open):**
  - *Model / accuracy:* (1) **true streaming export** (`Emformer.infer` chunk-by-chunk — the
    highest-leverage item: ~4× cheaper/hop + lower latency + battery, the wearable path; blocked
    today by the Emformer's data-dependent masks, hence the fixed-window export). (2) **in-house
    learner collection for the long surahs** (the known data hole — RetaSy covers only short
    surahs; raises the learner ceiling). (3) **full-Quran corpus** (beyond the current 1,057 ayat;
    out of MVP scope but the north star).
  - *Product / deployment:* (4) **iOS wrapper** (C++ core is portable; a Swift API + JNI-equivalent
    bridge, not a re-implementation). (5) **release signing** (beta APK is debug-signed today; a
    wider/Play beta needs a release keystore + signingConfig — a permanent-identity decision).
    (6) **word-level segment highlighting** (light the active waqf segment's exact words; main work
    is the offline segment→word map — word-exact boundaries, ~85% correct-phrase, verifiable
    offline; see the assessment 2026-07-09).
  - *Research:* (7) **deeper N-back context** for the 8 structural `needs_choice` cases
    (2:134↔2:141 unit class). (8) **posterior-aware matching** for the <12-phoneme retrieval floor
    (2.6% of units with too few 3-grams to retrieve).
