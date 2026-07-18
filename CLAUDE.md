# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

On-device, real-time detection of *which* Quranic ayah is being recited from a live mobile microphone stream. Detection must happen as early as possible — before the ayah finishes — on modest mobile hardware, fully offline.

Output: a ranked `surah:ayah` ID (e.g. `78:1`). Not transcription. Not tajweed scoring.

Scope: **FULL QURAN — all 114 surahs, Hafs ʿan ʿAsim.** ⚠️ Expanded to the full Quran —
user-signed-off 2026-07-13. All 71 quran-md-ayahs shards are now downloaded and the full
Quran is complete in the data (187,080 clips, 30 reciters, all 6,236 ayat, zero coverage
gaps — the earlier surah-4 truncation was just the then-missing middle shards). Prior scope
milestones preserved: Juz-Amma-only at git tag `juz30-v1` + `backups/juz30/`; the 1-3 + Juz
Amma state was the 2026-07-05 expansion (git history around commit that added surahs 1-3).

---

## Locked decisions — do not re-litigate without explicit user sign-off

| Decision | Value |
|---|---|
| Deployment | Fully on-device / offline. No server component. |
| Target speaker | App user reciting, including learners and non-native speakers. |
| Riwayah | Hafs ʿan ʿAsim only. |
| Scope | **Full Quran (all 114 surahs, 6,236 ayat).** ⚠️ Expanded from 1-3 + Juz Amma, user-signed-off 2026-07-13 (prior milestones: tag `juz30-v1`, and the 2026-07-05 surahs 1-3 expansion). |
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
- **`Buraaq/quran-md-ayahs`** (HuggingFace) — ~187k rows, 30 Hafs reciters, explicit `surah_id` + `ayah_id` fields (no derivation needed), multilingual text. 71 parquet shards (~34.9 GB total). **All 71 shards are now downloaded (2026-07-13) and the full Quran is complete** (187,080 clips over all 6,236 ayat, verified by `prepare.py`'s per-surah completeness check — zero INCOMPLETE surahs).
  - Pre-downloaded parquets: place in `data/raw/quran-md-ayahs/` (flat) before running `prepare.py`. `prepare.py` / `extract_audio.py` scan every shard present on disk and slice `CORPUS_SURAHS = set(range(1,115))` (full Quran).

### Learner / robustness
- **`tusers` — full-Quran learner set (primary learner corpus).** Source:
  https://archive.org/download/quran-speech-dataset (licensing cleared, user-confirmed 2026-07-16).
  17,837 clips / **17,811 distinct learner voices** / 45.6 h, all 114 surahs, 93.2% of ayat
  (median 3 clips/ayah; ~1 clip per voice → huge *speaker* diversity, thin per-ayah). Manifest:
  `data/raw/tusers/tusers_filtered.csv` → `data/raw/phase2/tusers_manifest.csv`. **Closed the
  learner data hole** (all-surah coverage) — mixed at 8.5% into the phase-3 corpus to train the
  shipped `best_full_tu` (see Status). Audio not committed (see standing rules).
- **`RetaSy/quranic_audio_dataset`** (HuggingFace) — ~6.8k rows, 1287 reciters, 81 countries, Hafs. Has `Surah`/`Aya` fields + 6-class correctness.
  - Skews to short final surahs (Al-Ikhlas, Al-Falaq, An-Nas, Al-Kawthar, Al-Kafirun, Al-Asr) + Al-Fatiha; ~100 unique verses.
  - **Filter out non-Quran rows** (Adhan, adhkar) before any use.

### Known gap
The all-surah learner hole is **closed** by the `tusers` set above (all 114 surahs, 93.2% of ayat).
Residual: 425 ayat still have zero learner example and ~half the surahs are thin (<100 clips), but the
recognizer is in diminishing-returns territory on bulk learner data — the remaining lever is *targeted
real phone-on-stand continuous* capture for the far-field decode residual, not volume (assessment 2026-07-16).

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

- **★ SHIPPED MODEL: `best_full_tu` — real-learner adaptation (2026-07-16).** Fine-tune of
  `best_full_p3` on the 17,837-clip `tusers` corpus (17,811 distinct learner voices, all 114
  surahs — see Data) mixed at 8.5% into the phase-3 manifest (`combined_train_tusers.csv`),
  8 epochs; best = epoch 7. **Strictly better on every axis vs `best_full_p3`:** audio_bench
  **141/151 (93%)** @ normRms 0.15 (was 137), clean test **95.8%**, val PER **0.0443** (best
  ever), suppression ratio 0.976. **Fixed all three failing real phone takes** — Al-Hijr now
  tracks 15:1-15:5 with the wrong-`2:1` fast-commit glitch GONE; An-Nahl complete 16:1-16:7;
  the quiet An-Nahl take recovers 16:1-16:3. Confirms the continuous-tracking wall is
  **decode-quality-limited and REAL learner data is the lever** (the earlier synthetic-aug
  retrain FAILED — see the quiet-mic entry). Deployed **model-only**: hosted
  `model_full_tu_{22s,5s}.int8.onnx` + re-exported streaming graphs, manifest
  **`best_full_tu-22s-v5`**; the app auto-downloads (no APK rebuild). RetaSy's narrow
  14-surah metric dipped to 74.9% — a slice artifact, do not chase it.
- **Page-context prior — SHIPPED (2026-07-17).** The app tells the detector which ayat are on
  the **currently-viewed page + the next** (`Detector::setPageContext`, pushed on every page
  flip from `MushafRepository.pageAyat`). Off-page units pay a cost penalty
  (`Config::chainPageBonus`, demo 0.08) in `windowBest`'s fire gate + blended selection, so
  on-page ayat win twin ambiguities and spurious jumps elsewhere in the Quran are suppressed —
  off-page still detects when the decode is clean (**soft prior, not a hard filter**). Measured
  on the 10 labeled real sessions: true-unit hits **40 -> 42** (recovered the hard 2:6-9 case by
  removing off-page competition), spurious emissions **6 -> 4**, ZERO regressions. Core defaults
  0 = off, so conformance stays byte-identical. `research/calib_page_prior.py`.
- **Collision blacklist — SHIPPED, toggleable (2026-07-18).** Misdetection magnets are ranked by
  **corpus collision count**, not length (`research/collision_rank.py`): for each unit, how many
  ayah contexts + the basmala retrieve-and-match it at <=0.45. The worst are short COMMON PHRASES,
  not the muqattaʿāt — كلّا (eff 823, ~700 ayat across 94 surahs + basmala), قل الله, بلى, وعد الله;
  **55:1 الرحمن ranks #26/3252** (basmala cost 0.11 -> fires before ~113 surahs, + 71 ayat). The
  `eff>100` set (110 units, 18 basmala-matchers) ships as `data/lang/short_unit_blacklist.json`
  and is **cold-fire-suppressed** in `windowBest`: such a unit fires only when the current page
  vouches for it or the voter already expects it (**context-confirm-only**, via the early-prefix
  path). `Config::chainBlacklistPath` + runtime `Detector::setBlacklistEnabled`; **default OFF in
  the core (conformance byte-identical), ON in the app with a "Collision blacklist" switch in the
  ☰ debug menu** for live A/B. Real-session sweep off->on: **40 -> 42 hits, zero regressions**.
  User-validated on-device 2026-07-18.
- **In-app launch calibration — NEGATIVE RESULT, do not build (2026-07-16).** Probed whether a
  first-launch "enrollment recite" could auto-tune `normRms`/`chainCost` per user
  (`research/calib_probe.py`, `calib_live_sweep.py`). **No exploitable per-user variance exists:**
  decode cost is nearly FLAT across normRms 0.10-0.25 (per-session best beats fixed 0.15 by
  +0.025 median = noise), and in the live rolling-window sweep **all 10 sessions share the same
  best chainCost (0.30) — per-user oracle 42/42 == best single global 42/42, zero headroom.**
  Also caught a regime trap: whole-stream infix cost (~0.19) is an optimistic LOWER BOUND on live
  fire cost, so a naive enrollment would set the threshold too tight. Caveat: one user's voice
  across 10 sessions (no labeled multi-user audio). **Side-finding (open):** with the better `tu`
  decode, the deployed `chainCost 0.45` looks LOOSER than optimal — 0.30 scored 42/42 vs 0.45's
  40/42 on this corpus. Needs a full audio_bench check (cost interacts with other knobs) before
  changing the global.
- **FULL-QURAN corpus expansion — data layer DONE (2026-07-13).** All 71 quran-md-ayahs
  shards downloaded; full Quran verified complete (187,080 clips / 30 reciters / all 6,236
  ayat, zero coverage gaps). Corpus scope constants moved to `set(range(1,115))` in
  prepare/extract_audio/build_manifests + the RetaSy scripts (RetaSy filter widened so
  surah-1/Al-Fatiha learner clips, 2,934, are no longer dropped). Regenerated: audio
  (155,370 new mp3, 187,080 total, ~34 GB), lhotse manifests (**984.4 h**; 149,664 train /
  18,708 val / 18,708 test; same 24/3/3 reciter holdout), `ayah_text.json` (6,236),
  `ayah_phonemes.json` + `tokens.txt` (6,236 ayat, 34/34 phonemes — no OOV; max 878 = 2:282),
  and `ambiguous_ayat.json` (**466 ambiguous ayat / 179 classes / 50 context-unresolvable**
  vs Juz Amma's 26/13/1 — the ambiguity/deferral load scales up as expected). **Deferred to
  the redeploy phase (coupled to a retrained model): segment refs (`segment_phonemes.json`),
  unit ambiguity map (`ambiguous_units.json`), forced-alignment spans (GPU).** NEXT (Phase B,
  needs GPU go-ahead): retrain on the full corpus, then re-export ONNX + rebuild chain/segment
  assets + re-host for Android (Phase C). Design watch-item: detection design (early-prefix,
  chain decoder, export window sizes) was tuned/validated at 1,057 ayat; the ~6x index needs
  re-validation via audio_bench.
- **Data pipeline (prior, Juz Amma): complete.** 16,920 clips / 30.8 h / 30 reciters extracted and
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
- **Short-unit rolling-window CROWDING — root-caused + fix prototyped (2026-07-11).** A live
  report (short-surah run 112->114 tracks then stalls, dropping tail ayat 114:5-6) traced NOT to
  a decoder-logic regression and NOT to decode quality (two wrong diagnoses — see
  research/CLAUDE.md) but to the **22 s rolling window crowding out short units**: `window_best`
  stops surfacing a short tail unit once the window is dominated by earlier content. **The
  continuous_eval benchmark + conformance goldens cannot see it** — both run on cached/synthetic
  phoneme streams, never re-windowing audio (a real measurement gap). Fix prototyped —
  **`Config.chainVadReset`** (off by default): a VAD speech-END drops the buffered ayah's
  audio/phonemes for a focused next-ayah window while keeping the voter/assembler chain context.
  Verdict via a new offline harness (`research/audio_bench.py`, drives real audio through the full
  Detector — the test `continuous_eval` structurally can't be): the BLUNT per-pause reset is a bad
  trade (rescues real-phone 11/15->15/15 but guts long ayat, Baqarah 1:5 3/5->1/5). **Fixed with a
  targeted gate — `Config.chainResetMaxGap` (default 4.0): reset ONLY when the pause closely follows
  a commit (short ayah just ended), NOT mid-long-ayah breath. Measured: real-phone +3/+4 (continuous
  15/15 exact), ZERO long-ayah regression, 1-unit clean cost — a real fix.** **Windowed-only
  (2026-07-11):** de-crowding is a re-decode technique — streaming decodes incrementally, so a
  boundary can't recover crowded tail phonemes (measured: no gain + slight harm; the earlier
  streaming divergence was a broken outBase time axis). `chainVadReset` is now a safe NO-OP in
  streaming (VAD gated to windowed chain); the download build gets the fix only when run windowed
  (trading the ~11x streaming RTF for the accuracy fix). Still OFF by default (needs on-device
  validation). Harness (two corpora: composed pro streams + real pulled phone WAVs) is the durable
  offline test. See research/CLAUDE.md "audio_bench.py".
- **Reassessment / taint audit (2026-07-11).** Audit of which findings rest on cached-phoneme-
  stream evidence (the blind spot the crowding episode exposed). Corpus B grown 2 → 16+ real WAVs
  (7 labeled `demo/test_fixtures` user recordings + 25 rescued pulled sessions, semi-auto labeled —
  `research/label_sessions.py`); `audio_bench.py` gained named ablation ARMS + parallel jobs;
  harness scoring now counts the trailing cold-start PROVISIONAL (what the user actually sees).
  Final verdicts (21 cases / 138 truth units; anchor = deployed config, 118/138 86%):
  (a) **model choice OVERTURNED — `best_s123_mic` beats deployed `best_s123_mic_clean` end-to-end
  (125 vs 118; better/equal on every hard case).** The clean retrain was selected on the cleaned
  learner test (+1.9) and never checked in the deployment regime — model swaps must now be gated
  by audio_bench. (b) **chainVadReset confirmed stronger than first measured** (126/138) and the
  **winning combo is `best_s123_mic` + cost 0.45 + vadReset = 129/138 (93%)**; cost 0.50 helps
  alone (+4) but ANTI-stacks with the reset. (c) chainCost 0.45 + early-prefix SURVIVE the audit;
  **chainSubMin 0.0 is NEUTRAL end-to-end** (the noisy-cache +1.7 didn't transfer). (d)
  **streaming ≠ windowed on hard audio** — "identical detections" held only on clean clips;
  streaming wins long quiet-mic cases, loses slightly on short runs. (e) **cold-start crowding
  class found** (fix_98_1_3: nothing commits → commit-gated reset never fires; focused-window
  truth costs prove 2/3 detectable) — candidate fix: matcher-state-aware pre-commit reset.
  Deployment decisions pending (user): model revert (needs streaming-graph re-export from
  best_s123_mic + re-hosting), enabling vadReset windowed (trades the ~11x streaming RTF).
  See research/CLAUDE.md "Reassessment / taint audit". **RESOLVED same day:** model reverted +
  re-hosted (best_s123_mic-22s-v2, streaming graphs re-exported); the windowed accuracy config
  shipped, then superseded by v13 (next entry).
- **Repetition suppression + v13 fresh-context suffix decode — found, fixed, shipped
  (2026-07-11 pm).** A post-rollout live report ("tracking is very bad") root-caused to a MODEL
  pathology, not a regression: the Emformer's left-context memory DELETES repeated phrases in
  continuous audio (proved by a context-replacement probe: «maliki n-naas» decodes 16 ph
  standalone, 5 ph with the true preceding audio — ending in the same phrase — in memory).
  Repetitive short surahs (112/114) recited continuously are the worst case; the model only ever
  trained on single-ayah clips. RETRO-EXPLAINS the crowding episode (focused windows work by
  flushing the repeated phrase from the model memory). Two shipped fixes: (a) interim
  confident-emission-armed reset gate (`max(lastConfirmSec, lastEmitSec)` anchor, bar
  0.5x chainCost); (b) **v13 — per-hop standalone decode of the buffer's last 5 s through a
  right-sized graph (`Config.chainSuffixSec`/`chainSuffixModelPath`, model_suffix.int8.onnx),
  fed to a restricted two-window matched-filter pass. audio_bench 145/151 (96%) vs 138 (91%);
  SUBSUMES chainVadReset (off in the demo); skips while expecting a unit too long for its length
  gate (Baqarah junk-flood guard); 7 s variant rejected (140). ~+15% hop decode cost. Wired
  C++ -> JNI -> Kotlin; delivered via manifest `suffixModel` + `-PbundleSuffix`; hosted;
  user-validated live ("live testing was great").** Root fix queued: phase-3 concatenation
  training (continuous multi-ayah training clips synthesized from the existing corpus — no new
  data needed; acceptance = the context-replacement probe). Streaming-side v13 variant
  (parallel reset-every-5s stream) designed, unbuilt. See research/CLAUDE.md "Repetition
  suppression".
- **Phase-3 continuous-corpus training — DONE, ship candidate `best_s123_p31` (2026-07-11 pm).**
  User-sourced real continuous per-surah recitations (45.2 h / 7 voices; held-out test reciter
  quarantined eval-only) replaced synthetic concatenation. Full pipeline built + validated in
  one evening: spec-driven downloader, hierarchical alignment (99.9% of ~9,700 ayat, incl.
  4.4 h Baqarah files), <=28 s multi-ayah training windows over a PCM memmap cache, mixed
  manifest, mechanistic suppression probe (`research/probe_suppression.py`). **Result: the
  repetition-suppression pathology is FIXED at the source** (in-context/alone ratio 0.43 ->
  0.88-0.94, including on the held-out voice). p3 alone failed the bench gate (141 < 145 —
  clean windows diluted quiet-mic robustness); a 5-epoch phase-2 RESTORE (p3.1) recovered it:
  **bench 145/151 (= anchor), learner 85.3% (best ever), clean test 96.2% (best ever), val PER
  0.069.** v13 suffix remains necessary (+7 even on the fixed model); streaming remains behind
  on hard audio (125) — its deficit is not only suppression. Deployment (v3 hosting) pending
  user go. See research/CLAUDE.md "Phase-3 concatenation training".
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
  - *Model / accuracy:* (1) **true streaming export — inference DONE + validated (2026-07-10).**
    The data-dependent-mask block is retired: static-shape streaming Emformer step exported
    (`export/onnx/stream_{conv,encoder.int8}.onnx`) + C++ `StreamingModel` (`sdk/core/src/streaming.*`,
    two ORT sessions, 48-tensor state + conv cache + cross-chunk CTC collapse), **5/5 EXACT phoneme
    parity** vs the Python runtime (fp32). Premise re-verified (continuous recall 87.5% ≥ stitched
    85.0% on the current model). **Detector integration DONE + validated (2026-07-10):** gated on
    `Config.streamConvPath`+`streamEncoderPath` (empty => windowed re-decode, the default),
    `stepChain` streams only the new audio into a persistent `chainPh`/`chainTm`/`chainAlts` (center=True
    feed invariant: settled interior frames + hop-aligned buffer), Phase-2 posteriors threaded via
    `Emit.alts`. Acceptance: windowed vs streaming give IDENTICAL detections + timestamps on 54 s
    continuous 78:1-8 and 47 s Al-Fatiha (`test_detector`). **Android wiring DONE (2026-07-10):**
    `Config.streaming` -> JNI -> C++ (safe fallback to windowed when absent); `ModelManager` extracts
    the two graphs; demo `-PbundleStreaming` stages them (default distribution stays windowed);
    `:demo:assembleDebug -PbundleModel -PbundleStreaming` builds + packages all three graphs.
    **RTF win MEASURED (2026-07-10):** decode-only ~730 ms/hop (windowed) -> ~65 ms/hop (streaming),
    **RTF 0.484 -> 0.043 = ~11x cheaper per hop, byte-identical detections on the clean acceptance
    clips** (⚠️ 2026-07-11: on hard/quiet real audio streaming and windowed DIVERGE case-by-case —
    neither dominates; see the taint-audit status entry). The win came from
    computing log-mel over only the NEW suffix in `streamFeed` (the whole-buffer log-mel had masked
    it; +11.5 MB graphs). Verified live on-device (surah 111). **Conformance golden DONE:**
    `golden/streaming/` pins the C++ StreamingModel vs the Python runtime, `test_streaming` ALL PASS
    (6/6), spec.md §Streaming model inference. **Enabled by default + manifest delivery DONE:**
    `Config.streaming` defaults true; the manifest gains optional `streamConv`/`streamEncoder`
    `{url,sha256}`, `ModelManager` downloads them into `models/stream/` (sha256, version-keyed,
    non-fatal on failure -> windowed fallback); `:demo:modelManifest` emits the keys. **Hosting DONE
    + download path validated on-device (2026-07-10):** `stream_conv.onnx` + `stream_encoder.int8.onnx`
    + the regenerated manifest uploaded to the `model` release (`gh release upload`); a fresh
    download-build install fetched the model + both graphs, on-device sha256 byte-exact vs the hosted
    files, detector init clean. Default download builds now stream. See `export/streaming-export-plan.md`. (2) **in-house
    learner collection for the long surahs** (the known data hole — RetaSy covers only short
    surahs; raises the learner ceiling). (3) **full-Quran corpus** (beyond the current 1,057 ayat;
    out of MVP scope but the north star).
  - *Product / deployment:* (4) **iOS wrapper** (C++ core is portable; a Swift API + JNI-equivalent
    bridge, not a re-implementation). (5) **release signing / Play Store — SET UP; releases are now
    routine (2026-07-18).** ⚠️ **The Play store page ALREADY EXISTS** (Developer account, app entry,
    store listing) — do NOT re-suggest first-time setup. Identity locked: `applicationId
    com.quranrecite`, name "Quran Recite", `targetSdk`/`compileSdk` 35; custom launcher icon shipped;
    privacy policy hosted (https://haztaj.github.io/QtAR/privacy-policy.html). Signing: gitignored
    `sdk/android/keystore.properties` -> `quranrecite-release.jks` (the UPLOAD key; Play App Signing
    re-signs). **A release is just:** bump `versionCode` in `sdk/android/demo/build.gradle.kts`
    (Play rejects a reused code) -> `JAVA_HOME=/home/hazem/jdk17 ./gradlew :demo:bundleRelease` ->
    upload `demo/build/outputs/bundle/release/demo-release.aab` to a track. **Currently at
    versionCode 4 / 0.4.0** (page prior + collision blacklist; 29 MB, model downloads at runtime).
    Note for Data Safety: the app fetches ~199 MB of page fonts + the model on first launch. (6) **word-level segment highlighting** (light the active waqf segment's exact words; main work
    is the offline segment→word map — word-exact boundaries, ~85% correct-phrase, verifiable
    offline; see the assessment 2026-07-09).
  - *Research:* (7) **deeper N-back context** for the 8 structural `needs_choice` cases
    (2:134↔2:141 unit class). (8) **posterior-aware matching — Phases 0/1/2 DONE
    (2026-07-09/10).** Phase 0 (posteriors in the cache) + Phase 1 (retrieval, neutral — v10
    already saturated retrieval) + Phase 2 (soft SCORING): on a noise-augmented ~30% PER eval,
    sub_min~0 gives aligned-hit 84.0→85.7 / SER 16.6→14.8 / exact 48.9→52.6 with BYTE-neutral
    clean audio — a real win in the phone regime, free on the benchmark. **C++ port DONE
    (2026-07-10):** `decoder::topKAlts` + `infixNormSoft` + `Config.chainSubMin` (demo sets
    0.0), conformance-pinned (`soft_score_run`) + cross-validated EXACT over 200 real noisy
    streams — on-device phone detection now gets the win. See research/CLAUDE.md, spec §Stage 2b.
