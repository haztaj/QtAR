# SDK & app architecture — recommendation

Goal: ship the ayah-detection engine as a **cross-platform SDK** importable into native
**Android** and **iOS** apps, plus a **demo app** that showcases it. On-device, offline.

This is a design recommendation, not yet built. The existing Python (`training/data.py`
front-end, `matcher/phoneme_matcher.py`, `demo/sliding.py`, `export/`) is the reference
spec to port.

---

## 1. Guiding constraints

- **Fully on-device / offline** — strong privacy story, no network at inference.
- **Numerical fidelity** — the log-mel front-end and the matcher must produce *identical*
  results to the Python reference (training). Drift = silent accuracy loss. This single
  constraint drives the biggest architectural decision (a shared core, §3).
- **Small footprint** — ~16 MB today (model dominates), fixed regardless of corpus.
- **Stable API across a changing model/corpus** — Juz Amma now → full Quran later must
  not break the host-app API; only the bundled model + lexicon change.

---

## 2. Layered architecture

```
┌─────────────────────────────────────── host app (Kotlin / Swift) ───────────┐
│  UI (mushaf view, highlight, auto-advance) · mic permission · lifecycle      │
└───────────────▲───────────────────────────────────────────────┬────────────┘
                │ events (onAyahDetected/Advance/Progress)        │ start/stop/feed
┌───────────────┴───────────────── SDK public API (thin native wrapper) ───────┐
│  Kotlin (Android)  ·  Swift (iOS)  — idiomatic, stable surface               │
└───────────────▲───────────────────────────────────────────────┬────────────┘
                │ JNI / C-ABI                                     │
┌───────────────┴───────────────── shared core (C++) ──────────────────────────┐
│  1. Audio ring buffer + (optional) managed capture frames                    │
│  2. Front-end DSP: resample→16k, log-mel(80), RMS-normalize  [matches data.py]│
│  3. Inference: ONNX Runtime (int8 model) + NNAPI/CoreML EP                    │
│  4. CTC greedy decode → phonemes                                             │
│  5. Stage-2 matcher: trie + sliding-window segmenter + sequential context    │
│  6. VAD/energy gate (Silero ONNX or energy)                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                       assets: model.int8.onnx · tokens.txt · ayah_phonemes.json · ayah_text.json
```

---

## 3. The core decision: ONE shared C++ core (recommended)

The front-end DSP and the matcher are non-trivial and **must** match the Python bit-for-
bit-ish. Re-implementing them twice (Kotlin + Swift) invites drift and doubles the
maintenance + conformance burden. So: **implement the engine once in C++**, wrap it with
thin idiomatic API layers per platform.

- **Why C++:** ONNX Runtime is C++-native on every platform (best integration), it's the
  proven path for mobile ML SDKs, builds with CMake to `.so`/`.a`, and is universally
  understood. The matcher/DSP port is ~a few hundred lines.
- **Rust is a credible alternative** (memory-safe; `ort` crate wraps ONNX Runtime;
  UniFFI auto-generates Kotlin/Swift bindings) — choose it only if the team prefers Rust;
  ORT integration is slightly less first-class than C++.
- **Two native implementations (Kotlin + Swift, no shared core): not recommended** — the
  numerical-drift risk on the log-mel + matcher is real and hard to keep in sync.

A pragmatic middle ground if C++ appetite is low: put **only** the numerically-sensitive
parts (log-mel + matcher) in C++ and keep orchestration native — but a single full C++
core is cleaner.

---

## 4. Component design

- **Front-end DSP** — port `data.py` exactly: 16 kHz, 80-dim log-mel, n_fft 400 / hop 160
  / win 400, fmin 20 / fmax 8000, log floor 1e-10, **RMS-normalize to 0.1**. Use a small
  FFT lib (pffft/KissFFT). This is the #1 conformance risk — see §10.
- **Inference** — ONNX Runtime Mobile, the int8 model, **sliding fixed window (4 s)** so
  export is a small fixed-shape graph (not the 30 s full-utterance one). Enable NNAPI
  (Android) / CoreML (iOS) execution providers with CPU fallback; A/B them for int8
  accuracy.
- **Decode** — CTC greedy (argmax → collapse repeats → drop blank). Trivial.
- **Matcher / segmenter** — port `phoneme_matcher.py` (trie, edit distance, partial
  scoring, `SequentialContext`) + `sliding.py` (`SlidingWindowSegmenter`). Pure CPU,
  fast. Default = **sliding mode** (continuous recitation, validated live).
- **VAD/energy gate** — Silero VAD runs as an ONNX model via the same ORT, or a cheap
  energy gate; used to skip silence.

---

## 5. Public API (sketch — same shape both platforms)

Offer **two capture models**: SDK-managed (convenience) and app-fed PCM (flexibility).

```kotlin
// Android (Kotlin) — mirror in Swift
class QuranReciteDetector(config: Config)          // model/lexicon paths, mode, window, hop

config: corpus (juzAmma|fullQuran), mode (sliding|buffer), captureManaged: Bool, …

// managed capture:
detector.start()                                   // SDK opens mic w/ recommended settings
detector.stop()

// or app-fed PCM (advanced; app controls capture per docs/mobile-audio-capture.md):
detector.feed(pcm16: ShortArray, sampleRate: Int)

// events (delivered on a chosen dispatcher / main thread):
onAyahDetected(surah, ayah, confidence)
onAyahAdvance(fromSurahAyah, toSurahAyah)          // sequence moved forward
onProgress(surah, ayah, fraction)                  // optional, for UI
onError(...)
```

- Identifiers are `surah:ayah` (the stable ID); the host app owns the mushaf text/render.
- Events run off a worker thread; the SDK marshals them to the host's main/UI thread.
- `confidence` from the matcher cost; thresholds configurable.

---

## 6. Packaging & distribution

- **Android**: `.aar` containing the C++ `.so` (arm64-v8a, armeabi-v7a, x86_64) + Kotlin
  API + assets; depend on the ONNX Runtime Android AAR. Publish to Maven (Central or
  private). `minSdk` ~24 (NNAPI / UNPROCESSED source).
- **iOS**: `.xcframework` (device arm64 + simulator) + Swift API + assets; depend on the
  ONNX Runtime iOS package. Distribute via **Swift Package Manager** (primary) and/or
  CocoaPods. iOS 13+.
- **Assets (~16 MB)**: bundle in the SDK for the MVP (simplest). Offer a
  **download-on-first-launch** option later (smaller install; needs caching + integrity
  check) — matters more at full-Quran scope only if the model grows (it won't much).

---

## 7. Audio capture strategy

Bake the §`docs/mobile-audio-capture.md` recommendations into the **SDK-managed capture**
so app developers get good results by default:
- Android: `VOICE_RECOGNITION` source + `AutomaticGainControl`; NS off/A-B; AEC off.
- iOS: `.measurement` (or VoiceProcessingIO with AGC, NS bypassed).
For app-fed PCM, document the same. The SDK's RMS-normalization is the software floor
regardless.

---

## 8. Lifecycle & threading

- One audio worker thread (capture/feed) → ring buffer → inference worker (ORT) →
  matcher → events. Keep the UI thread free.
- Handle interruptions (calls, route changes, backgrounding) — pause/resume the engine.
- Deterministic `start/stop/reset`; `reset` clears the sequential context (new session).

---

## 9. Corpus scaling (Juz Amma → full Quran)

The API is corpus-agnostic: only the bundled **lexicon** (`ayah_phonemes.json`) and the
model change. Lexicon grows 40 KB → ~450 KB; model stays ~15 MB. When scope expands, the
model may need re-tuning/capacity changes (noted separately) — but **no SDK API change**.
Expose a `corpus`/model-version field so the host app knows coverage.

---

## 10. Numerical-conformance testing (critical, do this early)

The port's biggest risk is the C++ log-mel + matcher diverging from Python. Mitigate with
a **golden-fixture test suite**, generated from the Python reference:
- fixtures: {raw audio → expected log-mel, → expected phonemes, → expected ayah events}.
- C++ core must match log-mel within tight tolerance and reproduce phonemes/events exactly.
- Run in CI on both platforms. This is what makes the shared-core port trustworthy.

---

## 11. The demo app

Best showcase = a **mushaf view that highlights the current ayah and auto-advances** as
you recite (the real use case). It exercises every SDK event.
- Recommended: **one native demo per platform** (Android Compose, iOS SwiftUI) — most
  representative of "import the native SDK." Start with whichever platform is primary.
- A single cross-platform demo (Flutter/Compose-Multiplatform over the SDK via a plugin)
  is an option to halve UI work, but a native demo proves the native-import story better.
- Demo features: listen button, live ayah highlight + auto-scroll, confidence/ξprogress
  indicator, surah picker, a debug overlay (decoded phonemes / top candidates) reusing
  the session-recording idea for field diagnostics.

---

## 12. Suggested build phases

1. **C++ core + conformance harness** — port DSP + decode + matcher; validate against
   Python golden fixtures on desktop first.
2. **ORT integration** — wire the int8 sliding-window model; verify parity vs Python.
3. **Android AAR + Kotlin API** + managed capture; bring up on a device.
4. **iOS XCFramework + Swift API** + managed capture.
5. **Demo app** (primary platform first), then the second.
6. Hardening: lifecycle/interruptions, NNAPI/CoreML A-B, on-device RTF/memory profiling
   (the eval roadmap item), packaging/publishing.

---

## 13. Decisions (locked 2026-06-30)

- **Core language: C++** — single shared core, JNI (Android) + ObjC++ (iOS) wrappers.
- **Primary platform: Android first** — C++ core + `.aar` + Kotlin API + Compose demo,
  then port to iOS (XCFramework + Swift + SwiftUI).
- **Demo: native per platform** — Android Compose demo first.
- **Assets: download-on-first-launch** — ship a small SDK; fetch model + lexicon on first
  run. Implies: a hosted, **versioned** model artifact (URL + version manifest), on-device
  **caching**, **integrity check** (checksum/signature), and graceful offline handling
  (can't run until first download completes — surface this in the API/UX). Bundle the tiny
  lexicon/tokens as a fallback; only the ~15 MB model is downloaded.
- **Capture**: SDK-managed (default, bakes in mic recs) + app-fed PCM (advanced). [as §7]
- **Distribution**: TBD (Maven Central vs private) — decide at publish time.

## 14. Immediate next step — conformance harness (in this repo)

Before any C++: build the **golden-fixture generator** from the Python reference
(`data.py`, `phoneme_matcher.py`, `sliding.py`) — a set of {audio → log-mel → phonemes →
sliding-window ayah events} fixtures + tolerances. This lives here, is pure Python, and
becomes the acceptance test the C++ core must pass. De-risks the whole port and is the
natural first deliverable.
