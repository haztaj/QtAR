# QuranRecite SDK

On-device, offline Quran ayah-detection SDK for Android (and iOS later). A shared **C++
core** wrapped by idiomatic platform APIs. See `docs/sdk-architecture.md` for the design
and `conformance/` for the acceptance test the core must pass.

```
sdk/
├── core/        shared C++ engine (DSP front-end · ORT inference · CTC decode · matcher)
└── android/     Android library (.aar) + Compose demo app
                 (iOS XCFramework comes later)
```

## Status

- **Front-end (`frontend.cpp`) — done + conformance-validated.** Log-mel matches the
  Python reference within 1e-2 (worst 4.5e-3) on all fixtures.
- **Matcher / segmenter (`matcher.cpp`, `segmenter.cpp`) — done + conformance-validated.**
  Exact event sequences (incl. the real quiet-mic session 114:1→2→3).
- **Inference (`inference.cpp`) — done + validated.** ONNX Runtime session + greedy CTC
  decode reproduce the Python phonemes exactly (same-model).
- **Detector orchestration (`detector.cpp`) — done + validated end-to-end.** Rolling-window
  loop (resample → buffer → front-end → inference → decode → segmenter → highlight → events).
  The full pipeline reproduces `114:1 → 114:2 → 114:3` on the real quiet-mic session, fed in
  100 ms chunks. **The C++ core is complete.**
- **Highlight controller (`highlight.cpp`) — done + conformance-validated.** Stage-3 state
  machine (port of `matcher/highlight_controller.py`): consumes committed detections, emits
  render-ready `HighlightSnapshot`s, and defers ambiguous ayat instead of guessing (see
  `conformance/spec.md` §Stage 3). The C++ snapshots are byte-identical to the Python
  reference. This is the **centralized output contract** platform UIs render — the granular
  detect/advance events are retained for back-compat.
- **Builds via CMake+Ninja** (needs `-DORT_HOME=<onnxruntime>`) or direct g++.
  `nlohmann/json` vendored under `core/third_party/`; desktop ORT fetched to
  `build/onnxruntime/` (gitignored). See `conformance/spec.md` for the int8/`ConvInteger`
  note and the MinGW `-D_stdcall=__stdcall` flag.
- **int8 runs everywhere — done.** Re-exported as weight-only dynamic quant restricted to
  MatMul (15.2 MB, argmax-lossless); avoids the `ConvInteger` op older ORT CPU builds reject.
  Static QDQ was tried and rejected (it tanks this transformer's accuracy). Validated through
  the C++ core end-to-end on the ORT that previously couldn't run the int8 model.
- **Android `.aar` + demo — running on-device (2026-07-05).** Library `.aar` (~22 MB) and demo
  APK build with Gradle 8.7 / AGP 8.5 / NDK 27; CMake cross-compiles the core + JNI for
  arm64-v8a, armeabi-v7a, x86_64. The Compose mushaf demo runs live on a real device.
  - **Auto mode is the default** (`Mode::Auto`): the sliding segmenter + prefix-anchored stream
    matcher merged (`sdk/core/src/{stream,autodet}.cpp`, ports of `demo/streaming.py`/`auto.py`),
    handling any ayah length.
  - **Silero VAD** ported to the core (`vad.cpp`, bundled `silero_vad.onnx`): a speech-END resets
    the buffer + matcher for clean paused ayah-by-ayah segmentation.
  - **Two-phase highlight:** `HighlightSnapshot.upNext` reveals the next ayah (darker) once the
    active one nears completion; added at the public-snapshot layer (HighlightController untouched).
  - **Capture decoupled:** `AudioCapture` reads on one thread, runs inference on another (fixes
    ~30% dropped audio when inference stalled the read loop, and a stop-time crash).
  - **Runtime debug** (`Detector::setDebug`) gating all native logcat, toggled from the demo UI.
  - Verified: the JNI `.so` links ORT (`NEEDED libonnxruntime.so`) + exports the native symbols;
    the APK bundles our `.so` + ORT `.so` + assets + VAD. See `android/README.md`.
  - **Unit-chain decoder** (`Mode::Chain`, `chain.{h,cpp}`) — the research winning design
    (waqf segments; conformance-pinned); live on-device with `best_s123_mic_clean`.
  - **Model delivery** — manifest-driven download with update detection (`{version,url,sha256,
    description}`): the default APK ships model-free and fetches on first launch, detects a new
    release without an app update, and shows a "what's new" dialog on update; `-PbundleModel`
    ships it in the APK for an offline build. See `android/README.md`.
- **TODO:** host the model artifact + manifest (font zip is wired via `MushafFonts`); on-device
  RTF/memory profiling; then iOS.

```bash
cmake -S core -B build/cmake -G Ninja -DORT_HOME=$PWD/build/onnxruntime
cmake --build build/cmake
build/cmake/conformance_runner.exe ../conformance build/cmake_out && python ../conformance/verify.py --candidate build/cmake_out
build/cmake/test_detector.exe ../export/onnx/model.onnx ../conformance <recitation.wav>
```

```bash
# build + run the conformance acceptance gate (no ORT needed for these stages)
cmake -S core -B build/cmake -G Ninja -DQR_BUILD_CONFORMANCE=ON
cmake --build build/cmake --target conformance_runner
build/cmake/conformance_runner.exe ../conformance build/cmake_out
python ../conformance/verify.py --candidate build/cmake_out      # -> ALL PASS
```

## Build order

1. `core/` — implement against the conformance spec, validate on desktop with
   `core/tests/conformance_runner` → `python conformance/verify.py --candidate <out>`.
2. `android/quranrecite` — JNI bridge + Kotlin API + managed capture, packaged as `.aar`.
3. `android/demo` — Compose app: mushaf view, live ayah highlight, auto-advance.

## Reference (the spec to port)

| C++ component | Python reference |
|---|---|
| `frontend` (log-mel) | `training/data.py: logmel_16k`, `conformance/assets/*.bin` |
| `decoder` (CTC greedy) | `eval/evaluate.py: greedy_phonemes` |
| `matcher` (trie, edit dist, context) | `matcher/phoneme_matcher.py` |
| `segmenter` (sliding window) | `demo/sliding.py` |
| `highlight` (deferral + snapshots) | `matcher/highlight_controller.py` |
| `inference` | `export/` (ONNX int8) |
