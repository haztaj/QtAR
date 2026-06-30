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
- **Builds via CMake+Ninja** (or direct g++). `nlohmann/json` is vendored under
  `core/third_party/` for lexicon/fixture I/O.
- **TODO:** `inference.cpp` (ONNX Runtime session), `detector.cpp` orchestration loop,
  resampling, then the Android `.aar` + Compose demo, then iOS.

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
| `inference` | `export/` (ONNX int8) |
