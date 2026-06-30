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
  loop (resample → buffer → front-end → inference → decode → segmenter → events). The full
  pipeline reproduces `114:1 → 114:2 → 114:3` on the real quiet-mic session, fed in 100 ms
  chunks. **The C++ core is complete.**
- **Builds via CMake+Ninja** (needs `-DORT_HOME=<onnxruntime>`) or direct g++.
  `nlohmann/json` vendored under `core/third_party/`; desktop ORT fetched to
  `build/onnxruntime/` (gitignored). See `conformance/spec.md` for the int8/`ConvInteger`
  note and the MinGW `-D_stdcall=__stdcall` flag.
- **int8 runs everywhere — done.** Re-exported as weight-only dynamic quant restricted to
  MatMul (15.2 MB, argmax-lossless); avoids the `ConvInteger` op older ORT CPU builds reject.
  Static QDQ was tried and rejected (it tanks this transformer's accuracy). Validated through
  the C++ core end-to-end on the ORT that previously couldn't run the int8 model.
- **TODO:** the Android `.aar` (JNI + Kotlin scaffold exists) + Compose demo, then iOS.

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
| `inference` | `export/` (ONNX int8) |
