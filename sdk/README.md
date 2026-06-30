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

Scaffold. Interfaces and project structure are in place; the numerically-sensitive
implementations (`frontend.cpp`, `matcher.cpp`, `segmenter.cpp`) are stubs that must be
filled per `conformance/spec.md` and validated with `conformance/verify.py`.

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
