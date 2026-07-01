# QuranRecite — Android SDK (`.aar`) + demo

Android library wrapping the shared C++ core (`sdk/core`) with a Kotlin API, plus a Compose
demo app. The library packages a JNI bridge + the core `.so` (built via CMake) and depends
on the ONNX Runtime Android AAR (consumed through prefab).

```
android/
├── quranrecite/   the SDK library (.aar) — Kotlin API, JNI bridge, CMake -> core
└── demo/          Compose demo: mushaf view, live highlight + auto-advance
```

## Prerequisites

- **JDK 17** (AGP 8.5 requires it; the repo's system JDK 8 is too old for Gradle here).
- **Android SDK** (compileSdk 34) + **NDK** + **CMake 3.22.1** (install via Android Studio's
  SDK Manager, or `sdkmanager`).
- Point Gradle at the SDK: create `sdk/android/local.properties` with `sdk.dir=<path>` (or
  set `ANDROID_HOME`). Android Studio does this on first open.

## Build

Simplest: open `sdk/android/` in **Android Studio** (it supplies Gradle + wires the SDK/NDK)
and Run the `demo` app, or build the library artifact.

Command line (needs a Gradle wrapper — run `gradle wrapper` once, or use Studio's):

```bash
cd sdk/android
# 1) stage the small runtime assets into the .aar (needs conformance/assets/*.bin —
#    run `python conformance/generate.py` at the repo root first). Wired into preBuild,
#    but you can run it explicitly:
./gradlew :quranrecite:bundleAssets

# 2) build the library .aar   -> quranrecite/build/outputs/aar/quranrecite-release.aar
./gradlew :quranrecite:assembleRelease

# or publish to the local Maven repo for consumers
./gradlew :quranrecite:publishToMavenLocal

# 3) run the demo on a connected device/emulator (needs a model — see below)
./gradlew :demo:installDebug
```

## Assets & the model

The engine needs five files. The **four small ones** (`ayah_phonemes.json`, `tokens.txt`,
`mel_filterbank.bin`, `hann_window.bin`) are copied into the `.aar` by the `bundleAssets`
task from `conformance/assets/` and extracted to `filesDir` at runtime.

The **~15 MB `model.int8.onnx`** is *not* committed (repo rule) and is delivered by
`ModelManager` at first launch:

- **Production:** set `MODEL_URL` + `MODEL_SHA256` in `ModelManager.kt` to a hosted,
  versioned artifact — downloaded once, sha256-verified, cached under
  `filesDir/quranrecite/<version>/`.
- **Dev/offline (to run now):** drop the model at
  `quranrecite/src/main/assets/quranrecite/model.int8.onnx` (produced by
  `python export/export_onnx.py`). `ModelManager` uses a bundled model directly, no network.
  This path is gitignored.

## API (host app)

```kotlin
val detector = QuranReciteDetector(context, Config())
detector.setListener(object : QuranReciteDetector.Listener {
    override fun onModelReady() { /* enable Listen */ }
    override fun onAyahDetected(ayah: AyahId, confidence: Float) { highlight(ayah) }
    override fun onAyahAdvance(from: AyahId, to: AyahId) { highlight(to) }
})
detector.prepare()                 // resolves assets (downloads model on first launch)
// after onModelReady():
detector.start()                   // managed mic capture (needs RECORD_AUDIO)
// ...or feed your own PCM: detector.feed(pcm16, sampleRate)
detector.release()
```

Events are delivered on the main thread. The native core is the same one validated by
`sdk/core` / `conformance` on desktop — see `sdk/README.md`.

## Status

Wired and buildable-in-Android-Studio: CMake pulls in the shared core (ORT via prefab),
the JNI bridge marshals `detect`/`advance` events to Kotlin, managed capture bakes in the
mic recommendations, and `ModelManager` implements extraction + download + sha256. Not yet
run on a device from this repo (no Android toolchain in the dev container) — next: bring up
on a device/emulator, then package/publish. The dedicated fixed **4 s streaming export**
(vs the current 30 s window) is a perf follow-up before shipping.
