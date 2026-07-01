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

- **JDK 17** (AGP 8.5 requires it; a system JDK 8 is too old).
- **Android SDK** (compileSdk 34, build-tools 34) + **NDK 26.1.10909125** + **CMake 3.22.1**
  (install via Android Studio's SDK Manager, or `sdkmanager`).
- Point Gradle at the SDK: create `sdk/android/local.properties` with
  `sdk.dir=<path>` (use forward slashes on Windows: `sdk.dir=C:/Users/you/android-sdk`), or
  set `ANDROID_HOME`. Android Studio writes this on first open.

The committed Gradle **wrapper** (`./gradlew`) pins Gradle 8.7 — no separate Gradle install
needed.

## Build

Open `sdk/android/` in **Android Studio** and Run the `demo` app, or from the command line:

```bash
cd sdk/android
# library .aar   -> quranrecite/build/outputs/aar/quranrecite-release.aar
./gradlew :quranrecite:assembleRelease

# demo APK       -> demo/build/outputs/apk/debug/demo-debug.apk
./gradlew :demo:assembleDebug

# publish the .aar to the local Maven repo for consumers
./gradlew :quranrecite:publishToMavenLocal

# install + run the demo on a connected device/emulator (needs a model — see below)
./gradlew :demo:installDebug
```

`preBuild` runs `bundleAssets` (stages the 4 small assets from `conformance/assets/` — run
`python conformance/generate.py` first if the `*.bin` are missing) and `extractOrt` (unpacks
the ORT headers + per-ABI `libonnxruntime.so` from the onnxruntime-android AAR, which is not
prefab-packaged, into an imported CMake target). The core `.so` links ORT at build time; the
ORT `.so` itself is packaged into the consuming app by the onnxruntime-android dependency.

## Assets & the model

The engine needs five files. The **four small ones** (`ayah_phonemes.json`, `tokens.txt`,
`mel_filterbank.bin`, `hann_window.bin`) are copied into the `.aar` by the `bundleAssets`
task from `conformance/assets/` and extracted to `filesDir` at runtime.

The **~15 MB `model.int8.onnx`** is *not* committed (repo rule) and is delivered by
`ModelManager` at first launch:

- **Production:** set `MODEL_URL` + `MODEL_SHA256` in `ModelManager.kt` to a hosted,
  versioned artifact — downloaded once, sha256-verified, cached under
  `filesDir/quranrecite/<version>/`. The library `.aar` ships model-free.
- **Dev/offline (default for the demo):** the demo's `bundleDevModel` Gradle task copies
  `export/onnx/model.int8.onnx` into the demo's assets at build time, so **the demo APK is
  fully self-contained and runs with no network**. `ModelManager` prefers a bundled model
  over downloading. Just run `python export/export_onnx.py` once, then build the demo. The
  bundled copy is gitignored; the library `.aar` is unaffected.

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

**Builds.** The library `.aar` (18 MB) and the demo APK both build (Gradle 8.7 / AGP 8.5 /
NDK 26); CMake cross-compiles the shared core + JNI for all three ABIs. Verified: the JNI
`.so` links ORT (`NEEDED libonnxruntime.so`) and exports the native symbols; the APK bundles
`libquranrecite_jni.so` + `libonnxruntime.so` + `libc++_shared.so` + the 4 runtime assets.
The JNI bridge marshals `detect`/`advance` to Kotlin (main thread), managed capture bakes in
the mic recommendations, and `ModelManager` implements asset extraction + download +
sha256.

The demo APK is **self-contained and offline** — the int8 model is dev-bundled, so on launch
it extracts the model + assets and is ready with no server. Install and recite:

```bash
./gradlew :demo:installDebug     # onto a connected device/emulator (grant the mic prompt)
```

Not yet **run** here (the dev container has no device/mic). Next: install on a device and
validate live detection; then host the model + package/publish; iOS after. The dedicated
fixed **4 s streaming export** (vs the current 30 s window) is a perf follow-up before
shipping.
