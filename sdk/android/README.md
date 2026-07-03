# QuranRecite — Android SDK (`.aar`) + demo

Android library wrapping the shared C++ core (`sdk/core`) with a Kotlin API, plus a Compose
demo app. The library packages a JNI bridge + the core `.so` (built via CMake) and depends
on the ONNX Runtime Android AAR (consumed through prefab).

```
android/
├── quranrecite/   the SDK library (.aar) — Kotlin API, JNI bridge, CMake -> core
└── demo/          Compose demo: real mushaf pages, live highlight + auto-advance
```

## Demo — mushaf reader

The demo renders **real Quran pages** with the KFGQPC V2 glyph fonts (604 page fonts,
`QCF2001…QCF2604`) driven by a bundled layout DB. Swipe **right-to-left** between the 604
pages, **jump to any page**, and **start/stop** live detection from the bottom bar; the
detected ayah highlights on the page and the pager auto-advances to follow the reciter
(wired to `onHighlightState`). The page auto-sizes its font to the screen, so it re-fits on
**foldable postures and orientation changes** (the Activity handles config changes itself and
the page measures via `BoxWithConstraints`, keeping the native detector alive across resize).

**Assets (not committed — large third-party binaries, gitignored under
`demo/src/main/assets/mushaf/`):**
- `fonts/pN.ttf` — the 604 KFGQPC V2 page fonts (word glyphs are page-local PUA codepoints).
- `layout.db` — `pages` table: per (page,line) the `line_type`, `is_centered`, and word-id range.
- `words.db` — `words` table: per word (id 1..83668) its `surah`, `ayah`, and glyph `text`.

Get these from [qul.tarteel.ai](https://qul.tarteel.ai): the *KFGQPC V2 mushaf-layout* (SQLite),
the *QPC V2 page-by-page font* (TTF), and the *QPC V2 Glyph word-by-word* script (SQLite),
placed as above. Rendering a line = concatenate each word's glyph in that page's font;
surah-name headers use a simple placeholder (the KFGQPC surah-name font isn't bundled) and
the basmalah uses the page font's glyph. The Juz-Amma model still detects across the whole
mushaf; only ayat the model was trained on (78–114) will highlight.

## Prerequisites

- **JDK 17** — AGP 8.5 requires it. Note: it's not enough to have JDK 17 *installed*; Gradle
  must **run on** it. If your machine default is older (e.g. JDK 8), `./gradlew` fails at
  configuration with *"Could not resolve com.android.tools.build:gradle … compatible with
  Java 8"*. Fix it one of two ways:
  - set `JAVA_HOME` to the JDK 17 for the shell, e.g. (PowerShell)
    `$env:JAVA_HOME="C:/path/to/jdk17"`, or
  - add `org.gradle.java.home=C:/path/to/jdk17` (forward slashes) to your **user**
    `~/.gradle/gradle.properties` — applies to every `./gradlew` without per-shell setup.
- **Android SDK** (compileSdk 34, build-tools 34) + **NDK 27.2.12479018** + **CMake 3.22.1**
  (install via Android Studio's SDK Manager, or `sdkmanager`). NDK r27 ships a 16 KB-aligned
  `libc++_shared.so`; r26 does not.
- Point Gradle at the SDK: create `sdk/android/local.properties` with
  `sdk.dir=<path>` (use forward slashes on Windows: `sdk.dir=C:/Users/you/android-sdk`), or
  set `ANDROID_HOME`. Android Studio writes this on first open.

The committed Gradle **wrapper** (`./gradlew`) pins Gradle 8.7 — no separate Gradle install
needed. (Android Studio sidesteps the JDK issue entirely — it runs Gradle on its bundled
JDK 17.)

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

The engine needs six files. The **five small ones** (`ayah_phonemes.json`, `tokens.txt`,
`mel_filterbank.bin`, `hann_window.bin`, and `ambiguous_ayat.json` — the Stage-3 confusable
map that enables ambiguity deferral) are copied into the `.aar` by the `bundleAssets` task
from `conformance/assets/` and extracted to `filesDir` at runtime. (`ambiguous_ayat.json` is
optional: if absent, deferral is disabled and every detection confirms immediately.)

The **~15 MB `model.int8.onnx`** is *not* committed (repo rule) and is delivered by
`ModelManager` at first launch:

- **Production:** set `MODEL_URL` + `MODEL_SHA256` in `ModelManager.kt` to a hosted,
  versioned artifact — downloaded once, sha256-verified, cached under
  `filesDir/quranrecite/<version>/`. The library `.aar` ships model-free.
- **Dev/offline (default for the demo):** the demo's `bundleDevModel` Gradle task copies
  the **4 s sliding-window model** (`export/onnx/model_4s.int8.onnx`, ~11 MB) into the demo's
  assets at build time, so **the demo APK is fully self-contained and runs with no network**.
  `ModelManager` prefers a bundled model over downloading. Run
  `python export/export_onnx.py --fixed-frames 416 --tag _4s` once, then build the demo. The
  bundled copy is gitignored; the library `.aar` is unaffected. (The 4 s model is ~15× cheaper
  per hop than the 30 s full-utterance export with identical detections — see `export/CLAUDE.md`.)

## API (host app)

```kotlin
val detector = QuranReciteDetector(context, Config())
detector.setListener(object : QuranReciteDetector.Listener {
    override fun onModelReady() { /* enable Listen */ }
    // Primary contract: render this snapshot wholesale. It already handles deferral +
    // ambiguity (never guesses), so no per-UI logic is needed.
    override fun onHighlightState(state: HighlightState) {
        render(state.confirmed, state.active)          // highlight settled + current ayah
        state.pending?.let { showOptions(it.options) } // deferred: surface the candidates
    }
})
detector.prepare()                 // resolves assets (downloads model on first launch)
// after onModelReady():
detector.start()                   // managed mic capture (needs RECORD_AUDIO)
// ...or feed your own PCM: detector.feed(pcm16, sampleRate)
detector.release()
```

`onHighlightState` is the **centralized output contract** — one immutable `HighlightState`
snapshot per change (`confirmed[]` · `pending{options,reason}` · `active`). The deferral,
ambiguity handling and retroactive resolution live once in the shared C++ core
(`matcher/highlight_controller.py` → `sdk/core`), so every platform/UI just renders the
snapshot. The granular `onAyahDetected`/`onAyahAdvance` callbacks remain for back-compat /
custom flows. Events are delivered on the main thread. The native core is the same one
validated by `sdk/core` / `conformance` on desktop — see `sdk/README.md`.

## Status

**Builds.** The library `.aar` and the demo APK both build (Gradle 8.7 / AGP 8.5 / NDK 27);
CMake cross-compiles the shared core + JNI for all three ABIs. Verified: the JNI `.so` links
ORT (`NEEDED libonnxruntime.so`) and exports the native symbols; the APK bundles
`libquranrecite_jni.so` + `libonnxruntime.so` + `libc++_shared.so` + the 4 runtime assets.
The JNI bridge marshals `detect`/`advance` to Kotlin (main thread), managed capture bakes in
the mic recommendations, and `ModelManager` implements asset extraction + download +
sha256.

**16 KB page-size compatible** (required by Google Play). All native libs are 16 KB-aligned:
NDK r27 (`-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON`) aligns `libquranrecite_jni.so` +
`libc++_shared.so`, and ORT was bumped **1.18.0 → 1.22.0** (whose prebuilt `libonnxruntime.so`
is 16 KB-aligned; 1.18 was 4 KB). Verified: ELF `p_align=0x4000` on every lib and
`zipalign -c -P 16` passes on the APK. The model output was re-validated on desktop ORT
1.22 (CPU EP, same kernels as Android): conformance + `test_detector` are byte-identical to
1.18 (`114:1→2→3`), so the ORT bump doesn't change detections.

The demo APK is **self-contained and offline** — the int8 model is dev-bundled, so on launch
it extracts the model + assets and is ready with no server. Install and recite:

```bash
./gradlew :demo:installDebug     # onto a connected device/emulator (grant the mic prompt)
```

Not yet **run** here (the dev container has no device/mic). Next: install on a device and
validate live detection; then host the model + package/publish; iOS after. The dedicated
fixed **4 s streaming export** (vs the current 30 s window) is a perf follow-up before
shipping.
