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
`QCF2001…QCF2604`) driven by a bundled layout DB, styled like a printed mushaf:

- **Top strip:** the page's surah name (`surah-name.ttf`, ligature `surahNNN`) on the left and
  the juz number (`quran-common.ttf`, ligature `juzNNN`) on the right; an ornate surah-header
  banner (`surah-header.ttf`) renders on the page itself.
- **Page:** maximized; swipe **right-to-left** between the 604 pages. The page number sits at
  the bottom in Eastern-Arabic numerals, **odd→right / even→left** (printed-mushaf style).
- **Tap** anywhere toggles two panels: a **top** panel (jump-to-page + debug toggles) and a
  **bottom** panel (start/stop detection + status).
- The detected ayah highlights (lighter) and, once it nears completion, the next ayah is shown
  darker as "up next" — a **two-phase** highlight (`onHighlightState`, `active` + `upNext`); the
  pager auto-advances to follow the reciter.
- **Auto-fit + re-fit:** the page auto-sizes its font to the screen and re-fits on **foldable
  postures and orientation changes** (the Activity handles config changes itself; the page
  measures via `BoxWithConstraints`, keeping the native detector alive across resize). A widened
  fit margin (`FIT 0.90`) avoids the widest justified line overflowing on wide/landscape screens.

**Debug (UI-controlled):** the top panel has **Debug logging** (gates native + SDK logcat via
`setDebugLogging` → `Detector::setDebug`) and **Record session audio** (dumps the exact fed PCM
to a WAV via `setRecording`) + **Share last recording** (FileProvider). Both persist in
`SharedPreferences`, off by default — the instrumentation lives in-tree, not stripped per commit.

**Assets (not committed — large third-party binaries, gitignored under
`demo/src/main/assets/mushaf/`, plus the relocated page fonts under `demo/mushaf-fonts/`):**
- `fonts/pN.ttf` — the 604 KFGQPC V2 page fonts (word glyphs are page-local PUA codepoints).
  **Downloaded once at runtime** (~199 MB) into external files, not shipped in the APK — see below.
- `fonts/surah-name.ttf`, `fonts/surah-header.ttf`, `fonts/quran-common.ttf`,
  `surah-header-ligatures.json` — bundled (small).
- `layout.db` — `pages` table: per (page,line) the `line_type`, `is_centered`, and word-id range.
- `words.db` — `words` table: per word (id 1..83668) its `surah`, `ayah`, and glyph `text`
  (a single page-local PUA codepoint per word).

Get these from [qul.tarteel.ai](https://qul.tarteel.ai): the *KFGQPC V2 mushaf-layout* (SQLite),
the *QPC V2 page-by-page font* (TTF), and the *QPC V2 Glyph word-by-word* script (SQLite).
Rendering a line = concatenate each word's glyph (space-separated) in that page's font; the
basmalah uses the page font's glyph. The Juz-Amma model still detects across the whole mushaf;
only ayat the model was trained on (78–114) will highlight.

### Page-font delivery (download-once)

The 604 page fonts are ~199 MB — too large to ship in the APK, where they would re-download on
every app update. Instead they are relocated out of the packaged assets (to `demo/mushaf-fonts/`)
and **downloaded once** into `getExternalFilesDir`, which **survives app updates** — so the beta
APK is **~64 MB** (down from ~205 MB) and updates never re-ship the fonts. `MushafFonts` fetches a
versioned zip (`FONTS_URL` + `FONTS_SHA256` + `FONTS_VERSION`), verifies the hash, unzips once, and
shows a determinate "Downloading text (one time)" progress screen with a retry on failure. Produce
the hosting zip with `./gradlew :demo:zipMushafFonts` (→ `build/mushaf-fonts.zip`). For offline dev,
`./gradlew :demo:assembleDebug -PbundleFonts` keeps the fonts in the APK and loads them from assets
(no download). The DBs + small fonts stay bundled.

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

The engine needs a handful of small assets plus the model. The **small ones**
(`ayah_phonemes.json`, `tokens.txt`, `mel_filterbank.bin`, `hann_window.bin`,
`ambiguous_ayat.json` — the Stage-3 confusable map that enables ambiguity deferral, and
`silero_vad.onnx` — the ~1.3 MB VAD that resets on paused-recitation boundaries) are copied into
the `.aar` by the `bundleAssets` task from `conformance/assets/` and extracted to `filesDir` at
runtime. Both `ambiguous_ayat.json` and `silero_vad.onnx` are optional: if absent, deferral /
paused-recitation VAD-reset are simply disabled. `conformance/generate.py` reproduces the VAD
asset from the pip `silero-vad` package (it's `.onnx`, hence gitignored).

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

**Running on-device (2026-07-05).** The Compose demo runs live on a real device (Samsung
foldable). Landed this session:
- **Auto mode** default (sliding + stream matchers merged; `sdk/core/src/{stream,autodet}.cpp`).
- **Silero VAD** in the core resets on paused-recitation boundaries (`vad.cpp`, bundled asset).
- **Capture decoupled** — `AudioCapture` reads on one thread and runs inference on another, so
  an inference stall no longer stalls `AudioRecord.read()` (this was dropping ~30% of samples →
  garbled audio → bad detection) and no longer races on Stop.
- **Two-phase highlight** (`upNext`), the mushaf reader redesign, **font download-once**
  packaging (APK ~205 MB → ~64 MB), and the **runtime debug** panel (above).

The demo's int8 model is still dev-bundled (the 4 s window model), so it runs with no server
once the fonts are downloaded. Install and recite:

```bash
./gradlew :demo:installDebug     # onto a connected device/emulator (grant the mic prompt)
```

Next: host the model + font-zip artifacts (both wired for download); on-device RTF/memory
profiling; then iOS.
