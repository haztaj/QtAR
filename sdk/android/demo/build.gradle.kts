import java.security.MessageDigest
import java.util.Properties

// Release signing. Secrets live in sdk/android/keystore.properties (gitignored) so the keystore and
// its passwords are NEVER committed — copy keystore.properties.example and fill it in, and generate
// the keystore with keytool (see that file). When the properties file is absent (fresh clone / CI
// without secrets), the release build falls back to the debug key so it still builds/installs — it
// is just not a distributable release. `storeFile` may be relative to sdk/android or absolute.
val keystorePropsFile = rootProject.file("keystore.properties")
val hasReleaseSigning = keystorePropsFile.exists()
val keystoreProps = Properties().apply {
    if (hasReleaseSigning) keystorePropsFile.inputStream().use { load(it) }
}

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.0"
}

android {
    namespace = "com.quranrecite.demo"
    compileSdk = 35
    defaultConfig {
        // Permanent Play identity — locked on first publish (the code namespace stays
        // com.quranrecite.demo; only the installed/published app id is com.quranrecite).
        applicationId = "com.quranrecite"
        minSdk = 24
        targetSdk = 35
        versionCode = 2
        versionName = "0.2.0"   // v13 suffix decode + phase-3 model (v3) + near-twin guard
        // Ship only real-phone (arm64) + emulator (x86_64) ABIs. Drops x86/armeabi-v7a
        // (~34 MB of unused ONNX Runtime) and the x86 mismatch (no x86 quranrecite_jni).
        // A Play release would instead use an App Bundle for per-device ABI delivery.
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }

    signingConfigs {
        // Only defined when keystore.properties is present; the release buildType falls back to the
        // debug key otherwise (see below), so a keyless clone still builds.
        if (hasReleaseSigning) create("release") {
            storeFile = rootProject.file(keystoreProps.getProperty("storeFile"))
            storePassword = keystoreProps.getProperty("storePassword")
            keyAlias = keystoreProps.getProperty("keyAlias")
            keyPassword = keystoreProps.getProperty("keyPassword")
        }
    }
    buildTypes {
        release {
            // Native-heavy app with a tiny Kotlin surface (JNI + reflection into the SDK) — keep R8
            // off so nothing gets stripped/renamed out from under the native bridge. Revisit with a
            // keep-rules pass if size becomes a concern.
            isMinifyEnabled = false
            signingConfig = if (hasReleaseSigning) signingConfigs.getByName("release")
                            else signingConfigs.getByName("debug")
        }
    }
    buildFeatures { compose = true }
    androidResources { noCompress += "onnx" }    // store the model uncompressed (clean extract)

    // The 604 page fonts (~199 MB) live OUTSIDE the packaged assets (in mushaf-fonts/) and are
    // downloaded once at runtime into external files, which survives app updates — so they are not
    // in the APK and updates don't re-ship them (see MushafFonts). `-PbundleFonts` adds them back as
    // an asset source for offline local dev (loaded straight from assets, no download).
    if (project.hasProperty("bundleFonts"))
        sourceSets["main"].assets.srcDir("mushaf-fonts")
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation(project(":quranrecite"))     // the SDK, imported like any consumer
    implementation(platform("androidx.compose:compose-bom:2024.06.00"))
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.foundation:foundation")   // HorizontalPager + BoxWithConstraints
    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}

// Model delivery. By DEFAULT the demo ships WITHOUT the model — it is downloaded once at runtime
// from the manifest (ModelManager.MODEL_MANIFEST_URL) and cached in external files (survives app
// updates). `-PbundleModel` instead stages the exported 22 s int8 model into the APK for a fully
// offline build (the "debug/deploy with model" variant, mirroring -PbundleFonts); ModelManager
// then uses the bundled model directly with no network. The staged copy is gitignored.
// best_s123_p31: phase-3 continuous-corpus model (repetition-suppression root fix + phase-2
// restore). Gated: bench 145/151 (= anchor), learner 85.3% + clean 96.2% (both best-ever),
// suppression ratio 0.43 -> 0.88. See research/CLAUDE.md "Phase-3 concatenation training".
val devModel = rootProject.projectDir.parentFile.parentFile   // sdk/android -> repo root
    .resolve("export/onnx/model_s123_p31_22s.int8.onnx")
val stagedModel = layout.projectDirectory.file("src/main/assets/quranrecite/model.int8.onnx").asFile
if (project.hasProperty("bundleModel")) {
    val bundleDevModel by tasks.registering(Copy::class) {
        onlyIf { devModel.exists() }
        from(devModel) { rename { "model.int8.onnx" } }   // ModelManager expects this name
        into(layout.projectDirectory.dir("src/main/assets/quranrecite"))
        doFirst {
            if (!devModel.exists()) logger.warn(
                "-PbundleModel but no model at $devModel — export it first " +
                    "(python export/export_onnx.py ... --fixed-frames 2200); the APK will ship " +
                    "without a model and rely on the runtime download.")
        }
    }
    tasks.named("preBuild") { dependsOn(bundleDevModel) }
} else {
    // Default (download) build: drop any model staged by a previous -PbundleModel build so the
    // APK genuinely ships without one.
    val unstageDevModel by tasks.registering(Delete::class) { delete(stagedModel) }
    tasks.named("preBuild") { dependsOn(unstageDevModel) }
}

// True-streaming acoustics (Mode.CHAIN battery/latency path). OFF the default distribution; staged
// only by `-PbundleStreaming` (pair with `-PbundleModel` — the two graphs MUST be exported from the
// same checkpoint as the bundled model). Config(streaming=true) then decodes incrementally; absent,
// the SDK falls back to the windowed re-decode. Staged copies are gitignored.
val streamGraphs = rootProject.projectDir.parentFile.parentFile   // repo root
    .let { root -> listOf(
        root.resolve("export/onnx/stream_conv.onnx"),
        root.resolve("export/onnx/stream_encoder.int8.onnx")) }
val stagedStream = streamGraphs.map {
    layout.projectDirectory.file("src/main/assets/quranrecite/${it.name}").asFile }
// v13 fresh-context suffix graph (5 s export of the SAME checkpoint as devModel). Staged by
// -PbundleSuffix (dev/offline); the download build gets it via the manifest's "suffixModel" key.
val suffixGraph = rootProject.projectDir.parentFile.parentFile
    .resolve("export/onnx/model_s123_p31_5s.int8.onnx")
val stagedSuffix = layout.projectDirectory
    .file("src/main/assets/quranrecite/model_suffix.int8.onnx").asFile
if (project.hasProperty("bundleSuffix")) {
    val bundleSuffixGraph by tasks.registering(Copy::class) {
        onlyIf { suffixGraph.exists() }
        from(suffixGraph) { rename { "model_suffix.int8.onnx" } }   // ModelManager.SUFFIX_MODEL
        into(layout.projectDirectory.dir("src/main/assets/quranrecite"))
        doFirst {
            require(suffixGraph.exists()) {
                "-PbundleSuffix but no graph at $suffixGraph — export it first " +
                    "(python export/export_onnx.py --checkpoint <ckpt> --fixed-frames 516 " +
                    "--tag _s123_mic_5s); must match the delivered model's checkpoint."
            }
        }
    }
    tasks.named("preBuild") { dependsOn(bundleSuffixGraph) }
} else {
    val unstageSuffixGraph by tasks.registering(Delete::class) { delete(stagedSuffix) }
    tasks.named("preBuild") { dependsOn(unstageSuffixGraph) }
}

if (project.hasProperty("bundleStreaming")) {
    val bundleStreamGraphs by tasks.registering(Copy::class) {
        onlyIf { streamGraphs.all { it.exists() } }
        from(streamGraphs)
        into(layout.projectDirectory.dir("src/main/assets/quranrecite"))
        doFirst {
            require(streamGraphs.all { it.exists() }) {
                "-PbundleStreaming but missing ${streamGraphs.filterNot { it.exists() }} — export " +
                    "them first (python export/streaming_runtime.py <ckpt>); they must match the " +
                    "bundled model's checkpoint."
            }
        }
    }
    tasks.named("preBuild") { dependsOn(bundleStreamGraphs) }
} else {
    val unstageStreamGraphs by tasks.registering(Delete::class) { delete(stagedStream) }
    tasks.named("preBuild") { dependsOn(unstageStreamGraphs) }
}

// Generate the remote model manifest (version + hosted URL + sha256) for the download build.
// Upload BOTH the model.int8.onnx and model_manifest.json to the hosting URL (a GitHub release
// on the 'model' tag), then bump ModelManager.MODEL_MANIFEST_URL if the location changes.
//   ./gradlew :demo:modelManifest   ->  build/model_manifest.json
// -PmodelVersion / -PmodelDesc set the manifest's version and "what's new" note (shown to users
// on update). Escape any quotes/backslashes so the JSON stays valid.
tasks.register("modelManifest") {
    doLast {
        require(devModel.exists()) { "export the model first: $devModel" }
        fun sha(f: java.io.File) = MessageDigest.getInstance("SHA-256")
            .digest(f.readBytes()).joinToString("") { b -> "%02x".format(b) }
        val shaModel = sha(devModel)
        val version = (project.findProperty("modelVersion") as String?) ?: "best_s123_p31-22s-v3"
        val desc = (project.findProperty("modelDesc") as String?)
            ?: "Major recognizer update: trained on continuous multi-ayah recitation — much " +
               "more reliable tracking when reciting straight through without pauses, and " +
               "better accuracy for learners."
        fun esc(s: String) = s.replace("\\", "\\\\").replace("\"", "\\\"")
        val hostBase = "https://github.com/haztaj/QtAR/releases/download/model"
        // True-streaming graphs (version-coupled to the model). Included in the manifest only if
        // BOTH exist — clients then download them and Mode.CHAIN decodes incrementally.
        val streamConv = devModel.parentFile.resolve("stream_conv.onnx")
        val streamEnc = devModel.parentFile.resolve("stream_encoder.int8.onnx")
        var streamJson = if (streamConv.exists() && streamEnc.exists())
            ""","streamConv":{"url":"$hostBase/stream_conv.onnx","sha256":"${sha(streamConv)}"}""" +
            ""","streamEncoder":{"url":"$hostBase/stream_encoder.int8.onnx","sha256":"${sha(streamEnc)}"}"""
        else ""
        // v13 fresh-context suffix graph (optional; version-coupled to the model like streaming)
        if (suffixGraph.exists())
            streamJson +=
                ""","suffixModel":{"url":"$hostBase/model_suffix.int8.onnx","sha256":"${sha(suffixGraph)}"}"""
        val out = layout.buildDirectory.file("model_manifest.json").get().asFile
        out.parentFile.mkdirs()
        out.writeText(
            """{"version":"${esc(version)}","url":"$hostBase/model.int8.onnx",""" +
                """"sha256":"$shaModel","description":"${esc(desc)}"$streamJson}""")
        logger.lifecycle(
            "wrote $out\n  version=$version  sha256=$shaModel  (${devModel.length() / 1024} KB)\n" +
                "  description=\"$desc\"\n" +
                (if (streamJson.isEmpty()) "  streaming: NOT included (graphs missing in export/onnx)\n"
                 else "  streaming: stream_conv.onnx + stream_encoder.int8.onnx included\n") +
                "upload to the 'model' release:\n" +
                "  $devModel  ->  $hostBase/model.int8.onnx\n" +
                (if (streamJson.isEmpty()) ""
                 else "  $streamConv  ->  $hostBase/stream_conv.onnx\n" +
                      "  $streamEnc  ->  $hostBase/stream_encoder.int8.onnx\n") +
                "  $out  ->  $hostBase/model_manifest.json")
    }
}

// Package the 604 page fonts into a single zip for hosting (upload to the assets host, then set
// MushafFonts.FONTS_URL/SHA256/VERSION). Entries are p1.ttf..p604.ttf at the zip root.
//   ./gradlew :demo:zipMushafFonts   ->  build/mushaf-fonts.zip
tasks.register<Zip>("zipMushafFonts") {
    from(layout.projectDirectory.dir("mushaf-fonts/mushaf/fonts")) { include("p*.ttf") }
    archiveFileName.set("mushaf-fonts.zip")
    destinationDirectory.set(layout.buildDirectory)
    doLast { logger.lifecycle("wrote ${destinationDirectory.get().file(archiveFileName.get()).asFile}") }
}
