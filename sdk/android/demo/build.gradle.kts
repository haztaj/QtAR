import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.0"
}

android {
    namespace = "com.quranrecite.demo"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.quranrecite.demo"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
        // Ship only real-phone (arm64) + emulator (x86_64) ABIs. Drops x86/armeabi-v7a
        // (~34 MB of unused ONNX Runtime) and the x86 mismatch (no x86 quranrecite_jni).
        // A Play release would instead use an App Bundle for per-device ABI delivery.
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
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
val devModel = rootProject.projectDir.parentFile.parentFile   // sdk/android -> repo root
    .resolve("export/onnx/model_s123_mic_clean_22s.int8.onnx")
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

// Generate the remote model manifest (version + hosted URL + sha256) for the download build.
// Upload BOTH the model.int8.onnx and model_manifest.json to the hosting URL (a GitHub release
// on the 'model' tag), then bump ModelManager.MODEL_MANIFEST_URL if the location changes.
//   ./gradlew :demo:modelManifest   ->  build/model_manifest.json
// -PmodelVersion / -PmodelDesc set the manifest's version and "what's new" note (shown to users
// on update). Escape any quotes/backslashes so the JSON stays valid.
tasks.register("modelManifest") {
    doLast {
        require(devModel.exists()) { "export the model first: $devModel" }
        val sha = MessageDigest.getInstance("SHA-256")
            .digest(devModel.readBytes()).joinToString("") { b -> "%02x".format(b) }
        val version = (project.findProperty("modelVersion") as String?) ?: "best_s123_mic_clean-22s-v1"
        val desc = (project.findProperty("modelDesc") as String?)
            ?: "Mic-adapted + learner-data-cleaned recognizer (surahs 1–3 + Juz Amma). " +
               "More accurate detection on phone microphones."
        fun esc(s: String) = s.replace("\\", "\\\\").replace("\"", "\\\"")
        val hostBase = "https://github.com/haztaj/QtAR/releases/download/model"
        val out = layout.buildDirectory.file("model_manifest.json").get().asFile
        out.parentFile.mkdirs()
        out.writeText(
            """{"version":"${esc(version)}","url":"$hostBase/model.int8.onnx",""" +
                """"sha256":"$sha","description":"${esc(desc)}"}""")
        logger.lifecycle(
            "wrote $out\n  version=$version  sha256=$sha  (${devModel.length() / 1024} KB)\n" +
                "  description=\"$desc\"\n" +
                "upload to the 'model' release:\n" +
                "  $devModel  ->  $hostBase/model.int8.onnx\n" +
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
