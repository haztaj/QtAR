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

// DEV-ONLY: bundle the exported int8 model into the demo so it runs fully offline (no server).
// Mode.CHAIN decodes the rolling 22 s buffer once per hop, so the demo bundles the 22 s
// fixed-window export of best_s123_mic (mic-adapted, 1,057-ayah corpus). The old
// 4 s model (model_4s.int8.onnx) pairs with Mode.AUTO's 4 s windows — swap back if reverting.
// ModelManager prefers a bundled model at assets/quranrecite/model.int8.onnx over downloading.
// Production consumers of the .aar instead use download-on-first-launch. The staged copy is
// gitignored; if the model hasn't been exported yet, the task is skipped and the demo falls
// back to ModelManager's (unset) download path.
val devModel = rootProject.projectDir.parentFile.parentFile   // sdk/android -> repo root
    .resolve("export/onnx/model_s123_mic_22s.int8.onnx")
val bundleDevModel by tasks.registering(Copy::class) {
    onlyIf { devModel.exists() }
    from(devModel) { rename { "model.int8.onnx" } }   // ModelManager expects this name
    into(layout.projectDirectory.dir("src/main/assets/quranrecite"))
    doFirst {
        if (!devModel.exists()) logger.warn(
            "dev model not found at $devModel — run " +
                "`python export/export_onnx.py --fixed-frames 416 --tag _4s`; " +
                "the demo won't run until a model is bundled or hosted.")
    }
}
tasks.named("preBuild") { dependsOn(bundleDevModel) }

// Package the 604 page fonts into a single zip for hosting (upload to the assets host, then set
// MushafFonts.FONTS_URL/SHA256/VERSION). Entries are p1.ttf..p604.ttf at the zip root.
//   ./gradlew :demo:zipMushafFonts   ->  build/mushaf-fonts.zip
tasks.register<Zip>("zipMushafFonts") {
    from(layout.projectDirectory.dir("mushaf-fonts/mushaf/fonts")) { include("p*.ttf") }
    archiveFileName.set("mushaf-fonts.zip")
    destinationDirectory.set(layout.buildDirectory)
    doLast { logger.lifecycle("wrote ${destinationDirectory.get().file(archiveFileName.get()).asFile}") }
}
