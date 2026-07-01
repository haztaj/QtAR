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
    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.2")
}

// DEV-ONLY: bundle the exported int8 model into the demo so it runs fully offline (no server).
// Uses the 4 s sliding-window model (model_4s.int8.onnx) — the SDK feeds 4 s windows, so this
// is ~10x cheaper per hop than the 30 s full-utterance export with identical detections.
// ModelManager prefers a bundled model at assets/quranrecite/model.int8.onnx over downloading.
// Production consumers of the .aar instead use download-on-first-launch. The staged copy is
// gitignored; if the model hasn't been exported yet, the task is skipped and the demo falls
// back to ModelManager's (unset) download path.
val devModel = rootProject.projectDir.parentFile.parentFile   // sdk/android -> repo root
    .resolve("export/onnx/model_4s.int8.onnx")
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
