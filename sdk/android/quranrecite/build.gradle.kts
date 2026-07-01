plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    `maven-publish`
}

// ONNX Runtime C++ headers + per-ABI .so are unpacked here from the (non-prefab) ORT AAR
// by the extractOrt task; CMake reads them via -DORT_DIR (see below + src/main/cpp).
val ortDir = layout.buildDirectory.dir("ort")

// A resolvable configuration holding just the ORT AAR artifact (no transitive deps).
val ortExtract: Configuration by configurations.creating
dependencies { ortExtract("com.microsoft.onnxruntime:onnxruntime-android:1.18.0@aar") }

android {
    namespace = "com.quranrecite.sdk"
    compileSdk = 34

    ndkVersion = "26.1.10909125"

    defaultConfig {
        minSdk = 24                       // NNAPI + UNPROCESSED audio source
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += "-DANDROID_STL=c++_shared"
                // ORT headers + per-ABI libonnxruntime.so, unpacked from the AAR by extractOrt.
                arguments += "-DORT_DIR=${ortDir.get().asFile.path.replace('\\', '/')}"
            }
        }
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release { isMinifyEnabled = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    publishing { singleVariant("release") }
}

dependencies {
    // ONNX Runtime for Android (provides the prefab onnxruntime native module + Java API).
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}

// Published to Maven (Central or private) as the importable .aar — see docs/sdk-architecture.md §6.
publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = "com.quranrecite"
            artifactId = "quranrecite-sdk"
            version = "0.1.0"
            afterEvaluate { from(components["release"]) }
        }
    }
}

// Bundle the small runtime assets (lexicon, tokens, DSP filterbank/window) into the .aar.
// They're produced by `python conformance/generate.py`; the ~15 MB model is downloaded at
// runtime (ModelManager). Copied assets are gitignored — this task repopulates them.
val bundleAssets by tasks.registering(Copy::class) {
    val repoRoot = rootProject.projectDir.parentFile.parentFile   // sdk/android -> repo root
    val assetsSrc = File(repoRoot, "conformance/assets")
    doFirst {
        require(File(assetsSrc, "mel_filterbank.bin").exists()) {
            "Missing conformance/assets/*.bin — run `python conformance/generate.py` first."
        }
    }
    from(assetsSrc) {
        include("ayah_phonemes.json", "tokens.txt", "mel_filterbank.bin", "hann_window.bin")
    }
    into(layout.projectDirectory.dir("src/main/assets/quranrecite"))
}

// Unpack the ORT AAR's headers/ + jni/<abi>/libonnxruntime.so so CMake can link the core
// against ONNX Runtime (the AAR isn't prefab-packaged). The .so is provided at runtime by
// the onnxruntime-android dependency in the app; this is link-time only.
val extractOrt by tasks.registering(Copy::class) {
    from(zipTree(ortExtract.singleFile)) { include("headers/**", "jni/**") }
    into(ortDir)
}
tasks.named("preBuild") { dependsOn(bundleAssets, extractOrt) }
