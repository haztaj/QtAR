plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    `maven-publish`
}

android {
    namespace = "com.quranrecite.sdk"
    compileSdk = 34

    defaultConfig {
        minSdk = 24                       // NNAPI + UNPROCESSED audio source
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += "-DANDROID_STL=c++_shared"
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
