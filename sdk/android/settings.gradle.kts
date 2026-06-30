pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "QuranReciteSDK"
include(":quranrecite")   // the SDK library (.aar)
include(":demo")          // the Compose demo app
