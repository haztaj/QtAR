package com.quranrecite.demo.mushaf

import android.content.Context
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.zip.ZipInputStream

/** Where the 604 page fonts come from. [Bundled] = still inside the APK (dev builds, load from
 *  assets); [Downloaded] = fetched once into external files (default/beta builds). */
sealed interface FontSource {
    object Bundled : FontSource
    data class Downloaded(val dir: File) : FontSource
}

/**
 * The 604 KFGQPC page fonts are ~199 MB — too large to ship in the APK (they would re-download on
 * every app update). Instead they are fetched ONCE into external files storage, which survives app
 * updates, so beta testers download them a single time; subsequent app updates are ~KB of code.
 *
 * A dev build may keep the fonts in the APK (`-PbundleFonts`); [ensure] then reports [FontSource.
 * Bundled] and the repository loads straight from assets (no network). The default build excludes
 * `p*.ttf` from the APK (see build.gradle.kts) and downloads [FONTS_URL] once.
 */
object MushafFonts {
    // Hosted zip of the page fonts (p1.ttf .. p604.ttf as root entries). Produce it with
    // `./gradlew :demo:zipMushafFonts` and upload (e.g. a GitHub release on a public assets repo);
    // then fill these in. FONTS_VERSION gates re-download — bump it when the font data changes.
    const val FONTS_URL = "https://github.com/haztaj/QtAR/releases/download/mushaf/mushaf-fonts.zip"
    const val FONTS_SHA256 = "4e2ccdf6774f5fd97d722ec82637a84b2be41eed8ac6ffec8151ff05ecec2f8f"            // optional integrity check ("" = skip)
    const val FONTS_VERSION = "kfgqpc-v2-2"   // tajweed color fonts (bump forces re-download)
    private const val PAGE_COUNT = 604

    private fun dir(context: Context) = File(context.getExternalFilesDir(null), "mushaf/fonts")

    /** True once the fonts are available without a network fetch (bundled or already downloaded). */
    fun isReady(context: Context): Boolean =
        assetHasPageFonts(context) || (versionOf(dir(context)) == FONTS_VERSION && isComplete(dir(context)))

    /** Resolve the page-font source, downloading once if needed. Call OFF the main thread. */
    fun ensure(context: Context, onProgress: (Float) -> Unit): FontSource {
        if (assetHasPageFonts(context)) return FontSource.Bundled          // dev: fonts in the APK

        val dir = dir(context)
        if (versionOf(dir) == FONTS_VERSION && isComplete(dir)) return FontSource.Downloaded(dir)

        check(FONTS_URL.isNotEmpty()) {
            "Page fonts aren't bundled and MushafFonts.FONTS_URL is unset — host the fonts zip " +
                "(./gradlew :demo:zipMushafFonts) and set FONTS_URL, or build with -PbundleFonts."
        }
        downloadAndUnzip(dir, onProgress)
        File(dir, ".version").writeText(FONTS_VERSION)                     // mark complete last
        return FontSource.Downloaded(dir)
    }

    private fun assetHasPageFonts(context: Context): Boolean =
        runCatching { context.assets.open("mushaf/fonts/p1.ttf").close() }.isSuccess

    private fun versionOf(dir: File): String? = File(dir, ".version").takeIf { it.exists() }?.readText()

    private fun isComplete(dir: File): Boolean =
        File(dir, "p1.ttf").exists() && File(dir, "p$PAGE_COUNT.ttf").exists()

    private fun downloadAndUnzip(dir: File, onProgress: (Float) -> Unit) {
        dir.mkdirs()
        File(dir, ".version").delete()                                    // invalidate while writing
        val tmp = File(dir, "fonts.zip.part")
        val conn = (URL(FONTS_URL).openConnection() as HttpURLConnection).apply {
            connectTimeout = 15_000; readTimeout = 30_000
        }
        try {
            val total = conn.contentLengthLong
            val md = if (FONTS_SHA256.isNotEmpty()) MessageDigest.getInstance("SHA-256") else null
            conn.inputStream.use { input ->
                tmp.outputStream().use { out ->
                    val buf = ByteArray(64 * 1024); var read = 0L
                    while (true) {
                        val n = input.read(buf); if (n < 0) break
                        out.write(buf, 0, n); md?.update(buf, 0, n); read += n
                        if (total > 0) onProgress(read.toFloat() / total)
                    }
                }
            }
            if (md != null) {
                val hex = md.digest().joinToString("") { "%02x".format(it) }
                check(hex == FONTS_SHA256) { "fonts zip sha256 mismatch" }
            }
            unzip(tmp, dir)
        } finally {
            conn.disconnect(); tmp.delete()
        }
    }

    private fun unzip(zip: File, dir: File) {
        ZipInputStream(zip.inputStream().buffered()).use { zin ->
            while (true) {
                val e = zin.nextEntry ?: break
                val name = e.name.substringAfterLast('/')                 // flatten any nesting
                if (!e.isDirectory && name.endsWith(".ttf"))
                    File(dir, name).outputStream().use { zin.copyTo(it) }
                zin.closeEntry()
            }
        }
    }
}
