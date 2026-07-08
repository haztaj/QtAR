package com.quranrecite.sdk

import android.content.Context
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlin.concurrent.thread

/** Resolved on-device asset paths the native engine needs. */
data class ModelAssets(
    val modelPath: String,
    val lexiconPath: String,
    val tokensPath: String,
    val filterbankPath: String,
    val hannPath: String,
    val ambiguousPath: String,   // Stage-3 confusable map; "" if not bundled (deferral off)
    val vadPath: String,         // Silero VAD; "" if not bundled (no paused-recitation reset)
    val unitPhonemesPath: String,// waqf-segment unit lexicon; "" if not bundled (Chain mode off)
)

/**
 * Download-on-first-launch asset delivery (the chosen strategy — see
 * docs/sdk-architecture.md §6/§13). The four small assets (lexicon, tokens, mel filterbank,
 * Hann window) ship inside the .aar under assets/quranrecite/ and are extracted to filesDir.
 * The ~15 MB ONNX model is fetched once and cached under
 * filesDir/quranrecite/<version>/, verified by sha256.
 *
 * Dev/offline path: if a model is *also* bundled at assets/quranrecite/model.int8.onnx it is
 * used directly (no network) — handy for the demo before the artifact is hosted.
 */
class ModelManager(private val context: Context, private val corpus: Corpus) {

    private val root = File(context.filesDir, "quranrecite").apply { mkdirs() }
    // Extracted assets live in a VERSIONED subdir: extractBundled skips existing files, so
    // without the version key a corpus/model update would silently reuse stale extractions.
    private val versionDir = File(root, MODEL_VERSION).apply { mkdirs() }

    /** Resolve all assets off the main thread; callbacks fire on the worker thread. */
    fun ensureAsync(
        onProgress: (Float) -> Unit,
        onReady: (ModelAssets) -> Unit,
        onError: (Throwable) -> Unit,
    ) {
        thread(name = "quranrecite-model") {
            try {
                val lexicon = extractBundled("ayah_phonemes.json")
                val tokens = extractBundled("tokens.txt")
                val filterbank = extractBundled("mel_filterbank.bin")
                val hann = extractBundled("hann_window.bin")
                // Optional Stage-3 confusable map — enables ambiguity deferral if bundled.
                val ambiguous = if (assetExists("quranrecite/ambiguous_ayat.json"))
                    extractBundled("ambiguous_ayat.json") else ""
                // Optional Silero VAD — enables paused ayah-by-ayah boundary reset if bundled.
                val vad = if (assetExists("quranrecite/silero_vad.onnx"))
                    extractBundled("silero_vad.onnx") else ""
                // Optional waqf-segment unit lexicon — enables Mode.CHAIN if bundled.
                val units = if (assetExists("quranrecite/unit_phonemes.json"))
                    extractBundled("unit_phonemes.json") else ""
                val model = resolveModel(onProgress)
                onReady(ModelAssets(model, lexicon, tokens, filterbank, hann, ambiguous, vad, units))
            } catch (t: Throwable) {
                onError(t)
            }
        }
    }

    /** Bundled dev model → else cached (verified) → else download + verify + cache. */
    private fun resolveModel(onProgress: (Float) -> Unit): String {
        if (assetExists("quranrecite/$BUNDLED_MODEL")) return extractBundled(BUNDLED_MODEL)

        val cached = File(versionDir, BUNDLED_MODEL)
        if (cached.exists() && (MODEL_SHA256.isEmpty() || sha256(cached) == MODEL_SHA256)) {
            return cached.absolutePath
        }
        check(MODEL_URL.isNotEmpty()) {
            "No bundled model and MODEL_URL is unset — host the model artifact or bundle it " +
                "at assets/quranrecite/$BUNDLED_MODEL for development."
        }
        download(MODEL_URL, cached, onProgress)
        if (MODEL_SHA256.isNotEmpty() && sha256(cached) != MODEL_SHA256) {
            cached.delete()
            error("Downloaded model failed sha256 verification")
        }
        return cached.absolutePath
    }

    private fun download(url: String, dest: File, onProgress: (Float) -> Unit) {
        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 15_000; readTimeout = 30_000
        }
        try {
            val total = conn.contentLengthLong
            val tmp = File(dest.parentFile, dest.name + ".part")
            conn.inputStream.use { input ->
                tmp.outputStream().use { output ->
                    val buf = ByteArray(64 * 1024)
                    var read = 0L
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        output.write(buf, 0, n)
                        read += n
                        if (total > 0) onProgress((read.toDouble() / total).toFloat())
                    }
                }
            }
            if (!tmp.renameTo(dest)) { tmp.copyTo(dest, overwrite = true); tmp.delete() }
        } finally {
            conn.disconnect()
        }
    }

    private fun assetExists(path: String): Boolean =
        runCatching { context.assets.open(path).close() }.isSuccess

    /** Copy a file from the .aar's assets/quranrecite/ into the versioned dir (once). */
    private fun extractBundled(name: String): String {
        val out = File(versionDir, name)
        if (!out.exists()) context.assets.open("quranrecite/$name").use { i ->
            out.outputStream().use { o -> i.copyTo(o) }
        }
        return out.absolutePath
    }

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { s ->
            val buf = ByteArray(64 * 1024)
            while (true) { val n = s.read(buf); if (n < 0) break; md.update(buf, 0, n) }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        // Remote model artifact (download-on-first-launch). Fill URL + sha256 when the
        // versioned artifact is hosted; until then bundle a dev model (see class doc).
        const val MODEL_VERSION = "best_s123_mic_clean-22s-v1"   // mic-adapted + RetaSy-cleaned
        const val MODEL_URL = ""      // TODO: hosted, versioned model URL
        const val MODEL_SHA256 = ""   // TODO: expected sha256 (empty = skip verify)
        const val BUNDLED_MODEL = "model.int8.onnx"
    }
}
