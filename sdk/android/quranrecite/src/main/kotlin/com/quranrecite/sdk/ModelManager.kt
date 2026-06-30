package com.quranrecite.sdk

import android.content.Context
import java.io.File

/** Resolved on-device asset paths the native engine needs. */
data class ModelAssets(
    val modelPath: String,
    val lexiconPath: String,
    val tokensPath: String,
    val filterbankPath: String,
    val hannPath: String,
)

/**
 * Download-on-first-launch asset delivery (the chosen strategy — see
 * docs/sdk-architecture.md §6/§13). Only the ~15 MB ONNX model is fetched + cached; the
 * tiny lexicon/tokens/filterbank ship inside the .aar as a fallback.
 *
 * Responsibilities:
 *  - version manifest (URL + version + sha256) per [corpus];
 *  - cache under context.filesDir/quranrecite/<version>/;
 *  - integrity check (sha256) before use; re-download on mismatch;
 *  - surface progress + a clear "not ready until first download" state to the host.
 */
class ModelManager(private val context: Context, private val corpus: Corpus) {

    private val dir = File(context.filesDir, "quranrecite").apply { mkdirs() }

    fun ensureAsync(
        onProgress: (Float) -> Unit,
        onReady: (ModelAssets) -> Unit,
        onError: (Throwable) -> Unit,
    ) {
        // TODO(impl): fetch the version manifest for `corpus`; if the cached model is
        // missing or fails the sha256 check, download with progress + verify; then resolve
        // ModelAssets (model from cache; lexicon/tokens/filterbank extracted from assets/).
        // Run off the main thread (coroutines/WorkManager). On success -> onReady(assets).
        onError(NotImplementedError("ModelManager download not yet implemented"))
    }

    /** Bundled small assets (extracted from the .aar's assets/ on first use). */
    private fun extractBundled(name: String): String {
        val out = File(dir, name)
        if (!out.exists()) context.assets.open("quranrecite/$name").use { i ->
            out.outputStream().use { o -> i.copyTo(o) }
        }
        return out.absolutePath
    }
}
