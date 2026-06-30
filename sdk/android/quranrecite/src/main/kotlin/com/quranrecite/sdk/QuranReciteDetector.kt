package com.quranrecite.sdk

import android.content.Context

/** Stable ayah identifier (surah:ayah). The host app owns mushaf text/rendering. */
data class AyahId(val surah: Int, val ayah: Int)

/** Coverage of the bundled model/lexicon. */
enum class Corpus { JUZ_AMMA, FULL_QURAN }

enum class Mode { SLIDING, BUFFER }

data class Config(
    val corpus: Corpus = Corpus.JUZ_AMMA,
    val mode: Mode = Mode.SLIDING,
    val managedCapture: Boolean = true,   // SDK opens the mic with recommended settings
    val windowSec: Float = 4.0f,
    val hopSec: Float = 1.0f,
)

/**
 * On-device Quran recitation detector. Import into any Android app:
 *
 *   val detector = QuranReciteDetector(context, Config())
 *   detector.setListener(object : QuranReciteDetector.Listener {
 *       override fun onAyahDetected(ayah: AyahId, confidence: Float) { highlight(ayah) }
 *       override fun onAyahAdvance(from: AyahId, to: AyahId) { highlight(to) }
 *   })
 *   detector.start()         // managed capture; or detector.feed(pcm, sr)
 *
 * Model/lexicon are fetched on first use (download-on-first-launch) via [ModelManager];
 * the detector is ready after [Listener.onModelReady].
 */
class QuranReciteDetector(
    private val context: Context,
    private val config: Config = Config(),
) {
    interface Listener {
        fun onAyahDetected(ayah: AyahId, confidence: Float) {}
        fun onAyahAdvance(from: AyahId, to: AyahId) {}
        fun onModelDownloadProgress(fraction: Float) {}
        fun onModelReady() {}
        fun onError(error: Throwable) {}
    }

    private var nativeHandle: Long = 0
    private var listener: Listener? = null
    private var capture: AudioCapture? = null

    fun setListener(listener: Listener) { this.listener = listener }

    /** Ensures the model is present (downloads on first launch), then creates the engine. */
    fun prepare() {
        ModelManager(context, config.corpus).ensureAsync(
            onProgress = { listener?.onModelDownloadProgress(it) },
            onReady = { assets ->
                nativeHandle = nativeCreate(
                    assets.modelPath, assets.lexiconPath, assets.tokensPath,
                    assets.filterbankPath, assets.hannPath)
                listener?.onModelReady()
            },
            onError = { listener?.onError(it) },
        )
    }

    /** Start managed mic capture (requires RECORD_AUDIO granted). No-op in feed mode. */
    fun start() {
        check(nativeHandle != 0L) { "call prepare() and await onModelReady() first" }
        if (config.managedCapture) {
            capture = AudioCapture { pcm, sr -> nativeFeed(nativeHandle, pcm, sr) }.also { it.start() }
        }
    }

    fun stop() { capture?.stop(); capture = null }

    /** Advanced: feed mono 16-bit PCM yourself (app-managed capture). */
    fun feed(pcm: ShortArray, sampleRate: Int) {
        if (nativeHandle != 0L) nativeFeed(nativeHandle, pcm, sampleRate)
    }

    /** New recitation session — clears rolling buffer + sequential context. */
    fun reset() { if (nativeHandle != 0L) nativeReset(nativeHandle) }

    fun release() {
        stop()
        if (nativeHandle != 0L) { nativeDestroy(nativeHandle); nativeHandle = 0 }
    }

    // Called from JNI (worker thread). Wrappers marshal to the main thread for the host.
    @Suppress("unused")
    private fun emitDetected(surah: Int, ayah: Int, confidence: Float) =
        listener?.onAyahDetected(AyahId(surah, ayah), confidence)

    @Suppress("unused")
    private fun emitAdvance(fromS: Int, fromA: Int, toS: Int, toA: Int) =
        listener?.onAyahAdvance(AyahId(fromS, fromA), AyahId(toS, toA))

    private external fun nativeCreate(
        modelPath: String, lexiconPath: String, tokensPath: String,
        filterbankPath: String, hannPath: String): Long
    private external fun nativeFeed(handle: Long, pcm: ShortArray, sampleRate: Int)
    private external fun nativeReset(handle: Long)
    private external fun nativeDestroy(handle: Long)

    companion object {
        init { System.loadLibrary("quranrecite_jni") }
    }
}
