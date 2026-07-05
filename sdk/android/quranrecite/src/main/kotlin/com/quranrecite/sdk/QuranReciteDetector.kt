package com.quranrecite.sdk

import android.content.Context
import android.os.Handler
import android.os.Looper
import org.json.JSONObject

/** Stable ayah identifier (surah:ayah). The host app owns mushaf text/rendering. */
data class AyahId(val surah: Int, val ayah: Int) {
    companion object {
        fun parse(sa: String): AyahId {
            val (s, a) = sa.split(":")
            return AyahId(s.toInt(), a.toInt())
        }
    }
}

/** Why a detection is being deferred instead of highlighted (see [HighlightState]). */
enum class PendingReason { AWAIT_SUCCESSOR, NEEDS_CHOICE }

/** A deferred, ambiguous detection: the confusable [options] and why it's held. */
data class HighlightPending(val ayah: AyahId?, val options: List<AyahId>, val reason: PendingReason)

/**
 * The centralized, render-ready highlight state — the SDK's primary output contract.
 * The engine emits one immutable snapshot per change; the UI just renders it (ambiguity is
 * deferred, never guessed). Mirrors matcher/highlight_controller.py (conformance-pinned).
 */
data class HighlightState(
    val confirmed: List<AyahId>,   // settled + highlighted, in confirm order
    val pending: HighlightPending?,// awaiting disambiguation (deferred), or null
    val active: AyahId?,           // the ayah just detected (lighter highlight), or null
    val upNext: AyahId? = null,    // predicted next ayah, set once `active` nears completion (darker)
) {
    companion object {
        fun fromJson(json: String): HighlightState {
            val o = JSONObject(json)
            val confirmed = o.getJSONArray("confirmed").let { a ->
                List(a.length()) { AyahId.parse(a.getString(it)) }
            }
            val pending = o.optJSONObject("pending")?.let { p ->
                val opts = p.getJSONArray("options").let { a ->
                    List(a.length()) { AyahId.parse(a.getString(it)) }
                }
                val reason = if (p.getString("reason") == "needs_choice")
                    PendingReason.NEEDS_CHOICE else PendingReason.AWAIT_SUCCESSOR
                val ayah = if (p.isNull("ayah")) null else AyahId.parse(p.getString("ayah"))
                HighlightPending(ayah, opts, reason)
            }
            val active = if (o.isNull("active")) null else AyahId.parse(o.getString("active"))
            val upNext = if (o.isNull("upNext")) null else AyahId.parse(o.getString("upNext"))
            return HighlightState(confirmed, pending, active, upNext)
        }
    }
}

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
        /**
         * The centralized highlight snapshot — the primary contract. Render this wholesale;
         * it already handles deferral + ambiguity so no per-UI logic is needed.
         */
        fun onHighlightState(state: HighlightState) {}
        // Granular events (kept for back-compat / custom flows). Prefer onHighlightState.
        fun onAyahDetected(ayah: AyahId, confidence: Float) {}
        fun onAyahAdvance(from: AyahId, to: AyahId) {}
        fun onModelDownloadProgress(fraction: Float) {}
        fun onModelReady() {}
        fun onError(error: Throwable) {}
    }

    private var nativeHandle: Long = 0
    private var listener: Listener? = null
    private var capture: AudioCapture? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    fun setListener(listener: Listener) { this.listener = listener }

    /** Ensures the model is present (downloads on first launch), then creates the engine. */
    fun prepare() {
        ModelManager(context, config.corpus).ensureAsync(
            onProgress = { f -> mainHandler.post { listener?.onModelDownloadProgress(f) } },
            onReady = { assets ->                       // worker thread: build engine here
                nativeHandle = nativeCreate(
                    assets.modelPath, assets.lexiconPath, assets.tokensPath,
                    assets.filterbankPath, assets.hannPath, assets.ambiguousPath, assets.vadPath)
                mainHandler.post { listener?.onModelReady() }
            },
            onError = { e -> mainHandler.post { listener?.onError(e) } },
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

    // Called from JNI (engine worker thread). Marshal to the main thread for the host.
    // NOTE: block body (returns Unit -> JVM void); the JNI lookup expects "(IIF)V"/"(IIII)V".
    @Suppress("unused")
    private fun emitDetected(surah: Int, ayah: Int, confidence: Float) {
        mainHandler.post { listener?.onAyahDetected(AyahId(surah, ayah), confidence) }
    }

    @Suppress("unused")
    private fun emitAdvance(fromS: Int, fromA: Int, toS: Int, toA: Int) {
        mainHandler.post { listener?.onAyahAdvance(AyahId(fromS, fromA), AyahId(toS, toA)) }
    }

    // Called from JNI with the serialized snapshot; parse + marshal to the main thread.
    @Suppress("unused")
    private fun emitHighlight(json: String) {
        val state = HighlightState.fromJson(json)
        mainHandler.post { listener?.onHighlightState(state) }
    }

    private external fun nativeCreate(
        modelPath: String, lexiconPath: String, tokensPath: String,
        filterbankPath: String, hannPath: String, ambiguousPath: String, vadPath: String): Long
    private external fun nativeFeed(handle: Long, pcm: ShortArray, sampleRate: Int)
    private external fun nativeReset(handle: Long)
    private external fun nativeDestroy(handle: Long)

    companion object {
        init { System.loadLibrary("quranrecite_jni") }
    }
}
