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
    // Waqf-segment progress within `active` (Mode.CHAIN): "segment N of M". activeSegmentCount
    // == 0 means no segment info (non-Chain mode / no active ayah); 1 = an unsegmented ayah;
    // N = split into N waqf segments. activeSegment is the current one (1-based), or 0.
    val activeSegment: Int = 0,
    val activeSegmentCount: Int = 0,
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
            return HighlightState(confirmed, pending, active, upNext,
                o.optInt("activeSegment", 0), o.optInt("activeSegmentCount", 0))
        }
    }
}

/** Coverage of the bundled model/lexicon. */
enum class Corpus { JUZ_AMMA, FULL_QURAN }

/** Detection engine. AUTO = merged sliding+stream matchers (ayah units); CHAIN = the
 *  unit-chain decoder over waqf segments (research winning design; needs the 22 s model
 *  + unit_phonemes.json in the asset bundle). SLIDING/BUFFER are legacy single-matcher
 *  modes. The native ordinal must match quranrecite::Mode (types.h). */
enum class Mode { AUTO, SLIDING, BUFFER, CHAIN }

data class Config(
    val corpus: Corpus = Corpus.JUZ_AMMA,
    val mode: Mode = Mode.AUTO,
    val managedCapture: Boolean = true,   // SDK opens the mic with recommended settings
    val windowSec: Float = 4.0f,
    val hopSec: Float = 1.0f,
    // Chain-mode window fire threshold. 0.30 = the research reference for clean decodes;
    // consumer phone mics decode at ~30% PER and need ~0.45 (verified on a live session —
    // the vote + deferral-assembly layers absorb the extra junk fires).
    val chainCost: Float = 0.45f,
    // Phase-2 posterior-aware scoring floor: 1.0 = off (hard distance); ~0 softens mismatches
    // the model nearly picked (a ~+1.7 aligned-hit win in the ~30% PER phone regime, free on
    // clean audio). Needs a model that emits posteriors (Mode.CHAIN decodes them per hop).
    val chainSubMin: Float = 1.0f,
    // True streaming acoustics (Mode.CHAIN): prefer the incremental StreamingModel (decode only the
    // new audio each hop — ~11x cheaper decode, RTF 0.484->0.043). ON by default: the graphs are
    // delivered by the manifest alongside the model (or bundled via -PbundleStreaming), and it
    // falls back to the windowed re-decode whenever they are absent (first launch offline, or a
    // manifest without them), so leaving it on is safe. See export/streaming-export-plan.md.
    val streaming: Boolean = true,
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
        /** A newly-released model was downloaded, replacing the previous one. [description] is
         *  the manifest's "what's new" note (may be empty). Fired before [onModelReady]. */
        fun onModelUpdated(version: String, description: String) {}
        fun onModelReady() {}
        fun onError(error: Throwable) {}
    }

    private var nativeHandle: Long = 0
    private var listener: Listener? = null
    @Volatile private var modelPath: String? = null   // resolved at onModelReady (for debug info)
    private var capture: AudioCapture? = null
    private var debugRec: DebugWavRecorder? = null
    @Volatile private var debugLogging = false   // logcat: engine assets + native per-hop stats
    @Volatile private var recording = false      // dump each session's PCM to a WAV (takes effect at start)
    private var lastRecordingPath: String? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    fun setListener(listener: Listener) { this.listener = listener }

    /** Toggle runtime debug logging (host UI). Gates the SDK's logcat lines and the native engine's
     *  per-hop/VAD/commit logs (tag "QuranReciteCore"). Off by default; safe to call any time. */
    fun setDebugLogging(enabled: Boolean) {
        debugLogging = enabled
        if (nativeHandle != 0L) nativeSetDebug(nativeHandle, enabled)
    }

    /** Toggle dumping the exact 16 kHz PCM fed to the engine to a WAV (host UI). Applies to the next
     *  [start]; the file lands under getExternalFilesDir and is returned by [lastRecording]. */
    fun setRecording(enabled: Boolean) { recording = enabled }

    /** Absolute path of the most recently recorded session WAV, or null. */
    fun lastRecording(): String? = lastRecordingPath

    /** Basename of the resolved model (e.g. "best_s123_mic_clean-22s-v1.onnx"), for debug
     *  info; null before [Listener.onModelReady]. */
    fun modelName(): String? = modelPath?.substringAfterLast('/')

    /** Ensures the model is present (downloads on first launch), then creates the engine. */
    fun prepare() {
        ModelManager(context, config.corpus).ensureAsync(
            onProgress = { f -> mainHandler.post { listener?.onModelDownloadProgress(f) } },
            onReady = { assets ->                       // worker thread: build engine here
                modelPath = assets.modelPath
                if (debugLogging) android.util.Log.i("QuranRecite",
                    "engine assets: model=${assets.modelPath.substringAfterLast('/')} " +
                        "vad=${if (assets.vadPath.isEmpty()) "<none>" else assets.vadPath.substringAfterLast('/')} " +
                        "ambiguous=${assets.ambiguousPath.isNotEmpty()}")
                // Streaming graphs only if requested AND bundled (else "" -> windowed re-decode).
                val streamConv = if (config.streaming) assets.streamConvPath else ""
                val streamEnc = if (config.streaming) assets.streamEncoderPath else ""
                if (debugLogging && config.streaming) android.util.Log.i("QuranRecite",
                    "streaming acoustics: ${if (streamEnc.isEmpty()) "<not bundled -> windowed>"
                        else "on"}")
                nativeHandle = nativeCreate(
                    assets.modelPath, assets.lexiconPath, assets.tokensPath,
                    assets.filterbankPath, assets.hannPath, assets.ambiguousPath, assets.vadPath,
                    config.mode.ordinal, assets.unitPhonemesPath, config.chainCost,
                    config.chainSubMin, streamConv, streamEnc)
                nativeSetDebug(nativeHandle, debugLogging)      // carry the current flag to the engine
                mainHandler.post { listener?.onModelReady() }
            },
            onError = { e -> mainHandler.post { listener?.onError(e) } },
            onModelUpdate = { version, desc ->
                mainHandler.post { listener?.onModelUpdated(version, desc) }
            },
        )
    }

    /** Start managed mic capture (requires RECORD_AUDIO granted). No-op in feed mode. */
    fun start() {
        check(nativeHandle != 0L) { "call prepare() and await onModelReady() first" }
        if (config.managedCapture) {
            val rec = if (recording) DebugWavRecorder(context) else null
            capture = AudioCapture { pcm, sr ->
                rec?.write(pcm)
                nativeFeed(nativeHandle, pcm, sr)
            }.also { it.start() }
            debugRec = rec
        }
    }

    fun stop() {
        capture?.stop(); capture = null       // joins threads before we close the recorder
        debugRec?.close()?.let {
            lastRecordingPath = it
            if (debugLogging) android.util.Log.i("QuranRecite", "debug audio saved: $it")
        }
        debugRec = null
    }

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
        filterbankPath: String, hannPath: String, ambiguousPath: String, vadPath: String,
        mode: Int, unitPhonemesPath: String, chainCost: Float, chainSubMin: Float,
        streamConvPath: String, streamEncoderPath: String): Long
    private external fun nativeFeed(handle: Long, pcm: ShortArray, sampleRate: Int)
    private external fun nativeReset(handle: Long)
    private external fun nativeSetDebug(handle: Long, enabled: Boolean)
    private external fun nativeDestroy(handle: Long)

    companion object {
        init { System.loadLibrary("quranrecite_jni") }
    }
}

/** Streams captured 16 kHz mono PCM16 to a WAV (patching sizes on close), for the debug
 *  "record session" toggle. Created only while recording is enabled (see [setRecording]). */
private class DebugWavRecorder(context: Context) {
    private val file = java.io.File(
        context.getExternalFilesDir(null),
        "session_${System.currentTimeMillis()}.wav")
    private val out = java.io.RandomAccessFile(file, "rw").apply {
        write(ByteArray(44))   // placeholder header, patched in close()
    }
    private var dataBytes = 0
    private var closed = false

    @Synchronized fun write(pcm: ShortArray) {
        if (closed) return                          // ignore a late write after close (no EBADF)
        val b = java.nio.ByteBuffer.allocate(pcm.size * 2).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (s in pcm) b.putShort(s)
        out.write(b.array()); dataBytes += pcm.size * 2
    }

    @Synchronized fun close(): String {
        closed = true
        val sr = 16000; val ch = 1; val bps = 16
        val byteRate = sr * ch * bps / 8
        val h = java.nio.ByteBuffer.allocate(44).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        h.put("RIFF".toByteArray()); h.putInt(36 + dataBytes); h.put("WAVE".toByteArray())
        h.put("fmt ".toByteArray()); h.putInt(16); h.putShort(1); h.putShort(ch.toShort())
        h.putInt(sr); h.putInt(byteRate); h.putShort((ch * bps / 8).toShort()); h.putShort(bps.toShort())
        h.put("data".toByteArray()); h.putInt(dataBytes)
        out.seek(0); out.write(h.array()); out.close()
        return file.absolutePath
    }
}
