package com.quranrecite.sdk

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AutomaticGainControl
import android.util.Log
import java.util.concurrent.ArrayBlockingQueue
import kotlin.concurrent.thread

/**
 * Managed 16 kHz mono capture, with the mic recommendations baked in
 * (see docs/mobile-audio-capture.md): VOICE_RECOGNITION source (ASR-tuned, light
 * processing) + AutomaticGainControl (fixes quiet mics). NS/AEC left off — NS can
 * distort madd. Feeds 16-bit PCM blocks to [onPcm].
 *
 * Capture and inference are DECOUPLED: a reader thread does nothing but `read()` + enqueue
 * (so it always keeps pace with the mic), and a separate worker thread drains the queue into
 * [onPcm] (which runs the model). Previously [onPcm] ran inline on the reader thread, so an
 * inference stall (ORT warmup, GC, a long stream-buffer decode) stalled `read()` and AudioRecord
 * silently dropped samples — ~30 % of a session went missing, punching holes in the audio and
 * wrecking detection. The queue absorbs those bursts; a large AudioRecord buffer adds headroom.
 */
internal class AudioCapture(private val onPcm: (ShortArray, Int) -> Unit) {
    private val sampleRate = 16000
    private val blockSize = 512   // 32 ms @ 16 kHz, matches the engine's hop granularity
    private var record: AudioRecord? = null
    @Volatile private var running = false
    private var reader: Thread? = null
    private var worker: Thread? = null
    // ~2 s of headroom; drop-oldest if inference ever falls behind realtime (logged).
    private val queue = ArrayBlockingQueue<ShortArray>(sampleRate * 2 / blockSize)

    fun start() {
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        // 1 s AudioRecord buffer: absorbs transient reader stalls without dropping at the driver.
        val bufBytes = maxOf(minBuf, sampleRate * 2)
        val rec = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,   // ASR-tuned source
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufBytes)
        // Enable AGC if the device supports it (the quiet-mic fix at the source).
        if (AutomaticGainControl.isAvailable()) {
            AutomaticGainControl.create(rec.audioSessionId)?.enabled = true
        }
        record = rec
        running = true
        queue.clear()
        rec.startRecording()

        // Reader: read + enqueue only. Never blocked by inference.
        reader = thread(name = "quranrecite-capture") {
            val buf = ShortArray(blockSize)
            var dropped = 0
            while (running) {
                val n = rec.read(buf, 0, buf.size)
                if (n > 0) {
                    val block = if (n == buf.size) buf.copyOf() else buf.copyOf(n)
                    if (!queue.offer(block)) {           // full -> inference is behind realtime
                        queue.poll()                     // drop the oldest, keep the newest
                        queue.offer(block)
                        if (++dropped % 32 == 0) Log.w("QuranRecite",
                            "audio queue full — inference behind realtime, dropped $dropped blocks")
                    }
                } else if (n < 0) break
            }
        }

        // Worker: drain the queue into the engine (model inference lives here). Stops promptly on
        // stop() — any queued audio after the user stops is dropped (not fed/decoded post-stop).
        worker = thread(name = "quranrecite-infer") {
            while (running) {
                val block = queue.poll(100, java.util.concurrent.TimeUnit.MILLISECONDS) ?: continue
                onPcm(block, sampleRate)
            }
        }
    }

    // Blocks until both threads have exited, so no onPcm callback (nativeFeed / debug write) can
    // fire after stop() returns — the caller may then close/destroy resources safely.
    fun stop() {
        running = false
        record?.stop()                 // unblock a pending rec.read()
        reader?.join(500); reader = null
        worker?.join(500); worker = null
        record?.release()              // release only after the reader has exited
        record = null
        queue.clear()
    }
}
