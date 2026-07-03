package com.quranrecite.sdk

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AutomaticGainControl
import kotlin.concurrent.thread

/**
 * Managed 16 kHz mono capture, with the mic recommendations baked in
 * (see docs/mobile-audio-capture.md): VOICE_RECOGNITION source (ASR-tuned, light
 * processing) + AutomaticGainControl (fixes quiet mics). NS/AEC left off — NS can
 * distort madd. Feeds 16-bit PCM blocks to [onPcm].
 */
internal class AudioCapture(private val onPcm: (ShortArray, Int) -> Unit) {
    private val sampleRate = 16000
    private var record: AudioRecord? = null
    @Volatile private var running = false

    fun start() {
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val blockSize = 512   // 32 ms @ 16 kHz, matches the engine's hop granularity
        val rec = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,   // ASR-tuned source
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, blockSize * 8))
        // Enable AGC if the device supports it (the quiet-mic fix at the source).
        if (AutomaticGainControl.isAvailable()) {
            AutomaticGainControl.create(rec.audioSessionId)?.enabled = true
        }
        record = rec
        running = true
        rec.startRecording()
        thread(name = "quranrecite-capture") {
            val buf = ShortArray(blockSize)
            while (running) {
                val n = rec.read(buf, 0, buf.size)
                if (n > 0) onPcm(if (n == buf.size) buf else buf.copyOf(n), sampleRate)
                else if (n < 0) break
            }
        }
    }

    fun stop() {
        running = false
        record?.run { stop(); release() }
        record = null
    }
}
