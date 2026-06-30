// QuranRecite SDK — main on-device detector (public C++ API).
//
// Pipeline:  PCM in -> 16k log-mel -> ONNX Runtime (int8) -> CTC greedy phonemes
//            -> sliding-window matcher/segmenter -> ayah events out.
//
// The detector is a pure processing engine: the host feeds PCM (or the platform layer's
// managed capture does). Events are delivered via the callback; the caller marshals to a
// UI thread as needed (the platform wrappers do this).
#pragma once
#include <memory>
#include "quranrecite/types.h"

namespace quranrecite {

class Detector {
public:
    explicit Detector(const Config& config);
    ~Detector();
    Detector(Detector&&) noexcept;
    Detector& operator=(Detector&&) noexcept;

    // Register the event sink. Called from the engine's worker thread.
    void setEventCallback(EventCallback cb);

    // Feed mono PCM (float -1..1). `sampleRate` may differ from 16k (resampled internally).
    // Thread-safe wrt the engine; intended to be called from one capture thread.
    void feed(const float* pcm, std::size_t numSamples, int sampleRate);

    // Convenience for 16-bit PCM (e.g. Android AudioRecord ENCODING_PCM_16BIT).
    void feedPcm16(const short* pcm, std::size_t numSamples, int sampleRate);

    // Clear the rolling buffer + sequential context (start a fresh recitation session).
    void reset();

    static const char* version();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace quranrecite
