// Silero VAD (ONNX) streaming voice-activity segmenter. Feed fixed 512-sample @16 kHz chunks;
// `feed` returns true on a speech-END event (>= minSilence of silence after speech) — an ayah
// boundary for paused ayah-by-ayah recitation. Port of silero_vad's VADIterator ('end' path).
#pragma once
#include <string>

namespace quranrecite {

class SileroVAD {
public:
    SileroVAD(const std::string& modelPath, int sampleRate, float threshold, float minSilenceSec);
    ~SileroVAD();
    void reset();
    bool feed(const float* chunk, int n);   // n must be chunkSize()
    static int chunkSize() { return 512; }   // Silero v5 window @ 16 kHz

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace quranrecite
