// Streaming Emformer+CTC inference — incremental decode of only the NEW audio each call,
// replacing Mode::Chain's per-hop 22 s window RE-decode. Two ONNX graphs: a dynamic-T
// Conv2dSubsampling (`stream_conv.onnx`) and a fixed-shape STATEFUL Emformer step
// (`stream_encoder.int8.onnx`, chunk + 48 state tensors -> log_probs + 48 state tensors).
// Byte-faithful port of export/streaming_runtime.py::StreamingRuntime (conv boundary cache +
// state threading + CTC collapse across chunk boundaries). Parity-gated against it.
#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace quranrecite {

class StreamingModel {
public:
    StreamingModel(const std::string& convOnnx, const std::string& encoderOnnx);
    ~StreamingModel();
    StreamingModel(StreamingModel&&) noexcept;
    StreamingModel& operator=(StreamingModel&&) noexcept;

    // Start a fresh session (zero state, empty caches). Call before the first feed of a session.
    void reset();

    // A decoded phoneme + its absolute encoder-output frame (25 fps; time = frame * 0.04 s) and,
    // if wantAlts, the top-k posterior alts (tokenId, prob) at that frame — Phase-2 soft scoring.
    struct Emit {
        int id;
        int frame;
        std::vector<std::pair<int, float>> alts;   // empty unless feed(..., wantAlts=true)
    };

    // Feed new log-mel frames (row-major [T][80]); returns the phonemes decoded from the
    // newly-available audio (CTC greedy, blank/repeat collapsed across chunk boundaries).
    std::vector<Emit> feed(const float* logmel, int numFrames, bool wantAlts = false);

    int vocab() const;   // output token dimension (for the CTC head)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace quranrecite
