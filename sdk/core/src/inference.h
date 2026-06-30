// ONNX Runtime wrapper for the acoustic model. Not a port risk (same engine as Python;
// parity established in export/). Fixed sliding-window graph (e.g. 4 s).
#pragma once
#include <string>
#include <vector>

namespace quranrecite {

class AcousticModel {
public:
    explicit AcousticModel(const std::string& onnxPath);
    ~AcousticModel();

    // log-mel [T][80] (flattened) -> CTC log-probs [Tout][vocab] (flattened). `outT`,
    // `outVocab` receive the dims. Pads/crops T to the model's fixed window internally.
    std::vector<float> run(const std::vector<float>& logmel, int T, int& outT, int& outVocab);

private:
    struct Impl;
    Impl* impl_;   // Ort::Env / Session (+ NNAPI EP on Android, CoreML on iOS)
};

}  // namespace quranrecite
