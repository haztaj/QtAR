#include "inference.h"
// TODO(port): #include <onnxruntime_cxx_api.h>

namespace quranrecite {

struct AcousticModel::Impl {
    // Ort::Env env; Ort::Session session; input/output names, fixed window length.
    // Android: append the NNAPI execution provider; iOS: CoreML; CPU fallback.
};

AcousticModel::AcousticModel(const std::string& /*onnxPath*/) : impl_(new Impl()) {
    // TODO(port): create Ort::Env, SessionOptions (+NNAPI/CoreML), Ort::Session(onnxPath).
}
AcousticModel::~AcousticModel() { delete impl_; }

std::vector<float> AcousticModel::run(const std::vector<float>& /*logmel*/, int /*T*/,
                                      int& outT, int& outVocab) {
    // TODO(port): pad/crop logmel to the fixed window; build input tensor [1,T,80] +
    // lengths; session.Run; copy log-probs out. Vocab = 35 (34 phonemes + blank).
    outT = 0; outVocab = 0;
    return {};
}

}  // namespace quranrecite
