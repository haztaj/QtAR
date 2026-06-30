#include "inference.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include <onnxruntime_cxx_api.h>

namespace quranrecite {

struct AcousticModel::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "quranrecite"};
    Ort::SessionOptions opts;
    Ort::Session session{nullptr};
    int fixedT = 0;   // model's fixed input window (e.g. 3000)
    int vocab = 0;
    std::string inFeat = "features", inLen = "lengths", outLP = "log_probs", outOL = "out_lengths";

    explicit Impl(const std::string& path) {
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        // TODO(platform): append NNAPI (Android) / CoreML (iOS) execution providers here;
        // they fall back to CPU. Desktop/conformance uses CPU.
#ifdef _WIN32
        std::wstring wpath(path.begin(), path.end());
        session = Ort::Session(env, wpath.c_str(), opts);
#else
        session = Ort::Session(env, path.c_str(), opts);
#endif
        auto inShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();   // [B,T,80]
        fixedT = static_cast<int>(inShape[1]);
        auto outShape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();  // [B,T',V]
        vocab = static_cast<int>(outShape.back());
    }
};

AcousticModel::AcousticModel(const std::string& path) : impl_(new Impl(path)) {}
AcousticModel::~AcousticModel() { delete impl_; }

std::vector<float> AcousticModel::run(const std::vector<float>& logmel, int T, int& outT, int& outVocab) {
    const int FT = impl_->fixedT;
    const int valid = std::min(T, FT);

    // pad/crop log-mel [T,80] -> fixed [FT,80]
    std::vector<float> feats(static_cast<std::size_t>(FT) * 80, 0.0f);
    std::copy_n(logmel.data(), static_cast<std::size_t>(valid) * 80, feats.data());
    std::int64_t lengths[1] = {valid};

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<std::int64_t, 3> featShape{1, FT, 80};
    std::array<std::int64_t, 1> lenShape{1};
    std::array<Ort::Value, 2> inputs{
        Ort::Value::CreateTensor<float>(mem, feats.data(), feats.size(), featShape.data(), 3),
        Ort::Value::CreateTensor<std::int64_t>(mem, lengths, 1, lenShape.data(), 1)};

    const char* inNames[] = {impl_->inFeat.c_str(), impl_->inLen.c_str()};
    const char* outNames[] = {impl_->outLP.c_str(), impl_->outOL.c_str()};
    auto outs = impl_->session.Run(Ort::RunOptions{nullptr}, inNames, inputs.data(), 2, outNames, 2);

    auto lpShape = outs[0].GetTensorTypeAndShapeInfo().GetShape();   // [1,Tout,V]
    const int Tout = static_cast<int>(lpShape[1]);
    const int V = static_cast<int>(lpShape[2]);
    const int validTout = std::min(static_cast<int>(outs[1].GetTensorData<std::int64_t>()[0]), Tout);

    const float* lp = outs[0].GetTensorData<float>();
    std::vector<float> out(static_cast<std::size_t>(validTout) * V);
    std::copy_n(lp, out.size(), out.data());
    outT = validTout;
    outVocab = V;
    return out;
}

}  // namespace quranrecite
