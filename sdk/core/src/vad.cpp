#include "vad.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace quranrecite {

struct SileroVAD::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "silero_vad"};
    Ort::SessionOptions opts;
    Ort::Session session{nullptr};
    static constexpr int kCtx = 64;  // silero v5 @16 kHz prepends the last 64 samples per chunk
    std::int64_t sr;
    float threshold;
    long minSilence;                 // samples of silence to declare a speech end
    std::vector<float> state;        // [2,1,128] = 256 floats, carried across calls
    std::vector<float> context;      // last kCtx samples of the previous chunk, prepended
    bool triggered = false;
    long tempEnd = 0, current = 0;

    Impl(const std::string& path, int sampleRate, float thr, float minSilenceSec)
        : sr(sampleRate), threshold(thr),
          minSilence(static_cast<long>(minSilenceSec * sampleRate)),
          state(2 * 128, 0.0f), context(kCtx, 0.0f) {
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
#ifdef _WIN32
        std::wstring wpath(path.begin(), path.end());
        session = Ort::Session(env, wpath.c_str(), opts);
#else
        session = Ort::Session(env, path.c_str(), opts);
#endif
    }

    void reset() {
        std::fill(state.begin(), state.end(), 0.0f);
        std::fill(context.begin(), context.end(), 0.0f);
        triggered = false;
        tempEnd = 0;
        current = 0;
    }

    bool feed(const float* chunk, int n) {
        current += n;
        // Prepend the 64-sample context: model input is [context(64) | chunk(n)].
        std::vector<float> in(kCtx + n);
        std::copy(context.begin(), context.end(), in.begin());
        std::copy(chunk, chunk + n, in.begin() + kCtx);
        std::copy(chunk + n - kCtx, chunk + n, context.begin());   // context <- last 64 of chunk

        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<std::int64_t, 2> inShape{1, kCtx + n};
        std::array<std::int64_t, 3> stShape{2, 1, 128};
        std::array<Ort::Value, 3> inputs{
            Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), inShape.data(), 2),
            Ort::Value::CreateTensor<float>(mem, state.data(), state.size(), stShape.data(), 3),
            Ort::Value::CreateTensor<std::int64_t>(mem, &sr, 1, nullptr, 0)};   // rank-0 scalar
        const char* inNames[] = {"input", "state", "sr"};
        const char* outNames[] = {"output", "stateN"};
        auto outs = session.Run(Ort::RunOptions{nullptr}, inNames, inputs.data(), 3, outNames, 2);
        const float prob = outs[0].GetTensorData<float>()[0];
        const float* ns = outs[1].GetTensorData<float>();
        std::copy(ns, ns + state.size(), state.data());

        // VADIterator 'end' logic (silero_vad).
        if (prob >= threshold && tempEnd) tempEnd = 0;
        if (prob >= threshold && !triggered) triggered = true;      // speech start (ignored here)
        if (prob < threshold - 0.15f && triggered) {
            if (!tempEnd) tempEnd = current;
            if (current - tempEnd >= minSilence) {
                tempEnd = 0;
                triggered = false;
                return true;                                        // speech END
            }
        }
        return false;
    }
};

SileroVAD::SileroVAD(const std::string& p, int sr, float t, float ms) : impl_(new Impl(p, sr, t, ms)) {}
SileroVAD::~SileroVAD() { delete impl_; }
void SileroVAD::reset() { impl_->reset(); }
bool SileroVAD::feed(const float* c, int n) { return impl_->feed(c, n); }

}  // namespace quranrecite
