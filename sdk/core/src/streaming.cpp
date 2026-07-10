#include "streaming.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "decoder.h"   // topKAlts — identical posterior rounding as the windowed path

namespace quranrecite {

namespace {
constexpr int kMel = 80;       // log-mel dim (conv input feature dim)
constexpr int kRF = 7;         // Conv2dSubsampling receptive field (input frames)
constexpr int kStride = 4;     // Conv2dSubsampling time stride

std::wstring wstr(const std::string& s) { return std::wstring(s.begin(), s.end()); }

// One encoder-state tensor kept alive between steps (recreated as an Ort::Value each Run).
struct State {
    std::vector<int64_t> shape;
    ONNXTensorElementDataType type;
    std::vector<float> f;       // used if FLOAT
    std::vector<int32_t> i;     // used if INT32
    int64_t count() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};
}  // namespace

struct StreamingModel::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "quranrecite-stream"};
    Ort::SessionOptions opts;
    Ort::Session conv{nullptr};
    Ort::Session enc{nullptr};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    int S = 0, R = 0, D = 0, nState = 0, vocab = 0;
    std::vector<std::string> encInNames, encOutNames;   // owned name strings
    std::vector<State> initState;                        // zero init (== states=None)

    // conv boundary cache (raw log-mel frames) + subsampled buffer for the encoder
    std::vector<float> cache;     // [cacheT][80]
    int cacheT = 0;
    int64_t cacheStart = 0;       // input frames dropped from the front of the conv stream
    int64_t emitted = 0;          // subsampled frames the conv has produced so far
    std::vector<float> sub;       // [subT][D] subsampled buffer
    int subT = 0;
    int seg = 0;                  // subsampled frames consumed by the encoder
    std::vector<State> state;     // 48 live encoder states
    int prev = -1;                // last emitted id (CTC collapse across chunks)
    int64_t outBase = 0;          // global encoder-output frames committed so far (25 fps)

    explicit Impl(const std::string& convOnnx, const std::string& encOnnx) {
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
#ifdef _WIN32
        conv = Ort::Session(env, wstr(convOnnx).c_str(), opts);
        enc = Ort::Session(env, wstr(encOnnx).c_str(), opts);
#else
        conv = Ort::Session(env, convOnnx.c_str(), opts);
        enc = Ort::Session(env, encOnnx.c_str(), opts);
#endif
        // encoder I/O: input 0 = "chunk" [1, S+R, D]; inputs 1.. = state s0..sN; output 0 =
        // "log_probs" [1, S, V]; outputs 1.. = new state ns0..nsN.
        Ort::AllocatorWithDefaultOptions al;
        const int nIn = (int)enc.GetInputCount();
        for (int k = 0; k < nIn; ++k)
            encInNames.push_back(enc.GetInputNameAllocated(k, al).get());
        for (int k = 0; k < (int)enc.GetOutputCount(); ++k)
            encOutNames.push_back(enc.GetOutputNameAllocated(k, al).get());
        auto chunkShape = enc.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();  // [1,S+R,D]
        D = (int)chunkShape[2];
        const int sr = (int)chunkShape[1];
        R = 1;                                            // right_context_length (Emformer default)
        S = sr - R;
        auto lpShape = enc.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();     // [1,S,V]
        vocab = (int)lpShape.back();
        nState = nIn - 1;                                 // 48 = 4 tensors x 12 layers

        // build the ZERO init state from the encoder's declared state input shapes/dtypes.
        for (int k = 1; k <= nState; ++k) {
            auto ti = enc.GetInputTypeInfo(k).GetTensorTypeAndShapeInfo();
            State st;
            st.shape = ti.GetShape();
            st.type = ti.GetElementType();
            if (st.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) st.i.assign(st.count(), 0);
            else st.f.assign(st.count(), 0.0f);
            initState.push_back(std::move(st));
        }
        reset();
    }

    void reset() {
        cache.clear(); cacheT = 0; cacheStart = 0; emitted = 0;
        sub.clear(); subT = 0; seg = 0; prev = -1; outBase = 0;
        state = initState;
    }

    // Streaming Conv2dSubsampling: cache raw frames, run conv over [cache ++ new], keep only the
    // NEW subsampled frames, advance the cache by the consumed input frames.
    std::vector<float> convStream(const float* feats, int T, int& outN) {
        std::vector<float> buf;
        buf.reserve((cacheT + T) * kMel);
        buf.insert(buf.end(), cache.begin(), cache.end());
        buf.insert(buf.end(), feats, feats + (std::size_t)T * kMel);
        const int bufT = cacheT + T;
        if (bufT < kRF) {                                 // not enough frames yet
            cache = std::move(buf); cacheT = bufT; outN = 0; return {};
        }
        std::array<int64_t, 3> shp{1, bufT, kMel};
        auto in = Ort::Value::CreateTensor<float>(mem, buf.data(), buf.size(), shp.data(), 3);
        const char* inN[] = {"feats"};
        const char* outNm[] = {"sub"};
        auto outv = conv.Run(Ort::RunOptions{nullptr}, inN, &in, 1, outNm, 1);
        Ort::Value& out = outv[0];
        auto oShape = out.GetTensorTypeAndShapeInfo().GetShape();   // [1, O, D]
        const int O = (int)oShape[1];
        const float* od = out.GetTensorData<float>();
        const int first = (int)(cacheStart / kStride);
        const int startIdx = (int)emitted - first;        // first new subsampled frame
        outN = O - startIdx;
        std::vector<float> newo((std::size_t)outN * D);
        std::copy_n(od + (std::size_t)startIdx * D, newo.size(), newo.data());
        emitted += outN;
        const int keep = (int)std::max<int64_t>(0, (int64_t)kStride * emitted - cacheStart);
        // cache = buf[keep:]
        const int newCacheT = bufT - keep;
        cache.assign(buf.begin() + (std::size_t)keep * kMel, buf.end());
        cacheT = newCacheT;
        cacheStart += keep;
        return newo;
    }

    std::vector<StreamingModel::Emit> feed(const float* feats, int T, bool wantAlts) {
        int nNew = 0;
        auto newo = convStream(feats, T, nNew);
        if (nNew) {
            sub.insert(sub.end(), newo.begin(), newo.end());
            subT += nNew;
        }
        std::vector<StreamingModel::Emit> emittedIds;
        while (seg + S + R <= subT) {
            // chunk = sub[seg : seg+S+R]
            std::array<int64_t, 3> cshp{1, S + R, D};
            std::vector<Ort::Value> ins;
            ins.reserve(1 + nState);
            ins.push_back(Ort::Value::CreateTensor<float>(
                mem, sub.data() + (std::size_t)seg * D, (std::size_t)(S + R) * D, cshp.data(), 3));
            for (auto& st : state) {
                if (st.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
                    ins.push_back(Ort::Value::CreateTensor<int32_t>(
                        mem, st.i.data(), st.i.size(), st.shape.data(), st.shape.size()));
                else
                    ins.push_back(Ort::Value::CreateTensor<float>(
                        mem, st.f.data(), st.f.size(), st.shape.data(), st.shape.size()));
            }
            std::vector<const char*> inN, outN;
            for (auto& s : encInNames) inN.push_back(s.c_str());
            for (auto& s : encOutNames) outN.push_back(s.c_str());
            auto res = enc.Run(Ort::RunOptions{nullptr}, inN.data(), ins.data(), ins.size(),
                               outN.data(), outN.size());
            // log_probs [1, S, V] -> collapse
            const float* lp = res[0].GetTensorData<float>();
            for (int t = 0; t < S; ++t) {
                const float* row = lp + (std::size_t)t * vocab;
                int best = 0;
                for (int v = 1; v < vocab; ++v) if (row[v] > row[best]) best = v;
                if (best != prev && best != 0) {
                    StreamingModel::Emit e{best, (int)(outBase + t), {}};
                    if (wantAlts) e.alts = topKAlts(row, vocab);
                    emittedIds.push_back(std::move(e));
                }
                prev = best;
            }
            outBase += S;
            // copy new states back into the live buffers (res[1..nState])
            for (int k = 0; k < nState; ++k) {
                auto& st = state[k];
                if (st.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
                    std::copy_n(res[1 + k].GetTensorData<int32_t>(), st.i.size(), st.i.data());
                else
                    std::copy_n(res[1 + k].GetTensorData<float>(), st.f.size(), st.f.data());
            }
            seg += S;
        }
        // drop consumed subsampled frames (bounded memory) — state carries the context.
        if (seg > 0) {
            sub.erase(sub.begin(), sub.begin() + (std::size_t)seg * D);
            subT -= seg;
            seg = 0;
        }
        return emittedIds;
    }
};

StreamingModel::StreamingModel(const std::string& convOnnx, const std::string& encoderOnnx)
    : impl_(std::make_unique<Impl>(convOnnx, encoderOnnx)) {}
StreamingModel::~StreamingModel() = default;
StreamingModel::StreamingModel(StreamingModel&&) noexcept = default;
StreamingModel& StreamingModel::operator=(StreamingModel&&) noexcept = default;

void StreamingModel::reset() { impl_->reset(); }
std::vector<StreamingModel::Emit> StreamingModel::feed(const float* logmel, int numFrames,
                                                       bool wantAlts) {
    return impl_->feed(logmel, numFrames, wantAlts);
}
int StreamingModel::vocab() const { return impl_->vocab; }

}  // namespace quranrecite
