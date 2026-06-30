#include "quranrecite/detector.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

#include "decoder.h"
#include "frontend.h"
#include "inference.h"
#include "matcher.h"
#include "segmenter.h"

namespace quranrecite {

namespace {
constexpr int kSr = 16000;

// Linear resample to 16 kHz (managed capture is already 16k; app-fed PCM may differ).
std::vector<float> resampleTo16k(const float* x, std::size_t n, int inSr) {
    if (inSr == kSr || n == 0) return std::vector<float>(x, x + n);
    std::size_t outN = (std::size_t)((double)n * kSr / inSr);
    std::vector<float> y(outN);
    double step = (double)inSr / kSr;
    for (std::size_t i = 0; i < outN; ++i) {
        double pos = i * step;
        std::size_t i0 = (std::size_t)pos;
        float a = x[i0], b = (i0 + 1 < n) ? x[i0 + 1] : x[i0];
        y[i] = a + (b - a) * (float)(pos - i0);
    }
    return y;
}
}  // namespace

struct Detector::Impl {
    Config cfg;
    EventCallback cb;

    FrontEnd frontend;
    AcousticModel model;
    Lexicon lex;
    SequentialContext ctx;
    SlidingSegmenter seg;

    std::mutex mtx;
    std::vector<float> rolling;   // recent 16 kHz audio (>= 2 windows)
    long totalSamples = 0;
    long lastProc = 0;

    explicit Impl(const Config& c)
        : cfg(c),
          frontend(c.melFilterbankPath, c.hannWindowPath, c.normRms),
          model(c.modelPath),
          lex(c.lexiconPath, c.tokensPath),
          ctx(lex, c),
          seg(lex, ctx, c) {}

    int windowSamples() const { return (int)(cfg.windowSec * kSr); }

    // One sliding step: decode the trailing window, feed the segmenter, emit on event.
    void step(double timeSec) {
        const int W = windowSamples();
        if ((int)rolling.size() < kSr / 2) return;                 // < 0.5 s -> wait
        const std::size_t wlen = std::min<std::size_t>(W, rolling.size());
        const float* win = rolling.data() + (rolling.size() - wlen);

        double ss = 0.0;                                            // energy gate (skip silence)
        for (std::size_t i = 0; i < wlen; ++i) ss += (double)win[i] * win[i];
        if (std::sqrt(ss / wlen) < 0.005) return;

        int T;
        auto lm = frontend.logMel(win, wlen, T);
        int Tout, V;
        auto lp = model.run(lm, T, Tout, V);
        auto ids = ctcGreedy(lp, Tout, V);                         // ids share tokens.txt space
        if (auto ev = seg.process(ids, timeSec); ev && cb) cb(*ev);
    }
};

Detector::Detector(const Config& config) : impl_(std::make_unique<Impl>(config)) {}
Detector::~Detector() = default;
Detector::Detector(Detector&&) noexcept = default;
Detector& Detector::operator=(Detector&&) noexcept = default;

void Detector::setEventCallback(EventCallback cb) { impl_->cb = std::move(cb); }

void Detector::feed(const float* pcm, std::size_t n, int sampleRate) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    auto r = resampleTo16k(pcm, n, sampleRate);
    impl_->rolling.insert(impl_->rolling.end(), r.begin(), r.end());
    impl_->totalSamples += (long)r.size();

    const std::size_t cap = (std::size_t)(2 * impl_->windowSamples());  // keep ~2 windows
    if (impl_->rolling.size() > cap)
        impl_->rolling.erase(impl_->rolling.begin(), impl_->rolling.end() - cap);

    const long hop = (long)(impl_->cfg.hopSec * kSr);
    if (impl_->totalSamples - impl_->lastProc >= hop) {
        impl_->lastProc = impl_->totalSamples;
        impl_->step((double)impl_->totalSamples / kSr);
    }
}

void Detector::feedPcm16(const short* pcm, std::size_t n, int sampleRate) {
    std::vector<float> f(n);
    for (std::size_t i = 0; i < n; ++i) f[i] = pcm[i] / 32768.0f;
    feed(f.data(), n, sampleRate);
}

void Detector::reset() {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->rolling.clear();
    impl_->totalSamples = impl_->lastProc = 0;
    impl_->seg.reset();
}

const char* Detector::version() { return "0.1.0"; }

}  // namespace quranrecite
