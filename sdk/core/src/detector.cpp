#include "quranrecite/detector.h"

#include <mutex>
#include <vector>

#include "decoder.h"
#include "frontend.h"
#include "inference.h"
#include "matcher.h"
#include "segmenter.h"

namespace quranrecite {

struct Detector::Impl {
    Config cfg;
    EventCallback cb;

    FrontEnd frontend;
    AcousticModel model;
    Lexicon lex;
    SequentialContext ctx;
    SlidingSegmenter seg;

    std::mutex mtx;
    std::vector<float> rolling;     // 16 kHz ring of recent audio (>= 2 windows)
    long totalSamples = 0;
    long lastProc = 0;

    explicit Impl(const Config& c)
        : cfg(c),
          frontend(c.melFilterbankPath, c.hannWindowPath, c.normRms),
          model(c.modelPath),
          lex(c.lexiconPath, c.tokensPath),
          ctx(lex, c),
          seg(lex, ctx, c) {}

    // One sliding step: decode the trailing window, feed the segmenter, emit on event.
    void step(double timeSec) {
        // TODO(port): take the last windowSec of `rolling`; gate on a simple energy VAD;
        //   int T; auto lm = frontend.logMel(window.data(), window.size(), T);
        //   int Tout, V; auto lp = model.run(lm, T, Tout, V);
        //   auto phon = ctcGreedy(lp, Tout, V);
        //   if (auto ev = seg.process(phon, timeSec); ev && cb) cb(*ev);
        (void)timeSec;
    }
};

Detector::Detector(const Config& config) : impl_(std::make_unique<Impl>(config)) {}
Detector::~Detector() = default;
Detector::Detector(Detector&&) noexcept = default;
Detector& Detector::operator=(Detector&&) noexcept = default;

void Detector::setEventCallback(EventCallback cb) { impl_->cb = std::move(cb); }

void Detector::feed(const float* pcm, std::size_t n, int sampleRate) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    // TODO(port): resample pcm to 16k if sampleRate != 16000 (linear/polyphase),
    // append to impl_->rolling (cap to ~2*window), advance totalSamples, and whenever
    // (totalSamples - lastProc) >= hop*16000 -> lastProc = totalSamples; step(t).
    (void)pcm; (void)n; (void)sampleRate;
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

const char* Detector::version() { return "0.1.0-scaffold"; }

}  // namespace quranrecite
