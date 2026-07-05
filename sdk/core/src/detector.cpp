#include "quranrecite/detector.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <vector>

#include "autodet.h"
#include "decoder.h"
#include "frontend.h"
#include "highlight.h"
#include "inference.h"
#include "matcher.h"
#include "segmenter.h"
#include "vad.h"

#ifdef __ANDROID__
#include <android/log.h>
#define QR_LOG(...) __android_log_print(ANDROID_LOG_INFO, "QuranReciteCore", __VA_ARGS__)
#else
#define QR_LOG(...) ((void)0)
#endif

namespace quranrecite {

namespace {
constexpr int kSr = 16000;

std::string ayahKey(const AyahId& a) { return std::to_string(a.surah) + ":" + std::to_string(a.ayah); }

AyahId parseAyah(const std::string& sa) {
    auto c = sa.find(':');
    return {std::stoi(sa.substr(0, c)), std::stoi(sa.substr(c + 1))};
}

// Convert the string-keyed reference state to the public AyahId snapshot.
HighlightSnapshot toPublic(const HighlightState& s) {
    HighlightSnapshot out;
    for (auto& k : s.confirmed) out.confirmed.push_back(parseAyah(k));
    if (s.active) { out.hasActive = true; out.active = parseAyah(*s.active); }
    if (s.pending) {
        out.hasPending = true;
        auto& p = out.pending;
        if (s.pending->ayah) { p.hasAyah = true; p.ayah = parseAyah(*s.pending->ayah); }
        for (auto& o : s.pending->options) p.options.push_back(parseAyah(o));
        p.reason = s.pending->reason == "needs_choice" ? PendingReason::NeedsChoice
                                                        : PendingReason::AwaitSuccessor;
    }
    return out;
}

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
    HighlightCallback hlCb;

    FrontEnd frontend;
    AcousticModel model;
    Lexicon lex;
    SequentialContext ctx;
    SlidingSegmenter seg;
    AutoDetector autod;
    HighlightController hl;
    std::unique_ptr<SileroVAD> vad;   // optional (Auto mode): resets on paused-recitation boundaries

    std::mutex mtx;
    std::vector<float> rolling;   // recent 16 kHz audio (up to ~30 s; sliding uses the last window)
    std::vector<float> vadBuf;    // 16 kHz audio pending VAD, fed in 512-sample chunks
    long totalSamples = 0;
    long lastProc = 0;
    long streamStartAbs = 0;      // absolute sample where the stream matcher's buffer begins

    HighlightSnapshot lastSnap;   // last emitted highlight; re-emitted per hop to reveal upNext
    std::string curActive;        // committed ayah whose completion gates the darker "up next"
    bool upNextShown = false;     // upNext already revealed for the current active ayah
    std::atomic<bool> debug{false};   // runtime debug logging (see setDebug)

    explicit Impl(const Config& c)
        : cfg(c),
          frontend(c.melFilterbankPath, c.hannWindowPath, c.normRms),
          model(c.modelPath),
          lex(c.lexiconPath, c.tokensPath),
          ctx(lex, c),
          seg(lex, ctx, c),
          autod(lex, c),
          hl(c.ambiguousPath) {
        if (!c.vadPath.empty())
            vad = std::make_unique<SileroVAD>(c.vadPath, kSr, c.vadThreshold, c.vadMinSilenceSec);
    }

    // A Silero speech-END: paused ayah boundary. Drop the buffered ayah + trailing silence and
    // re-anchor the matcher so the next ayah decodes fresh (mirrors demo/live_detect.py auto loop).
    void boundaryReset() {
        if (debug) QR_LOG("VAD speech-END -> boundaryReset at %.1fs", (double)totalSamples / kSr);
        rolling.clear();
        vadBuf.clear();
        streamStartAbs = totalSamples;
        lastProc = totalSamples;   // buffer is empty -> skip an immediate no-op step
        autod.reset();
    }

    int windowSamples() const { return (int)(cfg.windowSec * kSr); }

    std::vector<int> decodeWindow(const float* win, std::size_t wlen) {
        int T;
        auto lm = frontend.logMel(win, wlen, T);
        int Tout, V;
        auto lp = model.run(lm, T, Tout, V);
        return ctcGreedy(lp, Tout, V);                             // ids share tokens.txt space
    }

    void emit(EventType type, const std::string& key, double timeSec) {
        if (debug) QR_LOG("COMMIT %s (%s) at %.1fs", key.c_str(),
               type == EventType::Detect ? "detect" : type == EventType::Advance ? "advance" : "jump", timeSec);
        const auto c = key.find(':');
        AyahEvent ev;
        ev.type = type;
        ev.ayah = {std::stoi(key.substr(0, c)), std::stoi(key.substr(c + 1))};
        ev.timeSec = timeSec;
        if (cb) cb(ev);
        if (hlCb) {
            lastSnap = toPublic(hl.detect(key));   // new active ayah (lighter); upNext hidden until
            curActive = key;                       //   this ayah nears completion (see maybeUpNext)
            upNextShown = false;
            hlCb(lastSnap);
        }
    }

    // Once the active ayah is near-complete, reveal its same-surah successor as the darker "up
    // next" highlight (the ayah being verified now). Re-emits the cached snapshot with upNext set.
    void maybeUpNext(const std::vector<std::pair<std::string, float>>& progress) {
        if (!hlCb || curActive.empty() || upNextShown) return;
        const auto c = curActive.find(':');
        const int surah = std::stoi(curActive.substr(0, c));
        const int ayah = std::stoi(curActive.substr(c + 1));
        const std::string nxt = std::to_string(surah) + ":" + std::to_string(ayah + 1);
        if (lex.orderIndex(nxt) < 0) return;               // last ayah of the surah -> no upNext
        // Reveal the darker "up next" once the active ayah is near-complete via its own progress,
        // OR once the successor has become the leading candidate — i.e. the reciter has moved on,
        // so the active ayah is effectively done. The latter is what catches short ayat, which
        // drop out of the top-k before their own progress is ever seen crossing the threshold.
        float activeP = -1.0f;
        for (const auto& [k, pr] : progress) if (k == curActive) { activeP = pr; break; }
        const bool nextLeads = !progress.empty() && progress.front().first == nxt;
        if (activeP < cfg.doneProgress && !nextLeads) return;
        lastSnap.hasUpNext = true;
        lastSnap.upNext = {surah, ayah + 1};
        upNextShown = true;
        hlCb(lastSnap);
    }

    // One sliding-only step (Mode::Sliding).
    void step(double timeSec) {
        const int W = windowSamples();
        if ((int)rolling.size() < kSr / 2) return;                 // < 0.5 s -> wait
        const std::size_t wlen = std::min<std::size_t>(W, rolling.size());
        const float* win = rolling.data() + (rolling.size() - wlen);
        if (rms(win, wlen) < 0.005) return;
        if (auto ev = seg.process(decodeWindow(win, wlen), timeSec)) {
            if (cb) cb(*ev);
            if (hlCb) hlCb(toPublic(hl.detect(ayahKey(ev->ayah))));
        }
    }

    // One auto step (Mode::Auto): decode the fixed sliding window AND the stream anchored
    // buffer, merge. Refocus advances the stream buffer start. Port of the auto live loop.
    void stepAuto(double timeSec) {
        const int W = windowSamples();
        if ((int)rolling.size() < kSr / 2) return;
        const long rollStart = totalSamples - (long)rolling.size();
        const std::size_t wlen = std::min<std::size_t>(W, rolling.size());
        const float* swin = rolling.data() + (rolling.size() - wlen);
        const double r = rms(swin, wlen);
        if (r < 0.005) {
            if (debug) QR_LOG("hop t=%.1fs rms=%.4f -> silent, skip", timeSec, r);
            return;
        }

        const long sIdx = std::max<long>(0, streamStartAbs - rollStart);
        const float* stwin = rolling.data() + sIdx;
        const std::size_t stlen = rolling.size() - (std::size_t)sIdx;

        auto slidePh = decodeWindow(swin, wlen);
        auto streamPh = decodeWindow(stwin, stlen);
        if (debug) QR_LOG("hop t=%.1fs rms=%.4f slidePh=%zu streamPh=%zu (streamBuf=%.1fs)",
               timeSec, r, slidePh.size(), streamPh.size(), (double)stlen / kSr);
        auto st = autod.feed(slidePh, timeSec, streamPh);
        if (st.commit) emit(st.commit->event, st.commit->ayah, timeSec);
        maybeUpNext(st.progress);          // reveal the darker "up next" once active is near-complete
        if (st.refocusSec)
            streamStartAbs = std::max<long>(streamStartAbs, totalSamples - (long)(*st.refocusSec * kSr));
    }

    static double rms(const float* x, std::size_t n) {
        double ss = 0.0;
        for (std::size_t i = 0; i < n; ++i) ss += (double)x[i] * x[i];
        return std::sqrt(ss / std::max<std::size_t>(1, n));
    }
};

Detector::Detector(const Config& config) : impl_(std::make_unique<Impl>(config)) {}
Detector::~Detector() = default;
Detector::Detector(Detector&&) noexcept = default;
Detector& Detector::operator=(Detector&&) noexcept = default;

void Detector::setEventCallback(EventCallback cb) { impl_->cb = std::move(cb); }
void Detector::setHighlightCallback(HighlightCallback cb) { impl_->hlCb = std::move(cb); }

void Detector::feed(const float* pcm, std::size_t n, int sampleRate) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    auto r = resampleTo16k(pcm, n, sampleRate);
    impl_->rolling.insert(impl_->rolling.end(), r.begin(), r.end());
    impl_->totalSamples += (long)r.size();

    // Auto keeps up to ~30 s (the stream matcher's max buffer); sliding needs only ~2 windows.
    const std::size_t cap = impl_->cfg.mode == Mode::Auto
        ? (std::size_t)(30 * kSr) : (std::size_t)(2 * impl_->windowSamples());
    if (impl_->rolling.size() > cap)
        impl_->rolling.erase(impl_->rolling.begin(), impl_->rolling.end() - cap);

    // Silero VAD (Auto): feed the same 16 kHz audio in fixed 512-sample chunks; a speech-END
    // event marks an ayah boundary -> drop the buffer + re-anchor so the next ayah decodes fresh.
    if (impl_->vad) {
        impl_->vadBuf.insert(impl_->vadBuf.end(), r.begin(), r.end());
        const int VC = SileroVAD::chunkSize();
        bool boundary = false;
        std::size_t off = 0;
        while (impl_->vadBuf.size() - off >= (std::size_t)VC) {
            if (impl_->vad->feed(impl_->vadBuf.data() + off, VC)) boundary = true;
            off += VC;
        }
        impl_->vadBuf.erase(impl_->vadBuf.begin(), impl_->vadBuf.begin() + off);
        if (boundary) impl_->boundaryReset();
    }

    const long hop = (long)(impl_->cfg.hopSec * kSr);
    if (impl_->totalSamples - impl_->lastProc >= hop) {
        impl_->lastProc = impl_->totalSamples;
        if (impl_->cfg.mode == Mode::Auto) impl_->stepAuto((double)impl_->totalSamples / kSr);
        else impl_->step((double)impl_->totalSamples / kSr);
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
    impl_->vadBuf.clear();
    impl_->totalSamples = impl_->lastProc = impl_->streamStartAbs = 0;
    impl_->seg.reset();
    impl_->autod.reset();
    impl_->hl.reset();
    impl_->lastSnap = HighlightSnapshot{};
    impl_->curActive.clear();
    impl_->upNextShown = false;
    if (impl_->vad) impl_->vad->reset();
}

void Detector::setDebug(bool enabled) { impl_->debug.store(enabled); }

const char* Detector::version() { return "0.1.0"; }

}  // namespace quranrecite
