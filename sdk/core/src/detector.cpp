#include "quranrecite/detector.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <vector>

#include "autodet.h"
#include "chain.h"
#include "decoder.h"
#include "frontend.h"
#include "highlight.h"
#include "inference.h"
#include "matcher.h"
#include "segmenter.h"
#include "streaming.h"
#include "vad.h"

#ifdef __ANDROID__
#include <android/log.h>
#define QR_LOG(...) __android_log_print(ANDROID_LOG_INFO, "QuranReciteCore", __VA_ARGS__)
#else
// Desktop (harness/debug): same per-hop engine log to stderr, still gated on setDebug(true)
// at every call site (test_detector enables it via QR_DEBUG=1).
#define QR_LOG(...) (std::fprintf(stderr, "[core] " __VA_ARGS__), std::fprintf(stderr, "\n"))
#endif

namespace quranrecite {

namespace {
constexpr int kSr = 16000;
constexpr int kHop = 160;    // log-mel hop (10 ms) — streaming feed uses hop-aligned bookkeeping
constexpr int kMelDim = 80;  // log-mel feature dim

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

    // Mode::Chain — unit-chain decoder (waqf segments; research winning design)
    std::unique_ptr<UnitIndex> units;
    std::unique_ptr<ChainVoter> chainVoter;
    std::unique_ptr<ChainAssembler> chainAsm;
    std::string chainParent;          // last confirmed parent ayah ("" before the first)

    // True streaming acoustics (optional, Mode::Chain): decode only the NEW audio each hop and
    // keep a persistent, bounded phoneme stream — instead of re-decoding the whole rolling window.
    std::unique_ptr<StreamingModel> stream;
    // v13 fresh-context suffix decode (optional, windowed Chain): right-sized second graph,
    // same weights as `model`. See Config::chainSuffixSec.
    std::unique_ptr<AcousticModel> suffixModel;
    std::vector<int> chainPh;         // persistent phoneme ids (absolute-time keyed)
    std::vector<double> chainTm;      // their absolute times (frame * 0.04 s)
    PhonAlts chainAlts;               // per-phoneme top-k posteriors (Phase-2 soft; if subMin<1)
    long streamFedFrames = 0;         // absolute log-mel frames fed to the stream (monotonic)
    double decodeSec = 0.0;           // cumulative acoustic-decode wall-clock (RTF instrumentation)
    long decodeHops = 0;              // hops that ran a decode

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
    double lastConfirmSec = -1e9;     // time of the last confirmed unit (chainResetMaxGap gate)
    double lastEmitSec = -1e9;        // time of the last voter EMISSION (pre-commit gate arm:
                                      // cold-start takes emit + defer but never confirm, so the
                                      // commit-only gate starved them of the focused window)

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
        if (c.mode == Mode::Chain) {
            units = std::make_unique<UnitIndex>(c.unitPhonemesPath, c.tokensPath);
            ChainParams p;
            p.windowSec = c.chainWindowSec;
            p.hopSec = c.chainHopSec;
            p.costThresh = c.chainCost;
            p.votesNext = c.chainVotesNext;
            p.votesJump = c.chainVotesJump;
            chainVoter = std::make_unique<ChainVoter>(*units, p);
            chainAsm = std::make_unique<ChainAssembler>(*units);
            if (!c.streamConvPath.empty() && !c.streamEncoderPath.empty())
                stream = std::make_unique<StreamingModel>(c.streamConvPath, c.streamEncoderPath);
            if (c.chainSuffixSec > 0.0f && !c.chainSuffixModelPath.empty())
                suffixModel = std::make_unique<AcousticModel>(c.chainSuffixModelPath);
        }
    }

    // A Silero speech-END: paused ayah boundary. Drop the buffered ayah + trailing silence and
    // re-anchor the matcher so the next ayah decodes fresh (mirrors demo/live_detect.py auto loop).
    // Drop the buffered ayah + re-anchor so the next ayah decodes in a FOCUSED window (Auto's
    // paused-boundary reset; Chain's chainVadReset focused-window de-crowding). Chain keeps the
    // voter/assembler chain context (expected/streak/emitted survive the pause). WINDOWED chain
    // only — see the VAD gate in feed(): streaming decodes incrementally, so clearing the buffer
    // can't recover crowded tail phonemes (measured no gain + slight harm) and would break its
    // absolute output-frame time axis; chainVadReset is a no-op in streaming mode.
    // Drop rolling audio before absolute stream second `fromSec` (windowed chain only). The
    // whole-buffer decode's time axis derives from totalSamples - rolling.size(), so the trim
    // keeps all bookkeeping consistent; chain context (voter/assembler) is untouched.
    void focusTrim(double fromSec) {
        const long bufStart = totalSamples - (long)rolling.size();
        long cut = (long)(fromSec * kSr) - bufStart;
        if (cut <= 0) return;
        if (cut > (long)rolling.size()) cut = (long)rolling.size();
        if (debug) QR_LOG("chain EMIT -> focusTrim to %.1fs (dropped %.1fs)",
                          fromSec, (double)cut / kSr);
        rolling.erase(rolling.begin(), rolling.begin() + cut);
    }

    void boundaryReset() {
        if (debug) QR_LOG("VAD speech-END -> boundaryReset at %.1fs", (double)totalSamples / kSr);
        rolling.clear();
        vadBuf.clear();
        streamStartAbs = totalSamples;
        lastProc = totalSamples;   // buffer empty -> skip an immediate no-op step
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

    // seg/segCount are the waqf-segment progress of the active ayah (Mode::Chain); 0/0 elsewhere.
    void emit(EventType type, const std::string& key, double timeSec, int seg = 0, int segCount = 0) {
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
            lastSnap.activeSegment = seg;          //   this ayah nears completion (see maybeUpNext)
            lastSnap.activeSegmentCount = segCount;
            curActive = key;
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

    // One chain step (Mode::Chain): obtain the phoneme stream over ~the largest window (windowed
    // re-decode, or the incremental StreamingModel stream), slice all scale windows ending "now"
    // by time, fire -> vote -> assemble; each newly-confirmed unit whose parent differs from the
    // last drives an ayah event + highlight. The chain survives pauses natively (time-gated).
    void stepChain(double timeSec) {
        if ((int)rolling.size() < kSr / 2) return;
        const double r = rms(rolling.data() + rolling.size() - std::min<std::size_t>(rolling.size(), kSr),
                             std::min<std::size_t>(rolling.size(), kSr));
        const bool silent = r < 0.005;
        const bool soft = cfg.chainSubMin < 1.0f;

        if (stream) {
            // Always extend the stream (keep the acoustic state continuous, even on silence);
            // gate only the matching on energy.
            auto t0 = std::chrono::steady_clock::now();
            streamFeed(soft);
            decodeSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            ++decodeHops;
            if (silent) {
                if (debug) QR_LOG("chain hop t=%.1fs rms=%.4f -> silent, skip match", timeSec, r);
                return;
            }
            if (debug) QR_LOG("chain hop t=%.1fs rms=%.4f ph=%zu (stream)", timeSec, r, chainPh.size());
            chainMatch(timeSec, chainPh, chainTm, chainAlts, soft);
            return;
        }

        if (silent) {
            if (debug) QR_LOG("chain hop t=%.1fs rms=%.4f -> silent, skip", timeSec, r);
            return;
        }
        const double bufStartSec = (double)(totalSamples - (long)rolling.size()) / kSr;
        auto t0 = std::chrono::steady_clock::now();
        int T;
        auto lm = frontend.logMel(rolling.data(), rolling.size(), T);
        int Tout, V;
        auto lp = model.run(lm, T, Tout, V);
        std::vector<int> ph;
        std::vector<double> tm;
        PhonAlts alts;                                   // Phase-2 posteriors (if subMin < 1)
        int prev = -1;
        for (int f = 0; f < Tout; ++f) {
            const float* row = lp.data() + (std::size_t)f * V;
            int best = 0;
            for (int v = 1; v < V; ++v)
                if (row[v] > row[best]) best = v;
            if (best != prev && best != 0) {
                ph.push_back(best);
                tm.push_back(bufStartSec + f * 0.04);
                if (soft) alts.push_back(topKAlts(row, V));
            }
            prev = best;
        }
        decodeSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++decodeHops;
        if (debug) QR_LOG("chain hop t=%.1fs rms=%.4f ph=%zu (buf=%.1fs)",
                          timeSec, r, ph.size(), (double)rolling.size() / kSr);
        chainMatch(timeSec, ph, tm, alts, soft);

        // v13 fresh-context suffix pass: decode the buffer's last chainSuffixSec seconds as a
        // STANDALONE input (fresh Emformer memory — sidesteps repeated-phrase suppression) and
        // match over it with a restricted two-window bank. Only when the buffer exceeds the
        // suffix (otherwise the main decode already has fresh context).
        // Skip while confidently tracking a LONG expected unit: a fresh 5 s window over a long
        // ayah's MIDDLE can only fire spurious short units (its true unit fails the length
        // gate), and those disrupt the chain (measured -2 on long Baqarah). Short-unit crowding
        // — the class this pass exists for — always has a short or no expectation.
        // "Long" = cannot fit the suffix window's length gate (~5 phonemes/s of speech, 1.3x
        // upper band) — such a unit can never fire from this pass, only be disrupted by it:
        // near-threshold junk fires during the long wait flood the assembler's 2-deep pending
        // buffer and evict the true pending unit (measured on Baqarah). Any emission sets the
        // expectation, so no streak requirement — a lone jump emission counts.
        const int sufExp = chainVoter->expectedUnit();
        const bool trackingLong = sufExp >= 0 &&
                                  units->len(sufExp) > (int)(6.5 * cfg.chainSuffixSec);
        if (suffixModel && !trackingLong &&
            rolling.size() > (std::size_t)(cfg.chainSuffixSec * kSr)) {
            const std::size_t sn = (std::size_t)(cfg.chainSuffixSec * kSr);
            const double sufStartSec = (double)(totalSamples - (long)sn) / kSr;
            auto ts0 = std::chrono::steady_clock::now();
            int Ts;
            auto slm = frontend.logMel(rolling.data() + (rolling.size() - sn), sn, Ts);
            int TsOut, Vs;
            auto slp = suffixModel->run(slm, Ts, TsOut, Vs);
            std::vector<int> sph;
            std::vector<double> stm;
            PhonAlts salts;
            int sprev = -1;
            for (int f = 0; f < TsOut; ++f) {
                const float* row = slp.data() + (std::size_t)f * Vs;
                int best = 0;
                for (int v = 1; v < Vs; ++v)
                    if (row[v] > row[best]) best = v;
                if (best != sprev && best != 0) {
                    sph.push_back(best);
                    stm.push_back(sufStartSec + f * 0.04);
                    if (soft) salts.push_back(topKAlts(row, Vs));
                }
                sprev = best;
            }
            decodeSec += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - ts0).count();
            if (debug) QR_LOG("chain suffix t=%.1fs ph=%zu (%.1fs fresh)",
                              timeSec, sph.size(), cfg.chainSuffixSec);
            suffixMatch(timeSec, sph, stm, salts, soft);
        }
    }

    // Restricted matched-filter pass over the fresh-context suffix decode: two windows only —
    // the full suffix and its last 2 s (the small-unit scale) — so the extra pass adds bounded
    // vote opportunities instead of duplicating the whole scale bank on identical content.
    void suffixMatch(double timeSec, const std::vector<int>& ph, const std::vector<double>& tm,
                     const PhonAlts& alts, bool soft) {
        const double spans[2] = {(double)cfg.chainSuffixSec, 0.2 * cfg.chainWindowSec};
        for (double span : spans) {
            const double w0 = timeSec - span;
            auto lo = std::lower_bound(tm.begin(), tm.end(), w0);
            const std::size_t off = lo - tm.begin();
            std::vector<int> win(ph.begin() + off, ph.end());
            if ((int)win.size() < 4) continue;
            PhonAlts winAlts;
            if (soft && alts.size() == ph.size()) winAlts.assign(alts.begin() + off, alts.end());
            auto [u, cost] = windowBest(win, *units, cfg.chainCost, winAlts, cfg.chainSubMin);
            if (u < 0 || cost > cfg.chainCost) continue;
            if (cfg.chainStartAtAyahSec > 0.0f && chainAsm->confirmed().empty()
                && units->segIdxOf(u) >= 2) {   // cold start: decaying penalty on mid-ayah matches
                const double frac = (double)timeSec / cfg.chainStartAtAyahSec;   // 0..1 over the window
                const double m = frac >= 1.0 ? 1.0
                                 : cfg.chainStartAyahMult + (1.0 - cfg.chainStartAyahMult) * frac;
                if (cost > cfg.chainCost * m) continue;   // mid-ayah needs a tighter cost early
            }
            if (auto em = chainVoter->onFire(timeSec, u, cost)) {
                if (debug) QR_LOG("chain EMIT %s cost=%.2f at %.1fs (suffix)",
                                  units->key(em->unit).c_str(), cost, timeSec);
                if (cost <= 0.5f * cfg.chainCost) lastEmitSec = timeSec;
                auto confirms = chainAsm->push(em->unit);
                for (int cu : confirms) confirmUnit(cu, timeSec);
                maybeProvisional(em->unit, confirms.empty(), timeSec, cost);
            }
        }
    }

    // Streaming acoustics: extend the persistent phoneme stream with the newly-SETTLED log-mel
    // frames. The front-end is center=True/reflect-pad, so a frame's value is only final once
    // its full n_fft window is real audio -> feed up to T-guard (hold back the boundary tail;
    // it settles next hop). The rolling-buffer start is kept hop-aligned (see feed()), so buffer
    // frame f == absolute frame bufStartFrame+f and interior frames match the offline continuous
    // log-mel EXACTLY. streamFedFrames (absolute, monotonic) guarantees a gapless feed.
    void streamFeed(bool soft) {
        constexpr int guard = 2;      // end frames held back per hop (settle next hop)
        constexpr int margin = 3;     // suffix-start frames discarded (center-reflect boundary)
        const long bufStart = totalSamples - (long)rolling.size();     // hop-aligned
        const long bufStartFrame = bufStart / kHop;
        const long totalFrames = 1 + (long)rolling.size() / kHop;      // center=True whole-buffer count
        const long feedUntil = bufStartFrame + totalFrames - guard;    // absolute frame (exclusive)
        if (feedUntil > streamFedFrames) {
            // Log-mel over only the NEW suffix (a few frames of margin so the fed frames clear the
            // suffix's own reflect-padded start) — not the whole 22 s buffer. Interior frames are
            // identical to the whole-buffer log-mel; this is the compute win (the encoder already
            // only sees new audio via the conv cache).
            const long suffixStartFrame = std::max(bufStartFrame, streamFedFrames - margin);
            const std::size_t off = (std::size_t)(suffixStartFrame - bufStartFrame) * kHop;
            int Tsuf;
            auto lm = frontend.logMel(rolling.data() + off, rolling.size() - off, Tsuf);
            const long fRelStart = streamFedFrames - suffixStartFrame;   // >= 0 (interior)
            const long nFeed = feedUntil - streamFedFrames;
            if (fRelStart >= 0 && nFeed > 0 && fRelStart + nFeed <= Tsuf) {
                auto emits = stream->feed(lm.data() + (std::size_t)fRelStart * kMelDim, (int)nFeed, soft);
                for (auto& e : emits) {
                    chainPh.push_back(e.id);
                    chainTm.push_back(e.frame * 0.04);
                    if (soft) chainAlts.push_back(std::move(e.alts));
                }
                streamFedFrames = feedUntil;
            }
        }
        // bound to ~the largest scale window (+margin); state carries acoustic context.
        const double keep = chainTm.empty() ? 0.0
            : chainTm.back() - (cfg.chainWindowSec * kChainScales[4] + 2.0);
        std::size_t drop = 0;
        while (drop < chainTm.size() && chainTm[drop] < keep) ++drop;
        if (drop) {
            chainPh.erase(chainPh.begin(), chainPh.begin() + drop);
            chainTm.erase(chainTm.begin(), chainTm.begin() + drop);
            if (soft && chainAlts.size() >= drop)
                chainAlts.erase(chainAlts.begin(), chainAlts.begin() + drop);
        }
    }

    // Shared chain matching over a phoneme stream (ph/tm/alts, absolute-time keyed): context-gated
    // early detection + all scale windows -> fire -> vote -> assemble. Used by both the windowed
    // and the streaming decode paths.
    void chainMatch(double timeSec, const std::vector<int>& ph, const std::vector<double>& tm,
                    const PhonAlts& alts, bool soft) {
        // Context-gated early detection: fire the EXPECTED unit once >= earlyPrefix of its prefix
        // matches the decode tail (before the scale fires, like the reference).
        if (cfg.chainEarlyPrefix > 0.0f) {
            const int exp = chainVoter->expectedUnit();
            if (exp >= 0 && chainVoter->streak() >= 1 && (int)ph.size() >= 4) {
                const int L = units->len(exp);
                int minI = std::max(6, (int)std::ceil(cfg.chainEarlyPrefix * L - 1e-9));
                // Near-twin discrimination guard: consecutive ayat that are near-copies
                // (surah 113's «wa min sharri...» family) let the EXPECTED unit's prefix
                // match while the PREVIOUS unit is still being recited — early fires then
                // run several ayat ahead of the reciter (live report + trace, 2026-07-11 pm;
                // unmasked by the phase-3 decode, which no longer deletes the repeats).
                // Thresholds can't fix it — the prefix carries no discriminative information.
                // Gate: the early evidence must fit `expected` decisively BETTER than
                // `expected` resembles the unit that set the expectation:
                //     prefixCost <= dist(expected, lastEmitted) - margin
                // Distinct successors (dist ~0.8+) fire as before; near-twins wait for the
                // whole-unit window match at the true boundary.
                double twinDist = 1.0;
                if (!chainVoter->emitted().empty()) {
                    const auto& a = units->phonemes(chainVoter->emitted().back().unit);
                    const auto& b = units->phonemes(exp);
                    std::size_t lcp = 0;
                    while (lcp < a.size() && lcp < b.size() && a[lcp] == b[lcp]) ++lcp;
                    minI = std::max(minI, (int)lcp + 4);
                    // distance over the REGION THE PROBE SEES (the first minI phonemes) —
                    // 113's twins differ in their tails, so a full-ref distance never bites
                    std::vector<int> ap(a.begin(), a.begin() + std::min((std::size_t)minI, a.size()));
                    std::vector<int> bp(b.begin(), b.begin() + std::min((std::size_t)minI, b.size()));
                    twinDist = normEditDist(ap, bp);
                }
                if (L >= minI) {
                    const double pc = prefixNorm(units->phonemes(exp), ph, minI);
                    if (pc <= cfg.chainCost && pc <= twinDist - 0.15) {
                        if (auto em = chainVoter->onFire(timeSec, exp, pc)) {
                            if (debug) QR_LOG("chain EARLY %s cost=%.2f at %.1fs",
                                              units->key(em->unit).c_str(), pc, timeSec);
                            for (int cu : chainAsm->push(em->unit)) confirmUnit(cu, timeSec);
                        }
                    }
                }
            }
        }
        for (double sc : kChainScales) {
            const double w0 = timeSec - cfg.chainWindowSec * sc;
            auto lo = std::lower_bound(tm.begin(), tm.end(), w0);
            const std::size_t off = lo - tm.begin();
            std::vector<int> win(ph.begin() + off, ph.end());
            if ((int)win.size() < 4) continue;
            PhonAlts winAlts;
            if (soft && alts.size() == ph.size()) winAlts.assign(alts.begin() + off, alts.end());
            auto [u, cost] = windowBest(win, *units, cfg.chainCost, winAlts, cfg.chainSubMin);
            if (u < 0 || cost > cfg.chainCost) continue;
            if (cfg.chainStartAtAyahSec > 0.0f && chainAsm->confirmed().empty()
                && units->segIdxOf(u) >= 2) {   // cold start: decaying penalty on mid-ayah matches
                const double frac = (double)timeSec / cfg.chainStartAtAyahSec;   // 0..1 over the window
                const double m = frac >= 1.0 ? 1.0
                                 : cfg.chainStartAyahMult + (1.0 - cfg.chainStartAyahMult) * frac;
                if (cost > cfg.chainCost * m) continue;   // mid-ayah needs a tighter cost early
            }
            if (auto em = chainVoter->onFire(timeSec, u, cost)) {
                if (debug) QR_LOG("chain EMIT %s cost=%.2f at %.1fs",
                                  units->key(em->unit).c_str(), cost, timeSec);
                // Pre-commit VAD-reset gate arm: only CONFIDENT emissions qualify. Cold-start
                // takes emit their true first unit near-perfectly (measured 0.06-0.14) while
                // quiet-take junk fires sit at 0.35-0.45 — an unconditioned arm let junk trigger
                // early resets that clipped the first ayah (bench -5). Bar = half the fire
                // threshold, scale-free across the clean/phone chainCost configs.
                if (cost <= 0.5f * cfg.chainCost) lastEmitSec = timeSec;
                auto confirms = chainAsm->push(em->unit);
                for (int cu : confirms) confirmUnit(cu, timeSec);
                maybeProvisional(em->unit, confirms.empty(), timeSec, cost);
                // Focused-window trim (windowed only): the emitted unit's audio has served its
                // purpose — drop it so the next ayah decodes without wide-window collapse, keeping
                // a short tail for the successor's in-progress prefix. See types.h.
                if (cfg.chainEmitTrimKeep > 0.0f && !stream) {
                    focusTrim(timeSec - cfg.chainEmitTrimKeep);
                    break;   // window slices over the old decode are stale after the trim
                }
            }
        }
    }

    // Cold-start responsiveness: the FIRST detection is deferred by the assembler until a
    // supporter arrives (junk control) — a 10-20 s dead window on short surahs. Surface the
    // pending unit's ayah as the provisional ACTIVE highlight (lighter, unconfirmed); the
    // first real confirmation overwrites it. Only before anything is confirmed — mid-stream
    // pendings stay invisible (junk jumps would flicker the highlight).
    std::vector<int> provRecent_;   // parent surahs of recent emits (pre-commit corroboration)
    int parentSurah(int unit) const {
        const std::string& pk = units->parentKey(unit);
        return std::stoi(pk.substr(0, pk.find(':')));
    }
    void maybeProvisional(int unit, bool deferred, double timeSec, double cost) {
        provRecent_.push_back(parentSurah(unit));               // track every emit's surah...
        if (provRecent_.size() > 6) provRecent_.erase(provRecent_.begin());
        if (!deferred || !hlCb || !chainAsm->confirmed().empty()) return;
        // Surface a provisional only when it's trustworthy. Raw cost can't separate the correct
        // in-sequence unit from a wrong prefix-collision at the 6x index — their costs OVERLAP on
        // phone audio (correct 0.31-0.34, wrong 0.32-0.40, and wrong units can even fire < 0.30).
        // The reliable signal is CORROBORATION: the correct surah recurs across emits while a wrong
        // prefix jump is one-off. Require the unit's parent surah to appear in >= chainProvVotes
        // recent emits before highlighting. Kills the flicker (one-offs never corroborate, however
        // cheap) without the cold-start latency a pure tight-cost gate caused (correct emits at
        // 0.31-0.34 were suppressed until the late commit). chainProvVotes <= 1 disables the gate.
        const int corrob = (int)std::count(provRecent_.begin(), provRecent_.end(), parentSurah(unit));
        if (corrob < cfg.chainProvVotes) return;
        const auto& pend = chainAsm->pendingUnits();
        if (pend.empty() || pend.back() != unit) return;   // dropped, not deferred
        const auto c = units->parentKey(unit).find(':');
        const std::string& pk = units->parentKey(unit);
        if (debug) QR_LOG("chain PROVISIONAL %s at %.1fs", pk.c_str(), timeSec);
        lastSnap.hasActive = true;
        lastSnap.active = {std::stoi(pk.substr(0, c)), std::stoi(pk.substr(c + 1))};
        hlCb(lastSnap);
    }

    // A confirmed unit: parent transitions drive the public ayah events + highlight; within an
    // ayah, each newly-confirmed waqf segment advances the snapshot's activeSegment. The last
    // unit of an ayah reveals the darker "up next" (successor being verified).
    void confirmUnit(int unit, double timeSec) {
        lastConfirmSec = timeSec;                             // chainResetMaxGap gate
        const std::string& pk = units->parentKey(unit);
        const int seg = std::max(1, units->segIdxOf(unit));   // 1-based; unsegmented -> 1
        const int segCount = units->segCountOf(unit);
        if (pk != chainParent) {
            EventType type = EventType::Detect;
            if (!chainParent.empty()) {
                const auto c = chainParent.find(':');
                const std::string nxt = chainParent.substr(0, c + 1) +
                    std::to_string(std::stoi(chainParent.substr(c + 1)) + 1);
                type = pk == nxt ? EventType::Advance : EventType::Jump;
            }
            chainParent = pk;
            emit(type, pk, timeSec, seg, segCount);           // new ayah + its first segment
        } else if (hlCb) {
            // same ayah, next waqf segment -> update segment progress + re-emit the snapshot
            lastSnap.activeSegment = seg;
            lastSnap.activeSegmentCount = segCount;
            hlCb(lastSnap);
        }
        // last unit of the parent -> the successor ayah is being verified now
        const int succ = units->succFull(unit);
        if ((succ < 0 || units->parentOf(succ) != units->parentOf(unit)) && hlCb && !upNextShown) {
            const auto c = pk.find(':');
            const int surah = std::stoi(pk.substr(0, c));
            const int ayah = std::stoi(pk.substr(c + 1));
            if (units->firstUnitOf(std::to_string(surah) + ":" + std::to_string(ayah + 1)) >= 0) {
                lastSnap.hasUpNext = true;
                lastSnap.upNext = {surah, ayah + 1};
                upNextShown = true;
                hlCb(lastSnap);
            }
        }
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

    // Auto keeps up to ~30 s (the stream matcher's max buffer); sliding needs only ~2
    // windows; Chain needs the largest filter-bank window (2.2 x base).
    const std::size_t cap = impl_->cfg.mode == Mode::Auto ? (std::size_t)(30 * kSr)
        : impl_->cfg.mode == Mode::Chain
            ? (std::size_t)(impl_->cfg.chainWindowSec * kChainScales[4] * kSr)
            : (std::size_t)(2 * impl_->windowSamples());
    if (impl_->rolling.size() > cap) {
        if (impl_->stream) {
            // Streaming feed keys phonemes by absolute frame -> keep the buffer START hop-aligned
            // (erase a whole number of hops) so buffer-frame f maps to absolute frame directly.
            const long bufStart = impl_->totalSamples - (long)impl_->rolling.size();
            long desired = impl_->totalSamples - (long)cap;
            if (desired < 0) desired = 0;
            const long aligned = (desired / kHop) * kHop;   // round down: keeps <= cap + kHop
            const long e = aligned - bufStart;              // >= 0, a multiple of kHop
            if (e > 0) impl_->rolling.erase(impl_->rolling.begin(), impl_->rolling.begin() + e);
        } else {
            impl_->rolling.erase(impl_->rolling.begin(), impl_->rolling.end() - cap);
        }
    }

    // Silero VAD (Auto): feed the same 16 kHz audio in fixed 512-sample chunks; a speech-END
    // event marks an ayah boundary -> drop the buffer + re-anchor so the next ayah decodes
    // fresh. Chain mode needs no VAD reset: windows are time-gated and pause-tolerant.
    // Auto always; Chain only with chainVadReset AND windowed (streaming can't benefit — the
    // focused-window de-crowding is a re-decode technique; see boundaryReset).
    if (impl_->vad && (impl_->cfg.mode != Mode::Chain ||
                       (impl_->cfg.chainVadReset && !impl_->stream))) {
        impl_->vadBuf.insert(impl_->vadBuf.end(), r.begin(), r.end());
        const int VC = SileroVAD::chunkSize();
        bool boundary = false;
        std::size_t off = 0;
        while (impl_->vadBuf.size() - off >= (std::size_t)VC) {
            if (impl_->vad->feed(impl_->vadBuf.data() + off, VC)) boundary = true;
            off += VC;
        }
        impl_->vadBuf.erase(impl_->vadBuf.begin(), impl_->vadBuf.begin() + off);
        if (boundary) {
            // Chain: gate the reset — allow it only when the pause closely follows CONSUMED
            // content: a unit commit OR a confident voter emission (cost <= half the fire
            // threshold; cold starts + surah transitions emit near-perfectly but sit pending
            // awaiting a supporter, and the unfocused window then deletes that supporter via
            // repetition suppression — see research/CLAUDE.md 2026-07-11 pm). Mid-long-ayah
            // breaths follow neither, so the ayah's prefix survives. Non-Chain (Auto) always
            // resets.
            const double now = (double)impl_->totalSamples / kSr;
            const double anchor = std::max(impl_->lastConfirmSec, impl_->lastEmitSec);
            const bool suppress = impl_->cfg.mode == Mode::Chain &&
                now - anchor > impl_->cfg.chainResetMaxGap;
            if (!suppress) impl_->boundaryReset();
        }
    }

    const long hop = (long)((impl_->cfg.mode == Mode::Chain ? impl_->cfg.chainHopSec
                                                            : impl_->cfg.hopSec) * kSr);
    if (impl_->totalSamples - impl_->lastProc >= hop) {
        impl_->lastProc = impl_->totalSamples;
        if (impl_->cfg.mode == Mode::Chain) impl_->stepChain((double)impl_->totalSamples / kSr);
        else if (impl_->cfg.mode == Mode::Auto) impl_->stepAuto((double)impl_->totalSamples / kSr);
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
    if (impl_->chainVoter) impl_->chainVoter->reset();
    if (impl_->chainAsm) impl_->chainAsm->reset();
    if (impl_->stream) impl_->stream->reset();
    impl_->chainPh.clear();
    impl_->chainTm.clear();
    impl_->chainAlts.clear();
    impl_->streamFedFrames = 0;
    impl_->lastConfirmSec = -1e9;
    impl_->lastEmitSec = -1e9;
    impl_->decodeSec = 0.0;
    impl_->decodeHops = 0;
    impl_->chainParent.clear();
    impl_->hl.reset();
    impl_->lastSnap = HighlightSnapshot{};
    impl_->curActive.clear();
    impl_->upNextShown = false;
    if (impl_->vad) impl_->vad->reset();
}

void Detector::setDebug(bool enabled) { impl_->debug.store(enabled); }

void Detector::decodeStats(double& decodeSec, long& hops) const {
    decodeSec = impl_->decodeSec;
    hops = impl_->decodeHops;
}

const char* Detector::version() { return "0.1.0"; }

}  // namespace quranrecite
