// QuranRecite SDK — main on-device detector (public C++ API).
//
// Pipeline:  PCM in -> 16k log-mel -> ONNX Runtime (int8) -> CTC greedy phonemes
//            -> sliding-window matcher/segmenter -> ayah events out.
//
// The detector is a pure processing engine: the host feeds PCM (or the platform layer's
// managed capture does). Events are delivered via the callback; the caller marshals to a
// UI thread as needed (the platform wrappers do this).
#pragma once
#include <memory>
#include "quranrecite/types.h"

namespace quranrecite {

class Detector {
public:
    explicit Detector(const Config& config);
    ~Detector();
    Detector(Detector&&) noexcept;
    Detector& operator=(Detector&&) noexcept;

    // Register the granular event sink (detect/advance/jump). Called from the engine's
    // worker thread. Kept for back-compat; the snapshot callback below is the primary,
    // centralized contract.
    void setEventCallback(EventCallback cb);

    // Register the highlight-state sink: one render-ready snapshot per change (ambiguity
    // deferred, never guessed — see quranrecite/types.h + conformance/spec.md §Stage 3).
    // This is the contract UIs should render. Called from the engine's worker thread.
    void setHighlightCallback(HighlightCallback cb);

    // Feed mono PCM (float -1..1). `sampleRate` may differ from 16k (resampled internally).
    // Thread-safe wrt the engine; intended to be called from one capture thread.
    void feed(const float* pcm, std::size_t numSamples, int sampleRate);

    // Convenience for 16-bit PCM (e.g. Android AudioRecord ENCODING_PCM_16BIT).
    void feedPcm16(const short* pcm, std::size_t numSamples, int sampleRate);

    // Clear the rolling buffer + sequential context (start a fresh recitation session).
    void reset();

    // Page-context prior (Mode::Chain, requires Config::chainPageBonus > 0): tell the detector
    // which ayat are on the page(s) the user is currently viewing (e.g. the visible page + the
    // next one). On-page ayat get a soft matching preference — they win twin ambiguities and
    // fire a touch easier — while off-page ayat still detect (not a hard filter). Call it
    // whenever the viewed page changes; pass an empty list to clear. Thread-safe wrt feed.
    void setPageContext(const std::vector<AyahId>& pageAyat);

    // Toggle the collision blacklist at runtime (Mode::Chain; no-op unless Config::chainBlacklistPath
    // loaded a mask). When on, high-collision units (كلّا, قل الله, 55:1 …) are cold-fire-suppressed
    // and fire only via page/sequence context. Exposed for live A/B comparison in the app.
    void setBlacklistEnabled(bool enabled);

    // Runtime debug logging (Android logcat, tag "QuranReciteCore"): per-hop decode stats, VAD
    // resets, commits. Off by default (zero overhead); toggle live from the host. No-op elsewhere.
    void setDebug(bool enabled);

    // Cumulative acoustic-decode wall-clock since construction (Mode::Chain): `decodeSec` is the
    // total time spent decoding phonemes (windowed re-decode, or the streaming feed) over `hops`
    // non-silent hops. For RTF/battery comparison of the streaming vs windowed path.
    void decodeStats(double& decodeSec, long& hops) const;

    static const char* version();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace quranrecite
