// Sliding-window segmenter: assembles per-window phoneme detections into an ayah sequence
// by content (no pauses needed). Port of demo/sliding.py:SlidingWindowSegmenter.
// See conformance/spec.md §Stage 2.
#pragma once
#include <optional>
#include <string>
#include <vector>
#include "matcher.h"
#include "quranrecite/types.h"

namespace quranrecite {

class SlidingSegmenter {
public:
    SlidingSegmenter(const Lexicon& lex, SequentialContext& ctx, const Config& cfg);

    // Feed one window's phoneme ids + its center time. Returns an event if this window
    // produced a detect/advance/jump, else nullopt.
    std::optional<AyahEvent> process(const std::vector<int>& windowPhonemes, double timeSec);

    void reset();

private:
    // Best (key, cost) by context-biased whole-window edit distance, length-pruned.
    bool windowBest(const std::vector<int>& w, std::string& outKey, float& outCost) const;

    const Lexicon& lex_;
    SequentialContext& ctx_;
    Config cfg_;
    std::string current_;
    std::string pending_;
    int votes_ = 0;
};

}  // namespace quranrecite
