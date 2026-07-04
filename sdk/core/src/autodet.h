// AutoDetector: runs the sliding segmenter + the stream matcher and merges their commits.
// Port of demo/auto.py:AutoDetector. Handles any ayah length (sliding = short, stream = long),
// which never conflict (sliding is silent on long ayat). See demo/CLAUDE.md.
#pragma once
#include <deque>
#include <optional>
#include <string>
#include <vector>
#include "matcher.h"
#include "segmenter.h"
#include "stream.h"
#include "quranrecite/types.h"

namespace quranrecite {

struct AutoCommit {
    EventType event;       // Detect | Advance | Jump (derived from the emitted sequence)
    std::string ayah;      // "surah:ayah"
};

struct AutoStatus {
    std::optional<AutoCommit> commit;   // the merged detection this hop
    std::optional<float> refocusSec;    // from the stream sub-matcher (driver clips its buffer)
};

class AutoDetector {
public:
    AutoDetector(const Lexicon& lex, const Config& cfg);
    void reset();
    // slidePh: decode of the fixed sliding window; streamPh: decode of the stream anchored buffer.
    AutoStatus feed(const std::vector<int>& slidePh, double t, const std::vector<int>& streamPh);

private:
    EventType kindFor(const std::string& ayah) const;
    std::optional<AutoCommit> reconcile(const std::vector<std::pair<std::string, bool>>& cands);

    const Lexicon& lex_;
    SequentialContext slideCtx_;
    SequentialContext streamCtx_;
    SlidingSegmenter slider_;
    StreamDetector stream_;
    std::vector<std::string> emitted_;
    std::deque<std::string> recent_;
    std::size_t recentMax_ = 6;
};

}  // namespace quranrecite
