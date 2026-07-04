// Prefix-anchored streaming matcher. Port of demo/streaming.py:StreamDetector.
// Scores each ayah by prefix alignment (input -> ayah prefix) and commits on rank
// persistence. Consumed by AutoDetector alongside the sliding segmenter.
#pragma once
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "matcher.h"
#include "quranrecite/types.h"

namespace quranrecite {

// Min cost to turn `inp` into a PREFIX of `ay` (input fully consumed, ayah free to end
// anywhere). Returns (cost / |inp|, matched-prefix length). Mirrors streaming.prefix_align.
std::pair<float, int> prefixAlign(const std::vector<int>& inp, const std::vector<int>& ay);

struct StreamCommit {
    EventType kind;        // Detect (cold) | Advance (continuation) | Jump
    std::string ayah;
    float cost;
};

struct StreamStatus {
    std::vector<std::tuple<std::string, float, float>> ranked;  // (key, cost, progress), top-3
    std::optional<StreamCommit> commit;    // a new commit this hop (announce it)
    std::optional<float> refocusSec;       // if set, the driver clips the audio buffer to this tail
};

class StreamDetector {
public:
    StreamDetector(const Lexicon& lex, SequentialContext& ctx, const Config& cfg);
    void reset();
    StreamStatus feed(const std::vector<int>& phonemes);

private:
    std::string relation(const std::string& key) const;

    const Lexicon& lex_;
    SequentialContext& ctx_;
    Config cfg_;
    std::string leader_;
    int run_ = 0;
    std::string committed_;
    bool doneRef_ = false;
};

}  // namespace quranrecite
