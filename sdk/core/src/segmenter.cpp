#include "segmenter.h"

namespace quranrecite {

SlidingSegmenter::SlidingSegmenter(const Lexicon& lex, SequentialContext& ctx, const Config& cfg)
    : lex_(lex), ctx_(ctx), cfg_(cfg) {}

void SlidingSegmenter::reset() { current_.clear(); pending_.clear(); votes_ = 0; ctx_.reset(); }

bool SlidingSegmenter::windowBest(const std::vector<int>& w, std::string& outKey, float& outCost) const {
    // TODO(port): for each ayah with length L, length-prune (0.6*n <= L <= n/0.6),
    // score = editNorm(w, ayahPhonemes) - ctx_.bonusFor(key); keep the minimum.
    // Return false if no candidate. See conformance/spec.md §Stage 2.
    (void)w; (void)outKey; (void)outCost;
    return false;
}

std::optional<AyahEvent> SlidingSegmenter::process(const std::vector<int>& w, double t) {
    if (w.size() < 3) return std::nullopt;
    std::string key; float cost = 0.0f;
    if (!windowBest(w, key, cost) || cost > cfg_.windowCost) return std::nullopt;

    // TODO(port): state machine (spec.md §Stage 2):
    //   current empty            -> set current, ctx.setCurrent, emit Detect
    //   key == current           -> clear pending, no event
    //   key == expected next     -> set current, ctx.setCurrent, emit Advance(from=old)
    //   else                     -> vote; >= jumpVotes consecutive -> emit Jump
    // confidence = 1 - cost (clamped to [0,1]).
    (void)key; (void)cost; (void)t;
    return std::nullopt;
}

}  // namespace quranrecite
