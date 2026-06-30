#include "segmenter.h"

#include <algorithm>

namespace quranrecite {

SlidingSegmenter::SlidingSegmenter(const Lexicon& lex, SequentialContext& ctx, const Config& cfg)
    : lex_(lex), ctx_(ctx), cfg_(cfg) {}

void SlidingSegmenter::reset() {
    current_.clear();
    pending_.clear();
    votes_ = 0;
    ctx_.reset();
}

// Best (key, cost) by context-biased whole-window edit distance, length-pruned (len_tol=0.6).
bool SlidingSegmenter::windowBest(const std::vector<int>& w, std::string& outKey, float& outCost) const {
    const float lenTol = 0.6f;
    const float n = (float)w.size();
    float best = 1e9f;
    bool found = false;
    for (std::size_t i = 0; i < lex_.size(); ++i) {
        const auto& p = lex_.phonemes(i);
        float L = (float)p.size();
        if (!(lenTol * n <= L && L <= n / lenTol)) continue;          // length prune
        float c = editNorm(w, p) - ctx_.bonusFor(lex_.key(i));        // context bonus
        if (c < best) { best = c; outKey = lex_.key(i); found = true; }
    }
    outCost = best;
    return found;
}

std::optional<AyahEvent> SlidingSegmenter::process(const std::vector<int>& w, double t) {
    if (w.size() < 3) return std::nullopt;
    std::string key;
    float cost = 0.0f;
    if (!windowBest(w, key, cost) || cost > cfg_.windowCost) return std::nullopt;

    auto sa = [](const std::string& k) {
        auto c = k.find(':');
        return AyahId{std::stoi(k.substr(0, c)), std::stoi(k.substr(c + 1))};
    };
    auto commit = [&](EventType type) -> AyahEvent {
        AyahId from = current_.empty() ? AyahId{} : sa(current_);
        current_ = key;
        ctx_.setCurrent(key);
        pending_.clear();
        votes_ = 0;
        AyahEvent e;
        e.type = type;
        e.ayah = sa(key);
        e.from = from;
        e.confidence = std::clamp(1.0f - cost, 0.0f, 1.0f);
        e.timeSec = t;
        return e;
    };
    auto vote = [&](int need) {
        if (key == pending_) ++votes_;
        else { pending_ = key; votes_ = 1; }
        return votes_ >= need;
    };

    if (current_.empty()) return commit(EventType::Detect);          // cold start
    if (key == current_) { pending_.clear(); votes_ = 0; return std::nullopt; }

    int nx = ctx_.expectedNextIndex();
    std::string expected = nx >= 0 ? lex_.orderedKey(nx) : std::string();
    if (key == expected) return commit(EventType::Advance);          // continuation
    if (vote(cfg_.jumpVotes)) return commit(EventType::Jump);        // unexpected -> jump
    return std::nullopt;
}

}  // namespace quranrecite
