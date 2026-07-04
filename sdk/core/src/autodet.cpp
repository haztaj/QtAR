#include "autodet.h"

#include <algorithm>

namespace quranrecite {

namespace {
std::string ayahStr(const AyahId& a) { return std::to_string(a.surah) + ":" + std::to_string(a.ayah); }
}  // namespace

AutoDetector::AutoDetector(const Lexicon& lex, const Config& cfg)
    : lex_(lex),
      slideCtx_(lex, cfg),
      streamCtx_(lex, cfg),
      slider_(lex, slideCtx_, cfg),
      stream_(lex, streamCtx_, cfg) {
    reset();
}

void AutoDetector::reset() {
    slider_.reset();
    stream_.reset();
    emitted_.clear();
    recent_.clear();
}

EventType AutoDetector::kindFor(const std::string& ayah) const {
    if (emitted_.empty()) return EventType::Detect;
    const int i = lex_.orderIndex(emitted_.back());
    const std::string nxt = (i >= 0 && i + 1 < lex_.orderedCount()) ? lex_.orderedKey(i + 1) : std::string();
    return ayah == nxt ? EventType::Advance : EventType::Jump;
}

std::optional<AutoCommit> AutoDetector::reconcile(
    const std::vector<std::pair<std::string, bool>>& cands) {
    // Drop anything recently emitted (dedup + suppress belated backward dupes).
    std::vector<std::pair<std::string, bool>> fresh;
    for (const auto& c : cands)
        if (std::find(recent_.begin(), recent_.end(), c.first) == recent_.end())
            fresh.push_back(c);
    if (fresh.empty()) return std::nullopt;

    std::string pick;
    if (fresh.size() == 1) {
        pick = fresh[0].first;
    } else {
        // Prefer the continuation of the last emitted ayah; else prefer sliding (whole-window).
        const std::string last = emitted_.empty() ? std::string() : emitted_.back();
        const int i = last.empty() ? -1 : lex_.orderIndex(last);
        const std::string nxt = (i >= 0 && i + 1 < lex_.orderedCount()) ? lex_.orderedKey(i + 1) : std::string();
        for (const auto& c : fresh)
            if (c.first == nxt) { pick = c.first; break; }
        if (pick.empty())
            for (const auto& c : fresh)
                if (c.second) { pick = c.first; break; }   // c.second == isSliding
        if (pick.empty()) pick = fresh[0].first;
    }
    const EventType kind = kindFor(pick);            // vs the prior tail, before appending
    emitted_.push_back(pick);
    recent_.push_back(pick);
    while (recent_.size() > recentMax_) recent_.pop_front();
    return AutoCommit{kind, pick};
}

AutoStatus AutoDetector::feed(const std::vector<int>& slidePh, double t,
                              const std::vector<int>& streamPh) {
    std::vector<std::pair<std::string, bool>> cands;   // (ayah, isSliding)

    if (slidePh.size() >= 3) {
        if (auto ev = slider_.process(slidePh, t))
            cands.emplace_back(ayahStr(ev->ayah), true);
    }
    const StreamStatus ss = stream_.feed(streamPh);
    if (ss.commit)
        cands.emplace_back(ss.commit->ayah, false);

    AutoStatus out;
    out.refocusSec = ss.refocusSec;
    out.commit = reconcile(cands);
    return out;
}

}  // namespace quranrecite
