#include "stream.h"

#include <algorithm>

namespace quranrecite {

std::pair<float, int> prefixAlign(const std::vector<int>& inp, const std::vector<int>& ay) {
    const int n = static_cast<int>(inp.size()), L = static_cast<int>(ay.size());
    if (n == 0) return {0.0f, 0};
    std::vector<int> prev(L + 1), cur(L + 1);
    for (int k = 0; k <= L; ++k) prev[k] = k;               // D[0][k] = k
    for (int i = 1; i <= n; ++i) {
        cur[0] = i;
        const int ci = inp[i - 1];
        for (int k = 1; k <= L; ++k) {
            const int sub = prev[k - 1] + (ci != ay[k - 1] ? 1 : 0);
            cur[k] = std::min(std::min(prev[k] + 1, cur[k - 1] + 1), sub);
        }
        std::swap(prev, cur);
    }
    int bestK = 0, bestV = prev[0];                          // min over k of D[n][k]
    for (int k = 1; k <= L; ++k)
        if (prev[k] < bestV) { bestV = prev[k]; bestK = k; }
    return {static_cast<float>(bestV) / n, bestK};
}

namespace {
void parseKey(const std::string& sa, int& s, int& a) {
    const auto c = sa.find(':');
    s = std::stoi(sa.substr(0, c));
    a = std::stoi(sa.substr(c + 1));
}
}  // namespace

StreamDetector::StreamDetector(const Lexicon& lex, SequentialContext& ctx, const Config& cfg)
    : lex_(lex), ctx_(ctx), cfg_(cfg) { reset(); }

void StreamDetector::reset() {
    ctx_.reset();
    leader_.clear();
    run_ = 0;
    committed_.clear();
    doneRef_ = false;
}

std::string StreamDetector::relation(const std::string& key) const {
    const std::string& cur = ctx_.current();
    if (cur.empty()) return "cold";
    if (key == cur) return "current";
    const int i = lex_.orderIndex(cur);
    const std::string nxt = (i >= 0 && i + 1 < lex_.orderedCount()) ? lex_.orderedKey(i + 1) : std::string();
    if (key == nxt) return "continuation";
    int cs, ca, ks, ka;
    parseKey(cur, cs, ca);
    parseKey(key, ks, ka);
    if (ks == cs && ka < ca) return "backward";
    return "jump";
}

StreamStatus StreamDetector::feed(const std::vector<int>& ph) {
    const int n = static_cast<int>(ph.size());
    std::vector<std::tuple<float, std::string, float>> scored;   // (cost, key, progress)
    for (std::size_t i = 0; i < lex_.size(); ++i) {
        const auto& p = lex_.phonemes(i);
        const int L = static_cast<int>(p.size());
        if (static_cast<float>(L) < cfg_.lenTol * n) continue;   // too short to explain input
        const auto pr = prefixAlign(ph, p);
        const std::string& key = lex_.key(i);
        scored.emplace_back(pr.first - ctx_.bonusFor(key), key, static_cast<float>(pr.second) / L);
    }
    StreamStatus st;
    if (scored.empty()) return st;
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
    for (int i = 0; i < static_cast<int>(std::min<std::size_t>(3, scored.size())); ++i)
        st.ranked.emplace_back(std::get<1>(scored[i]), std::get<0>(scored[i]), std::get<2>(scored[i]));

    const float topCost = std::get<0>(scored[0]);
    const std::string top = std::get<1>(scored[0]);
    const float topProg = std::get<2>(scored[0]);

    run_ = (top == leader_) ? run_ + 1 : 1;                  // rank persistence
    leader_ = top;
    const std::string rel = relation(top);

    if (top != committed_ && (rel == "continuation" || rel == "jump") && run_ == 2)
        st.refocusSec = cfg_.keepLongSec;                    // long-ayah leader change -> bound window

    const int need = (rel == "cold" || rel == "current" || rel == "continuation")
                         ? cfg_.streamPersistence : cfg_.streamJumpPersistence;
    const bool eligible = topProg >= cfg_.streamMinProgress && topCost <= cfg_.commitCostMax
                          && rel != "backward";
    if (eligible && run_ >= need && top != committed_) {
        committed_ = top;
        ctx_.setCurrent(top);
        run_ = 0;
        doneRef_ = false;
        const EventType kind = rel == "cold" ? EventType::Detect
                             : (rel == "continuation" ? EventType::Advance : EventType::Jump);
        st.commit = StreamCommit{kind, top, topCost};
        if (topProg >= cfg_.doneProgress) { st.refocusSec = cfg_.keepDoneSec; doneRef_ = true; }
    }

    // A fully-recited committed ayah keeps hogging #1 (buffer still starts with it) -> release it.
    const std::string detected = !committed_.empty()
        ? committed_ : (topProg >= cfg_.streamMinProgress ? top : std::string());
    if (!committed_.empty() && !doneRef_ && detected == committed_) {
        float cp = 0.0f;
        for (const auto& s : scored)
            if (std::get<1>(s) == committed_) { cp = std::get<2>(s); break; }
        if (cp >= 0.95f) { st.refocusSec = cfg_.keepDoneSec; doneRef_ = true; }
    }
    return st;
}

}  // namespace quranrecite
