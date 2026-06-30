#include "matcher.h"
#include <algorithm>

namespace quranrecite {

// TODO(port): parse ayah_phonemes.json (key -> "ph ph ph") and tokens.txt (tok -> id);
// map each phoneme string to its token id; build the (surah,ayah)-sorted order_ list.
Lexicon::Lexicon(const std::string& /*ayahJson*/, const std::string& /*tokensTxt*/) {}

int Lexicon::tokenId(const std::string& tok) const {
    auto it = tok2id_.find(tok); return it == tok2id_.end() ? -1 : it->second;
}
int Lexicon::orderIndex(const std::string& key) const {
    auto it = orderIdx_.find(key); return it == orderIdx_.end() ? -1 : it->second;
}
const std::string& Lexicon::orderedKey(int idx) const { return order_[idx]; }

// Standard Levenshtein normalized by max length. See conformance/spec.md.
float editNorm(const std::vector<int>& a, const std::vector<int>& b) {
    const auto& s = a.size() <= b.size() ? a : b;
    const auto& l = a.size() <= b.size() ? b : a;
    std::vector<int> prev(s.size() + 1);
    for (std::size_t i = 0; i <= s.size(); ++i) prev[i] = static_cast<int>(i);
    for (std::size_t j = 1; j <= l.size(); ++j) {
        int diag = prev[0];
        prev[0] = static_cast<int>(j);
        for (std::size_t i = 1; i <= s.size(); ++i) {
            int cur = std::min({prev[i] + 1, prev[i - 1] + 1, diag + (s[i - 1] != l[j - 1] ? 1 : 0)});
            diag = prev[i];
            prev[i] = cur;
        }
    }
    std::size_t m = std::max<std::size_t>(1, std::max(a.size(), b.size()));
    return static_cast<float>(prev[s.size()]) / static_cast<float>(m);
}

SequentialContext::SequentialContext(const Lexicon& lex, const Config& cfg)
    : lex_(lex), cfg_(cfg) {}

void SequentialContext::reset() { current_.clear(); streak_ = 0; }

void SequentialContext::setCurrent(const std::string& key) {
    // TODO(port): streak grows only when `key` is the expected next of the old current,
    // else resets (spec.md §Stage 2). Then current_ = key.
    (void)key;
}

int SequentialContext::expectedNextIndex() const {
    int i = lex_.orderIndex(current_);
    return (i >= 0 && i + 1 < lex_.orderedCount()) ? i + 1 : -1;
}

float SequentialContext::bonusFor(const std::string& /*key*/) const {
    // TODO(port): eff = contextBonus + streakBonus*streak; next `contextWindow` ayat in
    // canonical order get eff*(1-(j-1)/(window+1)); same-surah gets max(prev, surahBonus).
    return 0.0f;
}

}  // namespace quranrecite
