// Stage-2 matcher: phoneme lexicon + normalized edit distance + sticky sequential context.
// Port of matcher/phoneme_matcher.py (the pieces the sliding segmenter uses).
// See conformance/spec.md §Stage 2.
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "quranrecite/types.h"

namespace quranrecite {

// Loaded lexicon: each ayah key "S:A" -> its phoneme-id sequence.
class Lexicon {
public:
    Lexicon(const std::string& ayahPhonemesJson, const std::string& tokensTxt);
    std::size_t size() const { return keys_.size(); }
    const std::vector<int>& phonemes(std::size_t i) const { return phon_[i]; }
    const std::string& key(std::size_t i) const { return keys_[i]; }
    int tokenId(const std::string& tok) const;   // phoneme string -> id (for CTC decode)
    // canonical (surah,ayah) order index, for the sequential context.
    int orderIndex(const std::string& key) const;
    const std::string& orderedKey(int idx) const;
    int orderedCount() const { return static_cast<int>(order_.size()); }

private:
    std::vector<std::string> keys_;
    std::vector<std::vector<int>> phon_;          // parallel to keys_
    std::unordered_map<std::string, int> tok2id_;
    std::vector<std::string> order_;              // keys sorted by (surah,ayah)
    std::unordered_map<std::string, int> orderIdx_;
};

// Levenshtein(a,b) / max(len). conformance/assets/edit_cases.json must match.
float editNorm(const std::vector<int>& a, const std::vector<int>& b);

// Sticky continuation prior (SequentialContext.bonus_for / set_current).
class SequentialContext {
public:
    SequentialContext(const Lexicon& lex, const Config& cfg);
    void reset();
    void setCurrent(const std::string& key);      // grows/resets streak, see spec.md
    float bonusFor(const std::string& key) const; // subtracted from a candidate's cost
    int expectedNextIndex() const;                // -1 if none
    const std::string& current() const { return current_; }
    int streak() const { return streak_; }

private:
    const Lexicon& lex_;
    Config cfg_;
    std::string current_;
    int streak_ = 0;
};

}  // namespace quranrecite
