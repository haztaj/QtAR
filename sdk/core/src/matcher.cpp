#include "matcher.h"

#include <algorithm>
#include <sstream>

#include "assets.h"
#include "nlohmann/json.hpp"

namespace quranrecite {

namespace {
constexpr int kStreakCap = 4;   // matches phoneme_matcher.py SequentialContext default

std::pair<int, int> surahAyah(const std::string& key) {
    auto c = key.find(':');
    return {std::stoi(key.substr(0, c)), std::stoi(key.substr(c + 1))};
}
}  // namespace

// Parse tokens.txt ("tok id" per line) and ayah_phonemes.json ("S:A" -> "ph ph ph").
Lexicon::Lexicon(const std::string& ayahPhonemesJson, const std::string& tokensTxt) {
    std::istringstream ts(readFile(tokensTxt));
    std::string line;
    while (std::getline(ts, line)) {
        if (line.empty()) continue;
        auto sp = line.rfind(' ');
        if (sp == std::string::npos) continue;
        tok2id_[line.substr(0, sp)] = std::stoi(line.substr(sp + 1));
    }

    auto j = nlohmann::json::parse(readFile(ayahPhonemesJson));
    for (auto it = j.begin(); it != j.end(); ++it) {
        keys_.push_back(it.key());
        std::vector<int> ids;
        std::istringstream ps(it.value().get<std::string>());
        std::string tok;
        while (ps >> tok) ids.push_back(tokenId(tok));
        phon_.push_back(std::move(ids));
    }

    // canonical (surah, ayah) order for the sequential context.
    order_ = keys_;
    std::sort(order_.begin(), order_.end(), [](const std::string& a, const std::string& b) {
        return surahAyah(a) < surahAyah(b);
    });
    for (int i = 0; i < (int)order_.size(); ++i) orderIdx_[order_[i]] = i;
}

int Lexicon::tokenId(const std::string& tok) const {
    auto it = tok2id_.find(tok);
    return it == tok2id_.end() ? -1 : it->second;
}
int Lexicon::orderIndex(const std::string& key) const {
    auto it = orderIdx_.find(key);
    return it == orderIdx_.end() ? -1 : it->second;
}
const std::string& Lexicon::orderedKey(int idx) const { return order_[idx]; }

float editNorm(const std::vector<int>& a, const std::vector<int>& b) {
    const auto& s = a.size() <= b.size() ? a : b;
    const auto& l = a.size() <= b.size() ? b : a;
    std::vector<int> prev(s.size() + 1);
    for (std::size_t i = 0; i <= s.size(); ++i) prev[i] = (int)i;
    for (std::size_t j = 1; j <= l.size(); ++j) {
        int diag = prev[0];
        prev[0] = (int)j;
        for (std::size_t i = 1; i <= s.size(); ++i) {
            int cur = std::min({prev[i] + 1, prev[i - 1] + 1, diag + (s[i - 1] != l[j - 1] ? 1 : 0)});
            diag = prev[i];
            prev[i] = cur;
        }
    }
    std::size_t m = std::max<std::size_t>(1, std::max(a.size(), b.size()));
    return (float)prev[s.size()] / (float)m;
}

SequentialContext::SequentialContext(const Lexicon& lex, const Config& cfg)
    : lex_(lex), cfg_(cfg) {}

void SequentialContext::reset() { current_.clear(); streak_ = 0; }

void SequentialContext::setCurrent(const std::string& key) {
    if (key.empty()) { current_.clear(); streak_ = 0; return; }
    int i = lex_.orderIndex(current_);
    if (!current_.empty() && i >= 0) {
        int nxt = i + 1;
        bool isExpected = nxt < lex_.orderedCount() && lex_.orderedKey(nxt) == key;
        streak_ = isExpected ? std::min(streak_ + 1, kStreakCap) : 0;
    }
    current_ = key;
}

int SequentialContext::expectedNextIndex() const {
    int i = lex_.orderIndex(current_);
    return (i >= 0 && i + 1 < lex_.orderedCount()) ? i + 1 : -1;
}

float SequentialContext::bonusFor(const std::string& key) const {
    int i0 = lex_.orderIndex(current_);
    if (current_.empty() || i0 < 0) return 0.0f;
    float eff = cfg_.contextBonus + cfg_.streakBonus * streak_;
    float b = 0.0f;
    for (int j = 1; j <= cfg_.contextWindow; ++j) {
        int i = i0 + j;
        if (i < lex_.orderedCount() && lex_.orderedKey(i) == key)
            b = std::max(b, eff * (1.0f - (float)(j - 1) / (cfg_.contextWindow + 1)));
    }
    if (surahAyah(key).first == surahAyah(current_).first)
        b = std::max(b, cfg_.surahBonus);
    return b;
}

}  // namespace quranrecite
