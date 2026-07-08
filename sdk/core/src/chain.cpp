#include "chain.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

#include "assets.h"
#include "nlohmann/json.hpp"

namespace quranrecite {

namespace {
// Reference constants (research/chain_sliding.py) — conformance-pinned, do not tune here.
constexpr int kShortlist = 60;          // raw-count shortlist size
constexpr int kShortlistNorm = 20;      // length-normalized union size
constexpr double kCoverBonus = 0.15;    // selection = cost - bonus*coverage
constexpr double kStrongCost = 0.15;    // near-certain fire: commits with a single vote
constexpr double kMinAdvance = 2.0;     // emission gate past the last commit
constexpr double kRepeatSuppress = 20.0;
constexpr int kStreakMin = 3;           // consecutive expected commits before jumps escalate
constexpr int kStreakExtra = 1;         // extra votes a NON-near jump needs when streaked
constexpr int kNearAhead = 2;           // same surah, 0..this many ayat ahead = cheap recovery

struct KeyParts { int s, a, seg; };
KeyParts parseKey(const std::string& k) {
    auto h = k.find('#');
    auto c = k.find(':');
    return {std::stoi(k.substr(0, c)),
            std::stoi(k.substr(c + 1, (h == std::string::npos ? k.size() : h) - c - 1)),
            h == std::string::npos ? 0 : std::stoi(k.substr(h + 1))};
}

long long packGram(int a, int b, int c) {
    return ((long long)a << 40) | ((long long)b << 20) | (long long)c;
}

// Infix-normalized edit distance: best alignment of `ref` as a SUBSTRING of `win`
// (free leading/trailing window gaps), / len(ref). (_infix_norm in the reference.)
double infixNorm(const std::vector<int>& ref, const std::vector<int>& win) {
    const int m = (int)ref.size(), n = (int)win.size();
    std::vector<int> prev(n + 1, 0), cur(n + 1);
    for (int i = 1; i <= m; ++i) {
        cur[0] = i;
        const int ri = ref[i - 1];
        for (int j = 1; j <= n; ++j)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ri != win[j - 1] ? 1 : 0)});
        std::swap(prev, cur);
    }
    return (double)*std::min_element(prev.begin(), prev.end()) / std::max(1, m);
}
}  // namespace

UnitIndex::UnitIndex(const std::string& unitPhonemesJson, const std::string& tokensTxt) {
    std::istringstream ts(readFile(tokensTxt));
    std::string line;
    while (std::getline(ts, line)) {
        if (line.empty()) continue;
        auto sp = line.rfind(' ');
        if (sp == std::string::npos) continue;
        tok2id_[line.substr(0, sp)] = std::stoi(line.substr(sp + 1));
    }

    auto j = nlohmann::json::parse(readFile(unitPhonemesJson));
    std::vector<std::pair<std::string, std::string>> raw;
    for (auto it = j.begin(); it != j.end(); ++it)
        raw.emplace_back(it.key(), it.value().get<std::string>());
    // canonical (surah, ayah, segment) order — the reference's deterministic tie-break
    std::sort(raw.begin(), raw.end(), [](const auto& a, const auto& b) {
        auto ka = parseKey(a.first), kb = parseKey(b.first);
        return std::tie(ka.s, ka.a, ka.seg) < std::tie(kb.s, kb.a, kb.seg);
    });

    std::unordered_map<std::string, int> parentId;    // parent key -> dense id
    for (auto& [key, seq] : raw) {
        int u = (int)keys_.size();
        keys_.push_back(key);
        keyIdx_[key] = u;
        std::vector<int> ids;
        std::istringstream ps(seq);
        std::string tok;
        while (ps >> tok) ids.push_back(tokenId(tok));
        phon_.push_back(std::move(ids));
        auto h = key.find('#');
        std::string pk = h == std::string::npos ? key : key.substr(0, h);
        auto [it, ins] = parentId.try_emplace(pk, (int)parentKeys_.size());
        if (ins) { parentKeys_.push_back(pk); parentFirst_[pk] = u; }
        parent_.push_back(it->second);
        segIdx_.push_back(h == std::string::npos ? 0 : std::stoi(key.substr(h + 1)));
    }

    // succFull: canonical adjacency within a parent; last unit -> next ayah's first unit
    succ_.assign(keys_.size(), -1);
    for (int u = 0; u < (int)keys_.size(); ++u) {
        if (u + 1 < (int)keys_.size() && parent_[u + 1] == parent_[u]) {
            succ_[u] = u + 1;
            continue;
        }
        auto kp = parseKey(keys_[u]);
        auto it = parentFirst_.find(std::to_string(kp.s) + ":" + std::to_string(kp.a + 1));
        if (it != parentFirst_.end()) succ_[u] = it->second;
    }

    // 3-gram inverted index; unit ids ascend (canonical order) — deterministic retrieval
    for (int u = 0; u < (int)keys_.size(); ++u) {
        const auto& ph = phon_[u];
        for (int i = 0; i + 2 < (int)ph.size(); ++i) {
            auto& v = grams_[packGram(ph[i], ph[i + 1], ph[i + 2])];
            if (v.empty() || v.back() != u) v.push_back(u);  // set semantics (dedupe per unit)
        }
    }
}

int UnitIndex::unitOf(const std::string& key) const {
    auto it = keyIdx_.find(key);
    return it == keyIdx_.end() ? -1 : it->second;
}
int UnitIndex::tokenId(const std::string& tok) const {
    auto it = tok2id_.find(tok);
    return it == tok2id_.end() ? -1 : it->second;
}
const std::vector<int>* UnitIndex::gramUnits(int a, int b, int c) const {
    auto it = grams_.find(packGram(a, b, c));
    return it == grams_.end() ? nullptr : &it->second;
}
const std::string& UnitIndex::parentKey(int unit) const { return parentKeys_[parent_[unit]]; }
int UnitIndex::firstUnitOf(const std::string& parentKey) const {
    auto it = parentFirst_.find(parentKey);
    return it == parentFirst_.end() ? -1 : it->second;
}

std::pair<int, double> windowBest(const std::vector<int>& win, const UnitIndex& idx,
                                  double fireCost) {
    const int n = (int)win.size();
    // Counter with Python insertion-order semantics: first-seen while scanning window
    // positions ascending, posting lists ascending — ties in the sorts below break by
    // this order, exactly like Counter.most_common / heapq.nlargest (both stable).
    std::unordered_map<int, int> cnt;
    std::vector<int> seen;                             // insertion order
    for (int i = 0; i + 2 < n; ++i) {
        const auto* g = idx.gramUnits(win[i], win[i + 1], win[i + 2]);
        if (!g) continue;
        for (int u : *g) {
            auto [it, ins] = cnt.try_emplace(u, 0);
            if (ins) seen.push_back(u);
            ++it->second;
        }
    }
    // shortlist: raw-count top-60 UNION length-normalized top-20 (over the FULL counter)
    std::vector<int> byRaw = seen, byNorm = seen;
    std::stable_sort(byRaw.begin(), byRaw.end(),
                     [&](int a, int b) { return cnt[a] > cnt[b]; });
    std::stable_sort(byNorm.begin(), byNorm.end(), [&](int a, int b) {
        return (double)cnt[a] / idx.len(a) > (double)cnt[b] / idx.len(b);
    });
    if ((int)byRaw.size() > kShortlist) byRaw.resize(kShortlist);
    if ((int)byNorm.size() > kShortlistNorm) byNorm.resize(kShortlistNorm);
    std::vector<int> shortlist = byRaw;
    for (int u : byNorm)
        if (std::find(shortlist.begin(), shortlist.end(), u) == shortlist.end())
            shortlist.push_back(u);

    // tight length band (each window scale serves its own ref-size class) + blended
    // selection: cost - kCoverBonus * coverage among fires <= fireCost
    int bestUnit = -1;
    double bestCost = 1e9, bestSel = 1e9;
    bool anyFire = false;
    for (int u : shortlist) {
        const int L = idx.len(u);
        if (!(0.5 * n <= L && L <= 1.3 * n)) continue;
        const double cost = infixNorm(idx.phonemes(u), win);
        const double sel = cost - kCoverBonus * std::min(L, n) / (double)n;
        if (cost <= fireCost && sel < bestSel) {
            bestSel = sel; bestUnit = u; bestCost = cost; anyFire = true;
        } else if (!anyFire && cost < bestCost) {
            bestUnit = u; bestCost = cost;
        }
    }
    return {bestUnit, bestCost};
}

double prefixNorm(const std::vector<int>& ref, const std::vector<int>& win, int minI) {
    const int m = (int)ref.size(), n = (int)win.size();
    constexpr int kEndSlack = 2;             // CTC timing noise at the window edge
    std::vector<int> prev(n + 1, 0), cur(n + 1);
    double best = 1e9;
    for (int i = 1; i <= m; ++i) {
        cur[0] = i;
        const int ri = ref[i - 1];
        for (int j = 1; j <= n; ++j)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ri != win[j - 1] ? 1 : 0)});
        if (i >= minI) {
            int tail = cur[n];
            for (int j = std::max(0, n - kEndSlack); j <= n; ++j) tail = std::min(tail, cur[j]);
            best = std::min(best, (double)tail / i);
        }
        std::swap(prev, cur);
    }
    return best;
}

void ChainVoter::reset() {
    emitted_.clear();
    expected_ = pending_ = -1;
    votes_ = 0;
    streak_ = 0;
    consumed_ = -1e9;
}

std::optional<UnitEmission> ChainVoter::onFire(double w1, int unit, double cost) {
    if (w1 < consumed_ + kMinAdvance) return std::nullopt;   // inside the last commit
    if (!emitted_.empty() && unit == emitted_.back().unit) return std::nullopt;
    for (const auto& e : emitted_) {                          // first occurrence only
        if (e.unit == unit) {
            if (w1 - e.timeSec < kRepeatSuppress) return std::nullopt;
            break;
        }
    }
    // Twin substitution: exact twins tie on cost AND length — context picks.
    if (expected_ >= 0 && unit != expected_ && idx_.phonemes(unit) == idx_.phonemes(expected_))
        unit = expected_;
    // Streak escalation: after kStreakMin consecutive expected commits, a fire that is
    // neither the expected successor nor a NEAR continuation (same surah, up to
    // kNearAhead ayat ahead — keeps recovery after a missed unit cheap) needs extra
    // votes, and a strong fire no longer commits alone.
    bool near = true;
    if (unit != expected_ && !emitted_.empty()) {
        const auto lp = parseKey(idx_.key(emitted_.back().unit));
        const auto tp = parseKey(idx_.key(unit));
        near = tp.s == lp.s && tp.a - lp.a >= 0 && tp.a - lp.a <= kNearAhead;
    }
    const bool escalate = streak_ >= kStreakMin && unit != expected_ && !near;
    int need = unit == expected_ ? p_.votesNext : p_.votesJump + (escalate ? kStreakExtra : 0);
    if (cost <= kStrongCost) need = std::min(need, escalate ? 2 : 1);  // confidence-scaled
    if (unit == pending_) ++votes_;
    else { pending_ = unit; votes_ = 1; }
    if (votes_ < need) return std::nullopt;
    streak_ = (unit == expected_ && expected_ >= 0) ? streak_ + 1 : 0;
    UnitEmission em{unit, w1};
    emitted_.push_back(em);
    consumed_ = w1 - 2.0;                                     // keep overlap for the next unit
    expected_ = idx_.succFull(unit);
    pending_ = -1;
    votes_ = 0;
    return em;
}

void ChainAssembler::reset() {
    confirmed_.clear();
    pending_.clear();
}

bool ChainAssembler::supports(int p, int u) const {
    return u == idx_.succFull(p) ||
           (idx_.parentOf(u) == idx_.parentOf(p) && idx_.segIdxOf(u) > idx_.segIdxOf(p));
}

std::vector<int> ChainAssembler::push(int u) {
    int sup = -1;
    for (int k = (int)pending_.size() - 1; k >= 0; --k)
        if (supports(pending_[k], u)) { sup = k; break; }
    if (sup >= 0) {                                    // retro-confirm the supported pending
        std::vector<int> out = {pending_[sup], u};     // (junk between/after it is dropped)
        confirmed_.push_back(pending_[sup]);
        confirmed_.push_back(u);
        pending_.clear();
        return out;
    }
    if (!confirmed_.empty()) {
        const int last = confirmed_.back();
        if (u == idx_.succFull(last)) {                // expected successor: confirm now
            confirmed_.push_back(u);
            pending_.clear();
            return {u};
        }
        if (idx_.parentOf(u) == idx_.parentOf(last)) {
            if (idx_.segIdxOf(u) > idx_.segIdxOf(last)) {
                confirmed_.push_back(u);               // forward skip within the parent
                pending_.clear();
                return {u};
            }
            return {};                                 // backward/repeat: drop (re-fire)
        }
    }
    pending_.push_back(u);                             // unexpected jump: await support
    if (pending_.size() > 2) pending_.erase(pending_.begin());  // oldest ages out
    return {};
}

std::vector<int> ChainAssembler::flush() {
    std::vector<int> out;
    for (int p : pending_) {                           // end of stream: chainable tail
        if (!confirmed_.empty() &&
            (p == idx_.succFull(confirmed_.back()) ||
             (idx_.parentOf(p) == idx_.parentOf(confirmed_.back()) &&
              idx_.segIdxOf(p) > idx_.segIdxOf(confirmed_.back())))) {
            confirmed_.push_back(p);
            out.push_back(p);
        }
    }
    if (confirmed_.empty() && !pending_.empty()) {     // lone emissions — keep the first
        confirmed_.push_back(pending_[0]);
        out.push_back(pending_[0]);
    }
    pending_.clear();
    return out;
}

std::vector<UnitEmission> decodeStream(const std::vector<int>& phonemes,
                                       const std::vector<double>& times,
                                       const UnitIndex& idx, const ChainParams& p) {
    if (phonemes.empty()) return {};
    // kind 0 = prefix-check event (largest scale only, carries the window); kind 1 = fire
    struct Ev { double w1; int kind; int unit; double cost; std::vector<int> win; };
    std::vector<Ev> evs;
    const double tEnd = times.back();
    const int N = (int)phonemes.size();
    const int nScales = (int)(sizeof(kChainScales) / sizeof(kChainScales[0]));
    for (int si = 0; si < nScales; ++si) {
        const double w = p.windowSec * kChainScales[si];
        const bool largest = si == nScales - 1;
        double t = 0.0;
        int j0 = 0;
        while (t <= tEnd + 1e-6) {
            const double w0 = t, w1 = t + w;
            while (j0 < N && times[j0] < w0) ++j0;
            int j1 = j0;
            while (j1 < N && times[j1] < w1) ++j1;
            std::vector<int> win(phonemes.begin() + j0, phonemes.begin() + j1);
            t += p.hopSec;
            if ((int)win.size() < 4) continue;
            if (p.earlyPrefix > 0 && largest) evs.push_back({w1, 0, -1, 0.0, win});
            auto [u, cost] = windowBest(win, idx, p.costThresh);
            if (u >= 0 && cost <= p.costThresh) evs.push_back({w1, 1, u, cost, {}});
        }
    }
    // reference sorts (w1, kind, key, cost) — Python tuple order, key is a string
    // (prefix events carry key "" which sorts before any unit key)
    std::sort(evs.begin(), evs.end(), [&](const Ev& a, const Ev& b) {
        if (a.w1 != b.w1) return a.w1 < b.w1;
        if (a.kind != b.kind) return a.kind < b.kind;
        const std::string& ka = a.unit >= 0 ? idx.key(a.unit) : std::string();
        const std::string& kb = b.unit >= 0 ? idx.key(b.unit) : std::string();
        if (ka != kb) return ka < kb;
        return a.cost < b.cost;
    });
    ChainVoter voter(idx, p);
    for (const auto& e : evs) {
        if (e.kind == 0) {
            // Early-prefix only on a TRUSTED expectation (last commit extended the
            // chain) — after a junk/jump emission it would probe for the junk's
            // successor every hop and manufacture the assembler's supporter.
            const int exp = voter.expectedUnit();
            if (exp < 0 || voter.streak() < 1) continue;
            const int L = idx.len(exp);
            const int minI = std::max(6, (int)std::ceil(p.earlyPrefix * L - 1e-9));
            if (L < minI) continue;
            const double pc = prefixNorm(idx.phonemes(exp), e.win, minI);
            if (pc <= p.costThresh) voter.onFire(e.w1, exp, pc);
        } else {
            voter.onFire(e.w1, e.unit, e.cost);
        }
    }
    return voter.emitted();
}

}  // namespace quranrecite
