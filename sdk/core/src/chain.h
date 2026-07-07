// Unit-chain decoder — C++ port of the research "winning design" (research/chain_sliding.py):
// multi-scale matched-filter sliding windows over a decoded phoneme stream + 3-gram
// retrieval + infix edit scoring + blended selection -> successor votes + twin
// substitution -> 2-deep deferral assembly. Units are waqf segments ("S:A#NN") plus
// unsegmented ayat ("S:A"); the ayah is a derived (parent) label.
// Conformance-pinned byte-identical to the Python reference: conformance/golden/chain/.
#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace quranrecite {

// Matched filter bank: each window scale only fires refs of its own length class
// (tight gate 0.5n..1.3n in windowBest). Multiplied by ChainParams::windowSec.
inline constexpr double kChainScales[] = {0.2, 0.7, 1.0, 1.5, 2.2};

// Loaded unit lexicon (unit_phonemes.json: "S:A" / "S:A#NN" -> "ph ph ..") + the 3-gram
// inverted index. Unit order is canonical (surah, ayah, segment) — the deterministic
// tie-break order the reference uses.
class UnitIndex {
public:
    UnitIndex(const std::string& unitPhonemesJson, const std::string& tokensTxt);

    std::size_t size() const { return keys_.size(); }
    const std::string& key(int i) const { return keys_[i]; }
    const std::vector<int>& phonemes(int i) const { return phon_[i]; }
    int len(int i) const { return (int)phon_[i].size(); }
    int unitOf(const std::string& key) const;         // -1 if unknown
    int tokenId(const std::string& tok) const;

    // Cross-ayah successor: within an ayah -> next segment; last unit -> the next
    // ayah's first unit. -1 at corpus edges. (make_succ_full in the reference.)
    int succFull(int unit) const { return succ_[unit]; }

    // Candidate units sharing the 3-gram, ascending canonical order (deterministic).
    const std::vector<int>* gramUnits(int a, int b, int c) const;

    int parentOf(int unit) const { return parent_[unit]; }      // parent index (dense id)
    int segIdxOf(int unit) const { return segIdx_[unit]; }      // 0 for whole-ayah units
    const std::string& parentKey(int unit) const;
    // First unit of an ayah ("S:A" -> "S:A#01" if segmented else "S:A"); -1 if absent.
    int firstUnitOf(const std::string& parentKey) const;

private:
    std::vector<std::string> keys_;                   // canonical (s, a, seg) order
    std::vector<std::vector<int>> phon_;
    std::vector<int> succ_, parent_, segIdx_;
    std::vector<std::string> parentKeys_;             // per dense parent id
    std::unordered_map<std::string, int> parentFirst_;// parent key -> first unit index
    std::unordered_map<std::string, int> keyIdx_;
    std::unordered_map<std::string, int> tok2id_;
    std::unordered_map<long long, std::vector<int>> grams_;  // packed 3-gram -> units
};

struct ChainParams {
    double windowSec = 10.0;
    double hopSec = 1.5;
    double costThresh = 0.30;   // window fire threshold (FIRE_COST)
    int votesNext = 1;
    int votesJump = 2;
};

struct UnitEmission {
    int unit;
    double timeSec;             // committing window end (w1)
};

// Stateful vote machine (the decode_sliding emission loop). Feed fires in time order.
class ChainVoter {
public:
    ChainVoter(const UnitIndex& idx, const ChainParams& p) : idx_(idx), p_(p) {}
    void reset();
    // One window fire -> optional emission.
    std::optional<UnitEmission> onFire(double w1, int unit, double cost);
    const std::vector<UnitEmission>& emitted() const { return emitted_; }

private:
    const UnitIndex& idx_;
    ChainParams p_;
    std::vector<UnitEmission> emitted_;
    int expected_ = -1;
    int pending_ = -1;
    int votes_ = 0;
    double consumed_ = -1e9;
};

// Streaming 2-deep deferral assembly (assemble() in the reference). push() returns the
// units confirmed by this emission (0..2); flush() returns the chainable tail at
// end-of-stream.
class ChainAssembler {
public:
    explicit ChainAssembler(const UnitIndex& idx) : idx_(idx) {}
    void reset();
    std::vector<int> push(int unit);
    std::vector<int> flush();
    const std::vector<int>& confirmed() const { return confirmed_; }

private:
    bool supports(int p, int u) const;
    const UnitIndex& idx_;
    std::vector<int> confirmed_;
    std::vector<int> pending_;                        // oldest first, len <= 2
};

// Best (unit, cost) for one window of phoneme ids: 3-gram shortlist (raw-count top-60
// UNION length-normalized top-20 over the full counter) -> tight length gate
// (0.5n..1.3n) -> infix edit-norm -> blended selection (cost - 0.15*coverage).
// Returns {-1, bigCost} when nothing retrieves.
std::pair<int, double> windowBest(const std::vector<int>& win, const UnitIndex& idx);

// Offline decode of a full phoneme stream (per-phoneme times, seconds): enumerate all
// multi-scale windows, sort fires like the reference ((w1, key, cost) tuple order),
// run the vote machine. The conformance entry point.
std::vector<UnitEmission> decodeStream(const std::vector<int>& phonemes,
                                       const std::vector<double>& times,
                                       const UnitIndex& idx, const ChainParams& p);

}  // namespace quranrecite
