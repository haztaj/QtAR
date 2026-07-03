// Stage-3 HighlightController: centralized, platform-agnostic ayah-highlight state machine.
// Port of matcher/highlight_controller.py (the SDK output contract). See conformance/spec.md
// §Stage 3. Consumes committed ayah detections + the confusable map (ambiguous_ayat.json)
// and emits render-ready HighlightState snapshots; ambiguity is deferred, never guessed.
#pragma once
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

namespace quranrecite {

struct Pending {
    std::optional<std::string> ayah;        // resolved ayah once known, else nullopt (deferred)
    std::vector<std::string> options;       // the confusable set to choose among
    std::string reason;                     // "await_successor" | "needs_choice"
};

struct HighlightState {
    std::vector<std::string> confirmed;     // settled + highlighted, in confirm order
    std::optional<Pending> pending;         // awaiting disambiguation
    std::optional<std::string> active;      // the ayah to emphasize right now

    nlohmann::json toJson() const;          // matches Python HighlightState.to_dict()
};

class HighlightController {
public:
    explicit HighlightController(const std::string& ambiguousJsonPath);

    void reset();
    HighlightState detect(const std::string& key);   // feed a committed detection
    HighlightState choose(const std::string& key);   // manually resolve a needs_choice pending
    HighlightState state() const;
    bool isAmbiguous(const std::string& key) const { return class_.count(key) > 0; }

private:
    // static confusability map (keyed by detected ayah)
    std::unordered_map<std::string, std::vector<std::string>> class_;  // key -> full class (sorted)
    std::unordered_map<std::string, std::string> pred_;   // "" == none
    std::unordered_map<std::string, std::string> succ_;   // "" == none

    // live state
    std::vector<std::string> confirmed_;
    std::optional<Pending> pending_;
    std::optional<std::string> active_;

    void confirm(const std::string& key);
    std::string predOf(const std::string& key) const;
    std::string succOf(const std::string& key) const;
    std::optional<std::string> resolveByPredecessor(const std::vector<std::string>& options,
                                                    const std::string& last) const;
    std::optional<std::string> resolveBySuccessor(const std::string& detected) const;
    bool successorsDistinct(const std::vector<std::string>& options) const;
};

}  // namespace quranrecite
