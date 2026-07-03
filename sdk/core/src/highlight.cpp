#include "highlight.h"

#include <algorithm>
#include <fstream>
#include <set>

namespace quranrecite {

using nlohmann::json;

namespace {
std::pair<int, int> keyOrder(const std::string& sa) {
    auto c = sa.find(':');
    return {std::stoi(sa.substr(0, c)), std::stoi(sa.substr(c + 1))};
}
// json null / missing -> "", else the string value.
std::string strOrEmpty(const json& j, const char* field) {
    auto it = j.find(field);
    if (it == j.end() || it->is_null()) return "";
    return it->get<std::string>();
}
}  // namespace

HighlightController::HighlightController(const std::string& ambiguousJsonPath) {
    std::ifstream f(ambiguousJsonPath);
    json data = json::parse(f);
    for (auto& [sa, info] : data["ambiguous"].items()) {
        // members = {sa} U confusable_with, deduped and sorted by (surah,ayah).
        std::vector<std::string> members;
        std::set<std::pair<int, int>> seen;
        auto push = [&](const std::string& k) {
            if (seen.insert(keyOrder(k)).second) members.push_back(k);
        };
        push(sa);
        for (auto& o : info["confusable_with"]) push(o.get<std::string>());
        std::sort(members.begin(), members.end(),
                  [](const std::string& a, const std::string& b) { return keyOrder(a) < keyOrder(b); });
        class_[sa] = members;
        pred_[sa] = strOrEmpty(info, "predecessor");
        succ_[sa] = strOrEmpty(info, "successor");
    }
    reset();
}

void HighlightController::reset() {
    confirmed_.clear();
    pending_.reset();
    active_.reset();
}

HighlightState HighlightController::state() const {
    HighlightState s;
    s.confirmed = confirmed_;
    s.pending = pending_;
    s.active = active_;
    return s;
}

std::string HighlightController::predOf(const std::string& key) const {
    auto it = pred_.find(key);
    return it == pred_.end() ? "" : it->second;
}
std::string HighlightController::succOf(const std::string& key) const {
    auto it = succ_.find(key);
    return it == succ_.end() ? "" : it->second;
}

void HighlightController::confirm(const std::string& key) {
    pending_.reset();
    confirmed_.push_back(key);
    active_ = key;
}

std::optional<std::string> HighlightController::resolveByPredecessor(
    const std::vector<std::string>& options, const std::string& last) const {
    if (last.empty()) return std::nullopt;
    std::string hit;
    int n = 0;
    for (auto& m : options)
        if (predOf(m) == last) { hit = m; ++n; }
    if (n == 1) return hit;
    return std::nullopt;
}

std::optional<std::string> HighlightController::resolveBySuccessor(const std::string& detected) const {
    if (!pending_) return std::nullopt;
    std::string hit;
    int n = 0;
    for (auto& m : pending_->options)
        if (succOf(m) == detected) { hit = m; ++n; }
    if (n == 1) return hit;
    return std::nullopt;
}

bool HighlightController::successorsDistinct(const std::vector<std::string>& options) const {
    std::set<std::string> seen;
    for (auto& m : options) {
        std::string s = succOf(m);
        if (s.empty()) return false;
        if (!seen.insert(s).second) return false;   // duplicate successor
    }
    return true;
}

HighlightState HighlightController::detect(const std::string& key) {
    // (1) a pending await_successor resolves the moment its successor arrives.
    if (pending_ && pending_->reason == "await_successor") {
        auto resolved = resolveBySuccessor(key);
        if (resolved) {
            confirm(*resolved);
        } else {
            pending_ = Pending{std::nullopt, pending_->options, "needs_choice"};
        }
    }

    // (2) handle the newly detected ayah.
    if (!isAmbiguous(key)) {
        confirm(key);
        return state();
    }

    const auto& options = class_.at(key);
    std::string last = confirmed_.empty() ? "" : confirmed_.back();
    auto pinned = resolveByPredecessor(options, last);
    if (pinned) {
        confirm(*pinned);
    } else if (successorsDistinct(options)) {
        pending_ = Pending{std::nullopt, options, "await_successor"};
    } else {
        pending_ = Pending{std::nullopt, options, "needs_choice"};
    }
    return state();
}

HighlightState HighlightController::choose(const std::string& key) {
    if (!pending_) return state();
    auto& opts = pending_->options;
    if (std::find(opts.begin(), opts.end(), key) == opts.end()) return state();
    confirm(key);
    return state();
}

json HighlightState::toJson() const {
    json j;
    j["confirmed"] = confirmed;
    if (pending) {
        json p;
        p["ayah"] = pending->ayah ? json(*pending->ayah) : json(nullptr);
        p["options"] = pending->options;
        p["reason"] = pending->reason;
        j["pending"] = p;
    } else {
        j["pending"] = nullptr;
    }
    j["active"] = active ? json(*active) : json(nullptr);
    return j;
}

}  // namespace quranrecite
