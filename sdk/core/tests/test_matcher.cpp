// Matcher/segmenter conformance: run each matcher fixture's window phonemes through the
// C++ SlidingSegmenter and compare the (event, ayah) sequence to golden. Exact match.
//
//   g++ -std=c++17 -O2 -I ../src -I ../third_party test_matcher.cpp \
//       ../src/matcher.cpp ../src/segmenter.cpp ../src/assets.cpp -o test_matcher
//   ./test_matcher <conformance_dir>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "assets.h"
#include "matcher.h"
#include "nlohmann/json.hpp"
#include "quranrecite/types.h"
#include "segmenter.h"

using namespace quranrecite;
namespace fs = std::filesystem;
using nlohmann::json;

static std::string eventName(EventType t) {
    switch (t) {
        case EventType::Detect: return "detect";
        case EventType::Advance: return "advance";
        case EventType::Jump: return "jump";
    }
    return "?";
}
static std::string ayahStr(const AyahId& a) { return std::to_string(a.surah) + ":" + std::to_string(a.ayah); }

int main(int argc, char** argv) {
    std::string conf = argc > 1 ? argv[1] : "conformance";
    Lexicon lex(conf + "/assets/ayah_phonemes.json", conf + "/assets/tokens.txt");
    std::printf("lexicon: %zu ayat\n", lex.size());

    bool ok = true;
    for (auto& e : fs::directory_iterator(conf + "/fixtures/matcher")) {
        if (e.path().extension() != ".json") continue;
        std::string name = e.path().stem().string();
        auto fx = json::parse(readFile(e.path().string()));
        auto cfgj = fx["config"];

        Config cfg;
        auto ctxj = cfgj["context"];
        cfg.contextBonus = ctxj["bonus"];
        cfg.contextWindow = ctxj["window"];
        cfg.surahBonus = ctxj["surah_bonus"];
        cfg.streakBonus = ctxj["streak_bonus"];
        cfg.windowCost = cfgj["max_cost"];

        SequentialContext ctx(lex, cfg);
        SlidingSegmenter seg(lex, ctx, cfg);

        std::vector<std::pair<std::string, std::string>> got;  // (event, ayah)
        int i = 0;
        for (auto& win : fx["windows"]) {
            std::vector<int> ids;
            for (auto& tok : win) ids.push_back(lex.tokenId(tok.get<std::string>()));
            if (auto ev = seg.process(ids, (double)i))
                got.emplace_back(eventName(ev->type), ayahStr(ev->ayah));
            ++i;
        }

        auto gold = json::parse(readFile(conf + "/golden/matcher/" + name + ".events.json"))["events"];
        std::vector<std::pair<std::string, std::string>> want;
        for (auto& ev : gold) want.emplace_back(ev["event"].get<std::string>(), ev["ayah"].get<std::string>());

        bool good = got == want;
        ok &= good;
        std::printf("%-22s %s  got=[", name.c_str(), good ? "PASS" : "FAIL");
        for (auto& g : got) std::printf("%s ", g.second.c_str());
        std::printf("] want=[");
        for (auto& w : want) std::printf("%s ", w.second.c_str());
        std::printf("]\n");
    }
    std::printf("\n%s\n", ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
