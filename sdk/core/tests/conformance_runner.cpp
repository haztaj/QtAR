// Runs the C++ engine over the conformance fixtures and writes outputs in the format
// conformance/verify.py expects:
//
//   conformance_runner <conformance_dir> <out_dir>
//   python conformance/verify.py --candidate <out_dir>
//
// Front-end: WAV -> logMel -> <name>.logmel.bin (float32 LE).
// Matcher:   windows -> SlidingSegmenter -> <name>.events.json.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "assets.h"
#include "frontend.h"
#include "highlight.h"
#include "matcher.h"
#include "nlohmann/json.hpp"
#include "quranrecite/types.h"
#include "segmenter.h"

using namespace quranrecite;
namespace fs = std::filesystem;
using nlohmann::json;

static std::vector<float> readWavMonoF32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto u32 = [&](size_t o) { uint32_t v; std::memcpy(&v, &b[o], 4); return v; };
    auto u16 = [&](size_t o) { uint16_t v; std::memcpy(&v, &b[o], 2); return v; };
    std::vector<float> out;
    int fmt = 3, bits = 32, ch = 1;
    size_t pos = 12;
    while (pos + 8 <= b.size()) {
        char id[5] = {0};
        std::memcpy(id, &b[pos], 4);
        uint32_t sz = u32(pos + 4);
        size_t body = pos + 8;
        if (!std::strcmp(id, "fmt ")) { fmt = u16(body); ch = u16(body + 2); bits = u16(body + 14); }
        else if (!std::strcmp(id, "data")) {
            if (fmt == 3 && bits == 32 && ch == 1) { out.resize(sz / 4); std::memcpy(out.data(), &b[body], sz); }
            break;
        }
        pos = body + sz + (sz & 1);
    }
    return out;
}

static std::string ayahStr(const AyahId& a) { return std::to_string(a.surah) + ":" + std::to_string(a.ayah); }
static std::string eventName(EventType t) {
    return t == EventType::Detect ? "detect" : t == EventType::Advance ? "advance" : "jump";
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: conformance_runner <conf_dir> <out_dir>\n"); return 2; }
    std::string conf = argv[1], out = argv[2];
    fs::create_directories(out);
    auto man = json::parse(readFile(conf + "/manifest.json"));

    // --- front-end ---
    FrontEnd fe(conf + "/assets/mel_filterbank.bin", conf + "/assets/hann_window.bin");
    for (auto& fx : man["frontend"]) {
        auto wav = readWavMonoF32(conf + "/" + fx["wav"].get<std::string>());
        int T;
        auto lm = fe.logMel(wav.data(), wav.size(), T);
        std::string name = fx["name"].get<std::string>();
        std::ofstream o(out + "/" + name + ".logmel.bin", std::ios::binary);
        o.write(reinterpret_cast<const char*>(lm.data()), lm.size() * sizeof(float));
    }

    // --- matcher / segmenter ---
    Lexicon lex(conf + "/assets/ayah_phonemes.json", conf + "/assets/tokens.txt");
    for (auto& fx : man["matcher"]) {
        std::string name = fx["name"].get<std::string>();
        auto spec = json::parse(readFile(conf + "/" + fx["windows"].get<std::string>()));
        auto ctxj = spec["config"]["context"];
        Config cfg;
        cfg.contextBonus = ctxj["bonus"]; cfg.contextWindow = ctxj["window"];
        cfg.surahBonus = ctxj["surah_bonus"]; cfg.streakBonus = ctxj["streak_bonus"];
        cfg.windowCost = spec["config"]["max_cost"];
        SequentialContext ctx(lex, cfg);
        SlidingSegmenter seg(lex, ctx, cfg);
        json events = json::array();
        int i = 0;
        for (auto& win : spec["windows"]) {
            std::vector<int> ids;
            for (auto& tok : win) ids.push_back(lex.tokenId(tok.get<std::string>()));
            if (auto ev = seg.process(ids, (double)i))
                events.push_back({{"event", eventName(ev->type)}, {"ayah", ayahStr(ev->ayah)},
                                  {"t", ev->timeSec}, {"cost", ev->confidence}});
            ++i;
        }
        std::ofstream o(out + "/" + name + ".events.json");
        o << json{{"events", events}}.dump(1);
    }

    // --- highlight (Stage-3 controller) ---
    if (man.contains("highlight")) {
        HighlightController hc(conf + "/assets/ambiguous_ayat.json");
        for (auto& fx : man["highlight"]) {
            std::string name = fx["name"].get<std::string>();
            auto spec = json::parse(readFile(conf + "/" + fx["steps"].get<std::string>()));
            hc.reset();
            json states = json::array();
            for (auto& step : spec["steps"]) {
                HighlightState s = step.contains("detect")
                                       ? hc.detect(step["detect"].get<std::string>())
                                       : hc.choose(step["choose"].get<std::string>());
                states.push_back(s.toJson());
            }
            std::ofstream o(out + "/" + name + ".states.json");
            o << json{{"states", states}}.dump(1);
        }
    }

    std::printf("wrote conformance outputs -> %s\n", out.c_str());
    return 0;
}
