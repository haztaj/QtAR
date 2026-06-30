// Standalone front-end conformance check: compute log-mel for each fixture WAV and
// compare to the golden .bin. No JSON/ORT needed (golden T is inferred from file size).
//
//   g++ -std=c++17 -O2 -I ../src test_frontend.cpp ../src/frontend.cpp ../src/assets.cpp -o test_frontend
//   ./test_frontend <conformance_dir>
#include <cmath>
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

namespace fs = std::filesystem;

// Minimal RIFF reader for mono 32-bit float WAV (what the fixtures are).
static std::vector<float> readWavMonoF32(const std::string& path, int& sr) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto u32 = [&](size_t o) { uint32_t v; std::memcpy(&v, &b[o], 4); return v; };
    auto u16 = [&](size_t o) { uint16_t v; std::memcpy(&v, &b[o], 2); return v; };
    std::vector<float> out;
    int fmt = 3, bits = 32, ch = 1;
    sr = 16000;
    size_t pos = 12;  // skip "RIFF"<size>"WAVE"
    while (pos + 8 <= b.size()) {
        char id[5] = {0};
        std::memcpy(id, &b[pos], 4);
        uint32_t sz = u32(pos + 4);
        size_t body = pos + 8;
        if (std::strcmp(id, "fmt ") == 0) {
            fmt = u16(body); ch = u16(body + 2); sr = (int)u32(body + 4); bits = u16(body + 14);
        } else if (std::strcmp(id, "data") == 0) {
            if (fmt == 3 && bits == 32 && ch == 1) {
                out.resize(sz / 4);
                std::memcpy(out.data(), &b[body], sz);
            }
            break;
        }
        pos = body + sz + (sz & 1);  // chunks are word-aligned
    }
    return out;
}

int main(int argc, char** argv) {
    std::string conf = argc > 1 ? argv[1] : "conformance";
    quranrecite::FrontEnd fe(conf + "/assets/mel_filterbank.bin", conf + "/assets/hann_window.bin");

    float worst = 0.0f;
    bool ok = true;
    for (auto& e : fs::directory_iterator(conf + "/fixtures/frontend")) {
        if (e.path().extension() != ".wav") continue;
        std::string name = e.path().stem().string();
        int sr;
        auto wav = readWavMonoF32(e.path().string(), sr);
        int T;
        auto mine = fe.logMel(wav.data(), wav.size(), T);
        auto gold = quranrecite::loadF32Bin(conf + "/golden/frontend/" + name + ".logmel.bin");
        float d = 0.0f;
        if (mine.size() == gold.size())
            for (size_t i = 0; i < mine.size(); ++i) d = std::max(d, std::abs(mine[i] - gold[i]));
        else
            d = 1e9f;
        worst = std::max(worst, d);
        bool good = mine.size() == gold.size() && d <= 1e-2f;
        ok &= good;
        std::printf("%-24s T=%-5d  max_abs_diff=%.2e  %s\n", name.c_str(), T, d,
                    good ? "PASS" : "FAIL");
    }
    std::printf("\nworst max_abs_diff=%.2e (tol 1e-2)  ->  %s\n", worst, ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
