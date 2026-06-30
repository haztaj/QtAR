// Inference conformance: feed each fixture's golden log-mel through the C++ ONNX Runtime
// session + CTC greedy decode, and compare the phoneme-id sequence to the golden produced
// by the Python int8 ONNX. Exact match (same engine + model => identical argmax).
//
//   g++ -std=c++17 -O2 -I ../include -I ../src -I <ort>/include test_inference.cpp \
//       ../src/inference.cpp ../src/decoder.cpp ../src/assets.cpp <ort>/lib/onnxruntime.dll -o test_inference
//   ./test_inference <conformance_dir> <model.onnx>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "assets.h"
#include "decoder.h"
#include "inference.h"

using namespace quranrecite;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: test_inference <conf_dir> <model.onnx>\n"); return 2; }
    std::string conf = argv[1], modelPath = argv[2];
    AcousticModel model(modelPath);

    bool ok = true;
    for (auto& e : fs::directory_iterator(conf + "/golden/inference")) {
        if (e.path().extension() != ".txt") continue;
        std::string name = e.path().stem().string();  // e.g. studio_ikhlas1_short.phonemes -> stem strips .txt
        // file is <name>.phonemes.txt; recover the fixture name
        std::string fxName = name.substr(0, name.rfind(".phonemes"));

        auto lm = loadF32Bin(conf + "/golden/frontend/" + fxName + ".logmel.bin");
        int T = static_cast<int>(lm.size() / 80);
        int outT, V;
        auto lp = model.run(lm, T, outT, V);
        auto ids = ctcGreedy(lp, outT, V);

        std::vector<int> want;
        std::istringstream ss(readFile(e.path().string()));
        for (int x; ss >> x;) want.push_back(x);

        bool good = ids == want;
        ok &= good;
        std::printf("%-24s phonemes=%-2zu  %s\n", fxName.c_str(), ids.size(), good ? "PASS" : "FAIL");
        if (!good) {
            std::printf("   got :"); for (int x : ids) std::printf(" %d", x); std::printf("\n");
            std::printf("   want:"); for (int x : want) std::printf(" %d", x); std::printf("\n");
        }
    }
    std::printf("\n%s\n", ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
