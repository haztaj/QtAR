// Streaming conformance: feed each fixture's golden log-mel through the C++ StreamingModel
// (incremental Conv2dSubsampling cache + stateful Emformer step + CTC collapse) in 20-frame
// chunks, and compare the phoneme-id sequence to the golden produced by the Python
// StreamingRuntime (fp32). Exact match (same graphs => identical decode). Pass the SAME fp32
// graphs that generate.py exported into conformance/assets/ (int8 argmax can flip on a cross-ORT
// tie — on-device int8 is validated on the target ORT, like test_inference).
//
//   test_streaming <conf_dir> <stream_conv.onnx> <stream_encoder.onnx>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "assets.h"
#include "streaming.h"

using namespace quranrecite;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: test_streaming <conf_dir> <stream_conv.onnx> <stream_encoder.onnx>\n");
        return 2;
    }
    std::string conf = argv[1], conv = argv[2], enc = argv[3];
    StreamingModel model(conv, enc);

    bool ok = true;
    for (auto& e : fs::directory_iterator(conf + "/golden/streaming")) {
        if (e.path().extension() != ".txt") continue;
        std::string name = e.path().stem().string();          // <name>.phonemes
        std::string fxName = name.substr(0, name.rfind(".phonemes"));

        auto lm = loadF32Bin(conf + "/golden/frontend/" + fxName + ".logmel.bin");
        const int T = static_cast<int>(lm.size() / 80);
        model.reset();
        std::vector<int> ids;
        for (int i = 0; i < T; i += 20) {                     // same 20-frame chunking as the golden
            const int n = std::min(20, T - i);
            for (auto& em : model.feed(lm.data() + (std::size_t)i * 80, n)) ids.push_back(em.id);
        }

        std::vector<int> want;
        std::istringstream ss(readFile(e.path().string()));
        for (int x; ss >> x;) want.push_back(x);

        bool good = ids == want;
        ok &= good;
        std::printf("%-24s phonemes=%-3zu  %s\n", fxName.c_str(), ids.size(), good ? "PASS" : "FAIL");
        if (!good) {
            std::printf("   got :"); for (int x : ids) std::printf(" %d", x); std::printf("\n");
            std::printf("   want:"); for (int x : want) std::printf(" %d", x); std::printf("\n");
        }
    }
    std::printf("\n%s\n", ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
