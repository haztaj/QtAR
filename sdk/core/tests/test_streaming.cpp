// Parity: the C++ StreamingModel must reproduce the Python StreamingRuntime phoneme ids on the
// same log-mel, fed in the same chunks.
//   test_streaming <stream_conv.onnx> <stream_encoder.int8.onnx> <lm.bin> <T> <ref.txt>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "streaming.h"

using namespace quranrecite;

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: test_streaming <conv.onnx> <enc.onnx> <lm.bin> <T> <ref.txt>\n");
        return 2;
    }
    const int T = std::stoi(argv[4]);
    std::vector<float> lm((std::size_t)T * 80);
    std::ifstream(argv[3], std::ios::binary).read(reinterpret_cast<char*>(lm.data()),
                                                  lm.size() * sizeof(float));
    std::vector<int> ref;
    { std::ifstream f(argv[5]); int x; while (f >> x) ref.push_back(x); }

    StreamingModel m(argv[1], argv[2]);
    m.reset();
    std::vector<int> got;
    for (int i = 0; i < T; i += 20) {           // same 20-frame chunking as the reference
        const int n = std::min(20, T - i);
        auto ids = m.feed(lm.data() + (std::size_t)i * 80, n);
        got.insert(got.end(), ids.begin(), ids.end());
    }

    bool ok = got == ref;
    std::printf("ref %zu phonemes, C++ %zu phonemes -> %s\n", ref.size(), got.size(),
                ok ? "MATCH" : "MISMATCH");
    if (!ok) {
        std::printf("ref:");
        for (int x : ref) std::printf(" %d", x);
        std::printf("\ncpp:");
        for (int x : got) std::printf(" %d", x);
        std::printf("\n");
    }
    return ok ? 0 : 1;
}
