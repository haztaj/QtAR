// Feed a real recitation WAV through the C++ SileroVAD in fixed 512-sample chunks and print the
// speech-END events (ayah boundaries). Validates parity with the Python VADIterator: a paused
// N-ayah recording should emit ~N speech-END events at the pauses.
//
//   ./test_vad <silero_vad.onnx> <recitation.wav>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vad.h"

using namespace quranrecite;

static std::vector<float> readWavMonoF32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto u32 = [&](size_t o) { uint32_t v; std::memcpy(&v, &b[o], 4); return v; };
    auto u16 = [&](size_t o) { uint16_t v; std::memcpy(&v, &b[o], 2); return v; };
    std::vector<float> out;
    int fmt = 3, bits = 32, ch = 1;
    size_t pos = 12;
    while (pos + 8 <= b.size()) {
        char id[5] = {0}; std::memcpy(id, &b[pos], 4);
        uint32_t sz = u32(pos + 4); size_t body = pos + 8;
        if (!std::strcmp(id, "fmt ")) { fmt = u16(body); ch = u16(body + 2); bits = u16(body + 14); }
        else if (!std::strcmp(id, "data")) {
            if (ch == 1 && fmt == 3 && bits == 32) { out.resize(sz / 4); std::memcpy(out.data(), &b[body], sz); }
            else if (ch == 1 && fmt == 1 && bits == 16) {
                out.resize(sz / 2);
                for (size_t i = 0; i < out.size(); ++i) { int16_t s; std::memcpy(&s, &b[body + i * 2], 2); out[i] = s / 32768.0f; }
            }
            break;
        }
        pos = body + sz + (sz & 1);
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: test_vad <silero_vad.onnx> <wav>\n"); return 2; }
    SileroVAD vad(argv[1], 16000, 0.5f, 0.5f);
    auto audio = readWavMonoF32(argv[2]);
    const int VC = SileroVAD::chunkSize();
    int ends = 0;
    std::printf("feeding %.1fs of audio in %d-sample chunks...\n", audio.size() / 16000.0, VC);
    for (std::size_t i = 0; i + VC <= audio.size(); i += VC) {
        if (vad.feed(audio.data() + i, VC)) {
            std::printf("  speech-END at %.2fs\n", (double)(i + VC) / 16000.0);
            ++ends;
        }
    }
    std::printf("total speech-END events: %d\n", ends);
    return 0;
}
