// End-to-end: feed a real recitation WAV through the full C++ Detector (front-end ->
// inference -> CTC decode -> sliding segmenter) in streaming chunks and check the emitted
// ayah sequence. Validates the whole pipeline + orchestration.
//
//   g++ -std=c++17 -O2 -D_stdcall=__stdcall -I ../include -I ../src -I ../third_party \
//       -I <ort>/include test_detector.cpp ../src/*.cpp <ort>/lib/onnxruntime.dll -o test_detector
//   ./test_detector <model.onnx> <conformance_dir> <recitation.wav>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "quranrecite/detector.h"

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
            if (ch == 1 && fmt == 3 && bits == 32) {            // IEEE float
                out.resize(sz / 4); std::memcpy(out.data(), &b[body], sz);
            } else if (ch == 1 && fmt == 1 && bits == 16) {     // PCM_16 (session recorder)
                out.resize(sz / 2);
                for (size_t i = 0; i < out.size(); ++i) {
                    int16_t s; std::memcpy(&s, &b[body + i * 2], 2);
                    out[i] = s / 32768.0f;
                }
            }
            break;
        }
        pos = body + sz + (sz & 1);
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: test_detector <model.onnx> <conf_dir> <wav> [vad.onnx|--chain]\n");
        return 2;
    }
    std::string model = argv[1], conf = argv[2], wav = argv[3];

    Config cfg;
    cfg.modelPath = model;
    cfg.lexiconPath = conf + "/assets/ayah_phonemes.json";
    cfg.tokensPath = conf + "/assets/tokens.txt";
    cfg.melFilterbankPath = conf + "/assets/mel_filterbank.bin";
    cfg.hannWindowPath = conf + "/assets/hann_window.bin";
    if (argc >= 5) {
        if (std::string(argv[4]) == "--chain") {   // unit-chain decoder (waqf segments)
            cfg.mode = Mode::Chain;
            cfg.unitPhonemesPath = conf + "/assets/unit_phonemes.json";
        } else {
            cfg.vadPath = argv[4];   // optional Silero VAD (paused-recitation reset)
        }
    }

    Detector det(cfg);
    std::vector<std::string> seq;
    det.setEventCallback([&](const AyahEvent& e) {
        const char* k = e.type == EventType::Detect ? "detect" : e.type == EventType::Advance ? "advance" : "jump";
        std::string a = std::to_string(e.ayah.surah) + ":" + std::to_string(e.ayah.ayah);
        std::printf("  %-8s %s  (conf %.2f, t %.1fs)\n", k, a.c_str(), e.confidence, e.timeSec);
        seq.push_back(a);
    });
    // waqf-segment progress within the active ayah (Mode::Chain)
    std::string lastSegLine;
    det.setHighlightCallback([&](const HighlightSnapshot& s) {
        if (!s.hasActive || s.activeSegmentCount == 0) return;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "    segment %d:%d  %d/%d",
                      s.active.surah, s.active.ayah, s.activeSegment, s.activeSegmentCount);
        if (buf != lastSegLine) { lastSegLine = buf; std::printf("%s\n", buf); }
    });

    auto audio = readWavMonoF32(wav);
    std::printf("feeding %.1fs of audio in 100 ms chunks...\n", audio.size() / 16000.0);
    const std::size_t chunk = 1600;  // 100 ms @ 16 kHz
    for (std::size_t i = 0; i < audio.size(); i += chunk)
        det.feed(audio.data() + i, std::min(chunk, audio.size() - i), 16000);

    std::printf("\ndetected sequence:");
    for (auto& s : seq) std::printf(" %s", s.c_str());
    std::printf("\n");
    return 0;
}
