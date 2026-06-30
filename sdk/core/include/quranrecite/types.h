// Public types for the QuranRecite SDK core.
#pragma once
#include <functional>
#include <string>

namespace quranrecite {

// Stable ayah identifier (surah:ayah). The host app owns the mushaf text/rendering.
struct AyahId {
    int surah = 0;
    int ayah = 0;
    bool operator==(const AyahId& o) const { return surah == o.surah && ayah == o.ayah; }
};

enum class EventType { Detect, Advance, Jump };

struct AyahEvent {
    EventType type;
    AyahId ayah;
    float confidence = 0.0f;   // 0..1, derived from matcher cost (1 - normalized cost)
    double timeSec = 0.0;      // stream time at detection
    AyahId from{};             // for Advance: the previous ayah
};

enum class Mode {
    Sliding,   // fixed-window content segmentation; handles continuous (no-pause) recitation
    Buffer     // legacy growing-buffer + completion (reciters who pause between ayat)
};

// All paths point at the downloaded/bundled asset bundle (see android ModelManager).
struct Config {
    std::string modelPath;        // ONNX int8 (sliding-window fixed graph)
    std::string lexiconPath;      // ayah_phonemes.json  (matcher trie source)
    std::string tokensPath;       // tokens.txt          (phoneme <-> id)
    std::string melFilterbankPath;// mel_filterbank.bin  [201,80] f32 (conformance asset)
    std::string hannWindowPath;   // hann_window.bin     [400]   f32 (conformance asset)

    Mode mode = Mode::Sliding;
    int sampleRate = 16000;

    // Sliding-window segmentation (see conformance/spec.md §Stage 2).
    float windowSec = 4.0f;
    float hopSec = 1.0f;
    float windowCost = 0.30f;     // max edit-cost for a confident window
    int jumpVotes = 2;

    // Sequential context (sticky continuation prior).
    float contextBonus = 0.22f;
    int contextWindow = 2;
    float surahBonus = 0.10f;
    float streakBonus = 0.05f;

    // Front-end (must match conformance/spec.md §Stage 1).
    float normRms = 0.1f;
};

using EventCallback = std::function<void(const AyahEvent&)>;

}  // namespace quranrecite
