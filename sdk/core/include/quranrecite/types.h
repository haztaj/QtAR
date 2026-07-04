// Public types for the QuranRecite SDK core.
#pragma once
#include <functional>
#include <string>
#include <vector>

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

// -- Highlight state (the centralized, platform-agnostic output contract) ---------------
// Mirrors matcher/highlight_controller.py. The engine emits one immutable snapshot per
// change; UIs just render it. Ambiguity is deferred (never guessed) — see spec.md §Stage 3.
enum class PendingReason { AwaitSuccessor, NeedsChoice };

struct HighlightPending {
    bool hasAyah = false;             // resolved ayah once known, else deferred
    AyahId ayah{};
    std::vector<AyahId> options;      // the confusable set to choose among
    PendingReason reason = PendingReason::AwaitSuccessor;
};

struct HighlightSnapshot {
    std::vector<AyahId> confirmed;    // settled + highlighted, in confirm order
    bool hasPending = false;
    HighlightPending pending;         // valid iff hasPending
    bool hasActive = false;           // the ayah to emphasize right now
    AyahId active{};
};

enum class Mode {
    Auto,      // sliding + stream merged — handles any ayah length (default)
    Sliding,   // fixed-window content segmentation; handles continuous short ayat
    Buffer     // legacy growing-buffer + completion (reciters who pause between ayat)
};

// All paths point at the downloaded/bundled asset bundle (see android ModelManager).
struct Config {
    std::string modelPath;        // ONNX int8 (sliding-window fixed graph)
    std::string lexiconPath;      // ayah_phonemes.json  (matcher trie source)
    std::string tokensPath;       // tokens.txt          (phoneme <-> id)
    std::string melFilterbankPath;// mel_filterbank.bin  [201,80] f32 (conformance asset)
    std::string hannWindowPath;   // hann_window.bin     [400]   f32 (conformance asset)
    std::string ambiguousPath;    // ambiguous_ayat.json (Stage-3 confusable map; optional —
                                  // empty disables deferral, every detection confirms)

    Mode mode = Mode::Auto;
    int sampleRate = 16000;

    // Sliding-window segmentation (see conformance/spec.md §Stage 2).
    float windowSec = 4.0f;
    float hopSec = 1.0f;
    float windowCost = 0.30f;     // max edit-cost for a confident window
    int jumpVotes = 2;

    // Stream matcher (prefix-anchored; port of demo/streaming.py StreamDetector).
    int streamPersistence = 3;    // hops the top-1 must lead to commit
    int streamJumpPersistence = 5;// higher bar for a non-continuation jump
    float streamMinProgress = 0.2f;
    float commitCostMax = 0.75f;  // loose garbage ceiling (rank persistence is the real gate)
    float lenTol = 0.6f;          // length prune: skip ayat shorter than lenTol * input
    float keepLongSec = 11.0f;    // buffer tail on a long-ayah leader change
    float keepDoneSec = 1.5f;     // buffer tail after a finished ayah
    float doneProgress = 0.85f;   // progress at commit meaning "ayah finished"

    // Sequential context (sticky continuation prior).
    float contextBonus = 0.22f;
    int contextWindow = 2;
    float surahBonus = 0.10f;
    float streakBonus = 0.05f;

    // Front-end (must match conformance/spec.md §Stage 1).
    float normRms = 0.1f;
};

using EventCallback = std::function<void(const AyahEvent&)>;
using HighlightCallback = std::function<void(const HighlightSnapshot&)>;

}  // namespace quranrecite
