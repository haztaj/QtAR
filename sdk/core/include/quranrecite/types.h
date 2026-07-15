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
    bool hasActive = false;           // the ayah just detected (lighter highlight)
    AyahId active{};
    bool hasUpNext = false;           // the predicted next ayah, shown only once `active` is
    AyahId upNext{};                  //   near-complete (darker highlight); same surah only
    // Waqf-segment progress within `active` (Mode::Chain only): which sub-ayah unit is being
    // recited. activeSegmentCount == 0 means no segment info (non-Chain mode, or no active ayah);
    // >= 1 means known — 1 for an unsegmented ayah, N for one split into N waqf segments.
    // activeSegment is the current segment, 1-based in [1, activeSegmentCount], else 0.
    int activeSegment = 0;
    int activeSegmentCount = 0;
};

enum class Mode {
    Auto,      // sliding + stream merged — handles any ayah length (default)
    Sliding,   // fixed-window content segmentation; handles continuous short ayat
    Buffer,    // legacy growing-buffer + completion (reciters who pause between ayat)
    Chain      // unit-chain decoder over waqf segments (the research winning design;
               //   requires unitPhonemesPath + a model window >= chainMaxWindowSec)
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

    // Unit-chain decoder (Mode::Chain): sliding matched-filter windows over waqf segments.
    // The phoneme stream is decoded once per hop from the rolling buffer (largest chain
    // window), then all scale windows are sliced from that decode by time.
    std::string unitPhonemesPath;   // unit_phonemes.json (waqf segments + unsegmented ayat)
    // True streaming acoustics (Mode::Chain only). If BOTH are set, stepChain decodes only the
    // NEW audio each hop via StreamingModel (stream_conv.onnx + stream_encoder.int8.onnx) and
    // maintains an incremental phoneme stream, instead of re-decoding the whole rolling window
    // with `modelPath` — the battery/latency path (see export/streaming-export-plan.md). Empty =>
    // windowed re-decode (default; off until the on-device acceptance test passes).
    std::string streamConvPath;
    std::string streamEncoderPath;
    float chainWindowSec = 10.0f;   // base window (scales 0.2/0.7/1.0/1.5/2.2 x this)
    float chainHopSec = 1.5f;
    float chainCost = 0.30f;        // window fire threshold; 0.30 = clean-decode reference,
                                    // ~0.45 for consumer phone mics (~30% PER decodes)
    int chainProvVotes = 2;         // corroboration required to surface a PROVISIONAL highlight:
                                    // the emitted unit's parent surah must appear in >= this many
                                    // recent emits. chainCost governs commit/recall, but at the
                                    // full-Quran 6x index a fire <= chainCost is often a wrong
                                    // prefix-collision unit whose cost OVERLAPS the correct one, so
                                    // a cost gate can't separate them; the correct surah recurs
                                    // while wrong jumps are one-off. <= 1 disables the gate.
    float chainStartAtAyahSec = 12.0f;  // ramp window for the cold-start ayah-begin penalty (below).
                                    // 0 disables the penalty entirely.
    float chainStartAyahMult = 0.5f;    // DECAYING cold-start penalty on mid-ayah (#02+) matches: a
                                    // reciter starts at an ayah's beginning, so a mid-ayah match early
                                    // is a wrong prefix collision. Rather than hard-block it (which lost
                                    // long opening ayat whose first segment decodes poorly), a mid-ayah
                                    // unit must clear a TIGHTER fire bar = chainCost * m(t), where m
                                    // ramps from chainStartAyahMult (t=0) up to 1.0 at chainStartAtAyahSec.
                                    // So mid-ayah needs a ~2x better cost at t=0, easing to normal — the
                                    // correct long ayah still locks in a bit later. 1.0 = no penalty.
    float chainStrongStartCost = 0.25f; // at cold start, a strong ayah-BEGIN match (cost <= this)
                                    // commits with a single vote so a clean opening ayah locks in
                                    // fast (before a quiet/hard decode degrades). Ayah-start only,
                                    // above kStrongCost (0.15). <= 0 disables (2-vote default).
    int chainVotesNext = 1;
    int chainVotesJump = 2;
    float chainEarlyPrefix = 0.5f;  // >0: fire the EXPECTED unit once this fraction of its
                                    // prefix matches the decode tail (early detection);
                                    // 0 disables (commit at unit end only)
    float chainSubMin = 1.0f;       // Phase-2 posterior-aware scoring: substitution cost
                                    // floor. 1.0 = hard 0/1 distance (off); ~0 softens
                                    // mismatches the model nearly picked (a ~+1.7 aligned-hit
                                    // win in the ~30% PER phone regime, free on clean audio)

    // Silero VAD (Auto mode): if vadPath is set, feed speech-END events reset the buffers +
    // matcher so paused ayah-by-ayah recitation segments cleanly. Empty -> no VAD (energy gate
    // only). Mirrors demo/live_detect.py's VADIterator(threshold=0.5, min_silence=500 ms).
    std::string vadPath;
    float vadThreshold = 0.5f;
    float vadMinSilenceSec = 0.5f;
    // EXPERIMENTAL (Mode::Chain): on a VAD speech-END, drop the buffered ayah's audio/phonemes so
    // the next ayah decodes in a FOCUSED window (the 22 s rolling window otherwise crowds out short
    // tail units after ayah-by-ayah pauses), while KEEPING the voter/assembler chain context. Off
    // by default — Chain is normally pause-tolerant/time-gated. Needs vadPath set. See
    // export/../research measurement (2026-07-11).
    bool chainVadReset = false;
    // Targeted gate for chainVadReset: only reset if the pause follows the last unit commit within
    // this many seconds. A short ayah commits then pauses (small gap -> reset, focused window); a
    // long ayah's mid-breath comes many seconds after the last commit (large gap -> NO reset, so the
    // ayah's prefix survives). Measured sweet spot 4.0 (research/audio_bench.py): recovers real-phone
    // crowding with zero long-ayah regression (1e9 = ungated/blunt: reset on every pause, which guts
    // long ayat — see research/CLAUDE.md).
    float chainResetMaxGap = 4.0f;
    // EXPERIMENTAL (Mode::Chain, WINDOWED only): on every voter emission, trim the rolling audio
    // buffer to (emission time - this many seconds). De-crowds CONTINUOUS recitation, which never
    // pauses long enough for the VAD reset: without a boundary the growing window's decode
    // COLLAPSES (deletes whole short ayat) once several ayat accumulate — measured on live 114
    // takes (2026-07-11). The kept tail preserves the in-progress next ayah's prefix (the v1
    // commit-and-reset cascade guard); votes/streak gating make junk-emission trims rare.
    // 0 = off. Streaming ignores it (incremental decode never re-windows).
    float chainEmitTrimKeep = 0.0f;
    // EXPERIMENTAL v13 (Mode::Chain, WINDOWED only): fresh-context suffix decode. Each hop,
    // ALSO decode the rolling buffer's last chainSuffixSec seconds as a STANDALONE input
    // through a right-sized graph (chainSuffixModelPath — must be the SAME weights as
    // modelPath, e.g. a --fixed-frames 516 export) and match over it. The Emformer's
    // left-context memory DELETES repeated phrases in continuous audio (repetitive short
    // surahs recited without pauses decode to a fraction of their phonemes — see
    // research/CLAUDE.md "Repetition suppression", 2026-07-11); a fresh-context decode
    // sidesteps the suppression unconditionally, where the VAD reset needs a pause and the
    // commit/emission gates need prior progress. 0 / empty = off.
    float chainSuffixSec = 0.0f;
    std::string chainSuffixModelPath;
};

using EventCallback = std::function<void(const AyahEvent&)>;
using HighlightCallback = std::function<void(const HighlightSnapshot&)>;

}  // namespace quranrecite
