// Runs the C++ engine over the conformance fixtures and writes outputs in the format
// conformance/verify.py expects, so the port can be checked:
//
//   conformance_runner <conformance_dir> <out_dir>
//   python conformance/verify.py --candidate <out_dir>
//
// For each manifest.frontend fixture: read the .wav, compute logMel, write <name>.logmel.bin.
// For each manifest.matcher fixture: read the windows, run the segmenter, write
// <name>.events.json. See conformance/spec.md "Candidate output layout".
#include <cstdio>
#include <string>

// TODO(port): a tiny WAV reader + JSON read/write (or vendor a header-only lib), then:
//   FrontEnd fe(conf+"/assets/mel_filterbank.bin", conf+"/assets/hann_window.bin");
//   for each frontend fixture -> fe.logMel(...) -> dump float32 .bin
//   Lexicon lex(conf+"/assets/ayah_phonemes.json", conf+"/assets/tokens.txt");
//   for each matcher fixture -> map phoneme strings to ids -> SlidingSegmenter -> events.json
//
// This executable is the local acceptance gate; keep it green as you fill the stubs.

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: conformance_runner <conformance_dir> <out_dir>\n");
        return 2;
    }
    std::fprintf(stderr, "scaffold: implement frontend + matcher, then emit outputs.\n");
    return 0;
}
