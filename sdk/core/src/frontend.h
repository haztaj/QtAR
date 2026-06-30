// Front-end DSP: 16 kHz mono PCM -> 80-dim log-mel. MUST match conformance/spec.md §Stage 1
// and training/data.py:logmel_16k exactly (validated by conformance/verify.py).
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace quranrecite {

class FrontEnd {
public:
    // Loads the EXACT mel filterbank [201,80] and Hann window [400] from the conformance
    // assets so there is no constant mismatch with the reference.
    FrontEnd(const std::string& melFilterbankPath, const std::string& hannWindowPath,
             float normRms = 0.1f);

    // 16 kHz mono wav -> log-mel, row-major [T][80] flattened. `outT` receives T.
    // Steps: RMS-normalize -> reflect-pad n_fft/2 -> framed Hann STFT (power) -> mel matmul
    //        -> log(max(., 1e-10)). See spec.md.
    std::vector<float> logMel(const float* wav16k, std::size_t n, int& outT) const;

    static constexpr int kNMels = 80;
    static constexpr int kNFft = 400;
    static constexpr int kHop = 160;
    static constexpr int kNFreqs = 201;   // n_fft/2 + 1

private:
    std::vector<float> filterbank_;  // [kNFreqs * kNMels], row-major [201][80]
    std::vector<float> window_;      // [kNFft]
    std::vector<double> cos_, sin_;  // rfft twiddles [kNFreqs * kNFft]
    float normRms_;
};

}  // namespace quranrecite
