#include "frontend.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "assets.h"

namespace quranrecite {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr float kLogFloor = 1e-10f;
}  // namespace

// Loads the EXACT mel filterbank [201,80] and Hann window [400] (conformance assets) and
// precomputes the rfft twiddle tables, so the C++ matches the Python reference constants.
FrontEnd::FrontEnd(const std::string& fbPath, const std::string& winPath, float normRms)
    : normRms_(normRms) {
    filterbank_ = loadF32Bin(fbPath, static_cast<std::size_t>(kNFreqs) * kNMels);  // [201*80]
    window_ = loadF32Bin(winPath, kNFft);                                          // [400]
    // Twiddles for rfft: bins k=0..200 over n=0..399.
    cos_.resize(static_cast<std::size_t>(kNFreqs) * kNFft);
    sin_.resize(static_cast<std::size_t>(kNFreqs) * kNFft);
    for (int k = 0; k < kNFreqs; ++k) {
        for (int n = 0; n < kNFft; ++n) {
            double ang = 2.0 * kPi * k * n / kNFft;
            cos_[static_cast<std::size_t>(k) * kNFft + n] = std::cos(ang);
            sin_[static_cast<std::size_t>(k) * kNFft + n] = std::sin(ang);
        }
    }
}

std::vector<float> FrontEnd::logMel(const float* wav, std::size_t n, int& outT) const {
    // ----- 1) RMS-normalize to normRms_ (data.py: normalize_rms) -----
    std::vector<float> x(wav, wav + n);
    double ss = 0.0;
    for (float v : x) ss += static_cast<double>(v) * v;
    double rms = std::sqrt(ss / std::max<std::size_t>(1, n));
    if (rms > 1e-6) {
        float g = static_cast<float>(normRms_ / rms);
        for (float& v : x) v = std::clamp(v * g, -1.0f, 1.0f);
    }

    // ----- 2) reflect-pad n_fft/2 each side (center=True, pad_mode=reflect) -----
    const int pad = kNFft / 2;  // 200
    const std::size_t L = n;
    std::vector<float> p(L + 2 * pad);
    for (std::size_t i = 0; i < L; ++i) p[pad + i] = x[i];
    for (int j = 0; j < pad; ++j) {
        // torch reflect: left = x[pad-j] (reflect about index 0, excluding it)
        p[j] = (L > 1) ? x[std::min<std::size_t>(pad - j, L - 1)] : x[0];
        // right = x[L-2-j] (reflect about index L-1)
        long ri = static_cast<long>(L) - 2 - j;
        p[pad + L + j] = (ri >= 0) ? x[static_cast<std::size_t>(ri)] : x[0];
    }

    // ----- 3) framed Hann STFT -> power; 4) mel matmul; 5) log -----
    const int T = 1 + static_cast<int>(L / kHop);  // torch.stft center frame count
    std::vector<float> out(static_cast<std::size_t>(T) * kNMels);
    std::vector<double> power(kNFreqs);
    std::vector<float> frame(kNFft);

    for (int t = 0; t < T; ++t) {
        const std::size_t start = static_cast<std::size_t>(t) * kHop;
        for (int i = 0; i < kNFft; ++i) frame[i] = p[start + i] * window_[i];  // windowed
        for (int k = 0; k < kNFreqs; ++k) {
            const double* c = &cos_[static_cast<std::size_t>(k) * kNFft];
            const double* s = &sin_[static_cast<std::size_t>(k) * kNFft];
            double re = 0.0, im = 0.0;
            for (int i = 0; i < kNFft; ++i) {
                re += frame[i] * c[i];
                im -= frame[i] * s[i];   // rfft: exp(-j 2pi k n / N)
            }
            power[k] = re * re + im * im;
        }
        float* orow = &out[static_cast<std::size_t>(t) * kNMels];
        for (int m = 0; m < kNMels; ++m) {
            double acc = 0.0;
            for (int k = 0; k < kNFreqs; ++k)
                acc += power[k] * filterbank_[static_cast<std::size_t>(k) * kNMels + m];
            orow[m] = std::log(std::max(static_cast<float>(acc), kLogFloor));
        }
    }
    outT = T;
    return out;
}

}  // namespace quranrecite
