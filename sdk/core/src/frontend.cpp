#include "frontend.h"
#include <cmath>
#include <string>
// TODO(port): include the chosen FFT lib (pffft/KissFFT).

namespace quranrecite {

// TODO(port): load the [201,80] filterbank and [400] window from the .bin assets
// (float32 little-endian, row-major). See assets.{h,cpp}. Storing them avoids any
// mel-formula / window mismatch with the Python reference.
FrontEnd::FrontEnd(const std::string& /*fbPath*/, const std::string& /*winPath*/, float normRms)
    : normRms_(normRms) {
    // filterbank_ = loadF32(fbPath, kNFreqs * kNMels);
    // window_     = loadF32(winPath, kNFft);
}

std::vector<float> FrontEnd::logMel(const float* /*wav*/, std::size_t /*n*/, int& outT) const {
    // ===== IMPLEMENT per conformance/spec.md §Stage 1 =====
    // 1. RMS-normalize: rms=sqrt(mean(x^2)); if rms>1e-6 x*=normRms_/rms; clamp [-1,1].
    // 2. reflect-pad n_fft/2 = 200 samples each side; T = 1 + n/hop.
    // 3. for each frame t: y = frame .* window_ (length 400); FFT -> 201 complex bins;
    //    power[k] = re^2 + im^2.
    // 4. mel[t][m] = sum_k power[k] * filterbank_[k*80 + m].
    // 5. logmel[t][m] = log(max(mel[t][m], 1e-10)).
    // Return row-major [T][80]. Validate: conformance/verify.py max_abs_diff <= 1e-2.
    outT = 0;
    return {};
}

}  // namespace quranrecite
