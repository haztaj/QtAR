#include "decoder.h"

namespace quranrecite {

// argmax per frame, collapse consecutive duplicates, drop blank (id 0).
std::vector<int> ctcGreedy(const std::vector<float>& logProbs, int T, int vocab) {
    std::vector<int> out;
    int prev = -1;
    for (int t = 0; t < T; ++t) {
        const float* row = logProbs.data() + static_cast<std::size_t>(t) * vocab;
        int best = 0;
        for (int v = 1; v < vocab; ++v)
            if (row[v] > row[best]) best = v;
        if (best != prev && best != 0) out.push_back(best);
        prev = best;
    }
    return out;
}

}  // namespace quranrecite
