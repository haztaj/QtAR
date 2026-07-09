#include "decoder.h"

#include <algorithm>
#include <cmath>
#include <numeric>

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

std::vector<std::pair<int, float>> topKAlts(const float* row, int vocab, int topk) {
    std::vector<int> idx(vocab);
    std::iota(idx.begin(), idx.end(), 0);
    const int k = std::min(topk + 1, vocab);       // +1 so it survives dropping the blank
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return row[a] > row[b]; });
    std::vector<std::pair<int, float>> out;
    for (int i = 0; i < k && (int)out.size() < topk; ++i)
        if (idx[i] != 0) {
            float p = std::round(std::exp(row[idx[i]]) * 1e4f) / 1e4f;  // 4 dp (cache parity)
            out.push_back({idx[i], p});
        }
    return out;
}

}  // namespace quranrecite
