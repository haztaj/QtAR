// CTC greedy decode: log-probs [T][vocab] -> phoneme token ids.
// Port of eval/evaluate.py:greedy_phonemes (argmax -> collapse repeats -> drop blank id 0).
#pragma once
#include <utility>
#include <vector>

namespace quranrecite {

std::vector<int> ctcGreedy(const std::vector<float>& logProbs, int T, int vocab);

// Top-k non-blank posterior alternatives (tokenId, prob) at one frame's log-prob row
// (prob = exp(logrow), rounded to 4 dp to match the Python cache). Phase-0 enabler for
// posterior-aware scoring: alts[0] is the greedy phoneme + its probability.
std::vector<std::pair<int, float>> topKAlts(const float* logRow, int vocab, int topk = 3);

}  // namespace quranrecite
