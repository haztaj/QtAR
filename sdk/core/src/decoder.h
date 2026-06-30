// CTC greedy decode: log-probs [T][vocab] -> phoneme token ids.
// Port of eval/evaluate.py:greedy_phonemes (argmax -> collapse repeats -> drop blank id 0).
#pragma once
#include <vector>

namespace quranrecite {

std::vector<int> ctcGreedy(const std::vector<float>& logProbs, int T, int vocab);

}  // namespace quranrecite
