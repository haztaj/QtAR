// Small asset loaders (float32 LE .bin, text). See conformance/spec.md "File formats".
#pragma once
#include <string>
#include <vector>

namespace quranrecite {

// Read a float32 little-endian .bin into a flat vector (row-major). `expected` > 0 asserts size.
std::vector<float> loadF32Bin(const std::string& path, std::size_t expected = 0);

std::string readFile(const std::string& path);

}  // namespace quranrecite
