#include "assets.h"
#include <fstream>
#include <stdexcept>

namespace quranrecite {

std::vector<float> loadF32Bin(const std::string& path, std::size_t expected) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::streamsize bytes = f.tellg();
    f.seekg(0);
    std::vector<float> out(static_cast<std::size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(out.data()), bytes);
    if (expected && out.size() != expected)
        throw std::runtime_error("unexpected size for " + path);
    return out;  // NOTE: assumes little-endian host (Android/iOS are LE).
}

std::string readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

}  // namespace quranrecite
