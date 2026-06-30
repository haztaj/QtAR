cmake_minimum_required(VERSION 3.22)
project(quranrecite_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # required for the Android .so

# --- ONNX Runtime (mobile) ---------------------------------------------------
# Point ORT_HOME at the onnxruntime headers + lib for the target platform.
# Desktop (for the conformance runner): a downloaded onnxruntime release.
# Android: the AAR's prefab module is wired in android/.../cpp/CMakeLists.txt instead.
set(ORT_HOME "" CACHE PATH "ONNX Runtime install (include/ + lib/)")

# --- third-party FFT ---------------------------------------------------------
# Use pffft or KissFFT for the STFT. Vendor under third_party/ and add here.
# add_subdirectory(third_party/pffft)

add_library(quranrecite_core STATIC
    src/frontend.cpp
    src/decoder.cpp
    src/matcher.cpp
    src/segmenter.cpp
    src/inference.cpp
    src/assets.cpp
    src/detector.cpp
)

target_include_directories(quranrecite_core
    PUBLIC  include
    PRIVATE src third_party
)

if(ORT_HOME)
    target_include_directories(quranrecite_core PRIVATE ${ORT_HOME}/include)
    target_link_directories(quranrecite_core PRIVATE ${ORT_HOME}/lib)
    target_link_libraries(quranrecite_core PRIVATE onnxruntime)
endif()

# --- desktop conformance runner ---------------------------------------------
# Builds the engine over conformance/ fixtures and writes outputs for verify.py.
option(QR_BUILD_CONFORMANCE "Build the conformance runner" ON)
if(QR_BUILD_CONFORMANCE)
    add_executable(conformance_runner tests/conformance_runner.cpp)
    target_include_directories(conformance_runner PRIVATE src third_party)
    target_link_libraries(conformance_runner PRIVATE quranrecite_core)
    if(MINGW)   # bundle the C++ runtime so the exe runs without the MinGW DLLs on PATH
        target_link_options(conformance_runner PRIVATE -static -static-libgcc -static-libstdc++)
    endif()
endif()
