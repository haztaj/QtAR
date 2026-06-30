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

# inference.cpp needs the ORT headers to compile; the desktop build links the DLL.
# (Android consumes ORT via the onnxruntime-android prefab — see android/.../cpp/CMakeLists.txt.)
if(NOT ORT_HOME)
    message(FATAL_ERROR "Set -DORT_HOME=<onnxruntime> (include/ + lib/) to build the core.")
endif()
target_include_directories(quranrecite_core PRIVATE ${ORT_HOME}/include)
if(MINGW)
    # ORT's header uses `_stdcall` (single underscore) which g++ rejects; harmless on x64.
    target_compile_definitions(quranrecite_core PRIVATE _stdcall=__stdcall)
    target_link_libraries(quranrecite_core PRIVATE ${ORT_HOME}/lib/onnxruntime.dll)
else()
    target_link_directories(quranrecite_core PRIVATE ${ORT_HOME}/lib)
    target_link_libraries(quranrecite_core PRIVATE onnxruntime)
endif()

# --- desktop tests ----------------------------------------------------------
# conformance_runner: front-end + matcher over conformance/ fixtures (writes outputs for
#   verify.py; no ORT pulled in -> can fully static-link).
# test_inference / test_detector: the ORT-backed stages and the full end-to-end pipeline.
option(QR_BUILD_CONFORMANCE "Build the desktop conformance/test executables" ON)
if(QR_BUILD_CONFORMANCE)
    add_executable(conformance_runner tests/conformance_runner.cpp)
    target_include_directories(conformance_runner PRIVATE src third_party)
    target_link_libraries(conformance_runner PRIVATE quranrecite_core)
    if(MINGW)   # no ORT symbols pulled -> bundle the full runtime
        target_link_options(conformance_runner PRIVATE -static -static-libgcc -static-libstdc++)
    endif()

    foreach(t test_inference test_detector)
        add_executable(${t} tests/${t}.cpp)
        target_include_directories(${t} PRIVATE include src ${ORT_HOME}/include)
        target_link_libraries(${t} PRIVATE quranrecite_core)
        if(MINGW)   # ORT is dynamic -> static C++ runtime only; link the DLL
            target_link_libraries(${t} PRIVATE ${ORT_HOME}/lib/onnxruntime.dll)
            target_link_options(${t} PRIVATE -static-libgcc -static-libstdc++)
        endif()
    endforeach()
endif()
