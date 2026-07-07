cmake_minimum_required(VERSION 3.22)
project(quranrecite_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # required for the Android .so

# --- ONNX Runtime -----------------------------------------------------------
# Two ways to provide ORT to the core, depending on platform:
#   * QR_ORT_TARGET — an existing imported CMake target (its INTERFACE includes carry the
#     headers). Android passes the onnxruntime-android prefab: `onnxruntime::onnxruntime`.
#   * ORT_HOME     — a local install dir (include/ + lib/) for the desktop conformance build.
set(ORT_HOME "" CACHE PATH "ONNX Runtime install (include/ + lib/) — desktop build")
set(QR_ORT_TARGET "" CACHE STRING "Imported ORT target (e.g. Android prefab onnxruntime::onnxruntime)")

# --- third-party FFT ---------------------------------------------------------
# Use pffft or KissFFT for the STFT. Vendor under third_party/ and add here.
# add_subdirectory(third_party/pffft)

add_library(quranrecite_core STATIC
    src/frontend.cpp
    src/decoder.cpp
    src/matcher.cpp
    src/segmenter.cpp
    src/stream.cpp
    src/autodet.cpp
    src/highlight.cpp
    src/chain.cpp
    src/inference.cpp
    src/vad.cpp
    src/assets.cpp
    src/detector.cpp
)

target_include_directories(quranrecite_core
    PUBLIC  include
    PRIVATE src third_party
)

# inference.cpp needs the ORT headers to compile.
if(QR_ORT_TARGET)
    # Android (and any consumer supplying an imported ORT target): interface includes
    # carry the headers; the target provides the runtime .so.
    target_link_libraries(quranrecite_core PRIVATE ${QR_ORT_TARGET})
elseif(ORT_HOME)
    target_include_directories(quranrecite_core PRIVATE ${ORT_HOME}/include)
    if(MINGW)
        # ORT's header uses `_stdcall` (single underscore) which g++ rejects; harmless on x64.
        target_compile_definitions(quranrecite_core PRIVATE _stdcall=__stdcall)
        target_link_libraries(quranrecite_core PRIVATE ${ORT_HOME}/lib/onnxruntime.dll)
    else()
        target_link_directories(quranrecite_core PRIVATE ${ORT_HOME}/lib)
        target_link_libraries(quranrecite_core PRIVATE onnxruntime)
    endif()
else()
    message(FATAL_ERROR
        "Provide ORT: -DQR_ORT_TARGET=<imported target> (Android prefab) "
        "or -DORT_HOME=<install> (include/ + lib/, desktop).")
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

    foreach(t test_inference test_detector test_vad)
        add_executable(${t} tests/${t}.cpp)
        target_include_directories(${t} PRIVATE include src ${ORT_HOME}/include)
        target_link_libraries(${t} PRIVATE quranrecite_core)
        if(MINGW)   # ORT is dynamic -> static C++ runtime only; link the DLL
            target_link_libraries(${t} PRIVATE ${ORT_HOME}/lib/onnxruntime.dll)
            target_link_options(${t} PRIVATE -static-libgcc -static-libstdc++)
        endif()
    endforeach()
endif()
