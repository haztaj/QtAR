cmake_minimum_required(VERSION 3.22)
project(quranrecite_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# ONNX Runtime Android prefab module (from the onnxruntime-android AAR dependency).
# Resolve it first so its imported target exists before the core links against it.
find_package(onnxruntime REQUIRED CONFIG)

# Pull in the shared C++ core (sdk/core). ORT comes from the prefab target (not ORT_HOME);
# the desktop-only conformance/test executables are disabled here.
set(QR_BUILD_CONFORMANCE OFF CACHE BOOL "" FORCE)
set(QR_ORT_TARGET onnxruntime::onnxruntime CACHE STRING "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core
                 ${CMAKE_BINARY_DIR}/quranrecite_core)

add_library(quranrecite_jni SHARED jni_bridge.cpp)

target_link_libraries(quranrecite_jni
    PRIVATE quranrecite_core
            onnxruntime::onnxruntime
            android log)
