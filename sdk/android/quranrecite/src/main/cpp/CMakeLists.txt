cmake_minimum_required(VERSION 3.22)
project(quranrecite_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Pull in the shared C++ core (sdk/core). It builds without ORT_HOME here because ORT is
# provided by the onnxruntime-android prefab below.
set(QR_BUILD_CONFORMANCE OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core
                 ${CMAKE_BINARY_DIR}/quranrecite_core)

# ONNX Runtime Android prefab module (from the onnxruntime-android AAR dependency).
find_package(onnxruntime REQUIRED CONFIG)

add_library(quranrecite_jni SHARED jni_bridge.cpp)

target_link_libraries(quranrecite_jni
    PRIVATE quranrecite_core
            onnxruntime::onnxruntime
            android log)
