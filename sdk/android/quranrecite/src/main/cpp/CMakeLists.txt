cmake_minimum_required(VERSION 3.22)
project(quranrecite_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# ONNX Runtime for Android. The onnxruntime-android AAR is NOT prefab-packaged (it ships
# headers/ + jni/<abi>/libonnxruntime.so but no CMake config), so instead of find_package we
# build an IMPORTED target from the files Gradle's `extractOrt` task unpacks (passed as
# ORT_DIR). The runtime .so comes from the onnxruntime-android dependency in the final APK;
# here we only link against it (DT_NEEDED), we don't bundle a second copy.
if(NOT ORT_DIR)
    message(FATAL_ERROR "ORT_DIR not set — the Gradle extractOrt task provides it.")
endif()
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/headers")

# Pull in the shared C++ core (sdk/core). ORT comes from the imported target above; the
# desktop-only conformance/test executables are disabled here.
set(QR_BUILD_CONFORMANCE OFF CACHE BOOL "" FORCE)
set(QR_ORT_TARGET onnxruntime CACHE STRING "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core
                 ${CMAKE_BINARY_DIR}/quranrecite_core)

add_library(quranrecite_jni SHARED jni_bridge.cpp)

target_link_libraries(quranrecite_jni
    PRIVATE quranrecite_core
            onnxruntime
            android log)
