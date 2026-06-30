// JNI bridge: thin glue between the Kotlin QuranReciteDetector and the C++ Detector.
// Owns a Detector*, forwards feed/reset, and posts events back to Kotlin via a global
// callback ref. Keep logic in the C++ core; this file is plumbing only.
#include <jni.h>
#include <memory>
#include "quranrecite/detector.h"

using namespace quranrecite;

namespace {
struct Handle {
    std::unique_ptr<Detector> det;
    JavaVM* vm = nullptr;
    jobject listener = nullptr;   // global ref to a Kotlin callback object
};

// TODO(port): on each AyahEvent, attach the current thread to the JVM, build a Kotlin
// event object (or call onAyahDetected/onAyahAdvance), and detach. Marshal to main thread
// in the Kotlin layer.
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_quranrecite_sdk_QuranReciteDetector_nativeCreate(
        JNIEnv* env, jobject /*thiz*/, jstring modelPath, jstring lexiconPath,
        jstring tokensPath, jstring filterbankPath, jstring hannPath /*, config fields */) {
    auto* h = new Handle();
    Config cfg;
    auto str = [&](jstring s) {
        const char* c = env->GetStringUTFChars(s, nullptr);
        std::string r(c); env->ReleaseStringUTFChars(s, c); return r;
    };
    cfg.modelPath = str(modelPath);
    cfg.lexiconPath = str(lexiconPath);
    cfg.tokensPath = str(tokensPath);
    cfg.melFilterbankPath = str(filterbankPath);
    cfg.hannWindowPath = str(hannPath);
    env->GetJavaVM(&h->vm);
    h->det = std::make_unique<Detector>(cfg);
    // h->det->setEventCallback([h](const AyahEvent& e){ /* post to Kotlin */ });
    return reinterpret_cast<jlong>(h);
}

extern "C" JNIEXPORT void JNICALL
Java_com_quranrecite_sdk_QuranReciteDetector_nativeFeed(
        JNIEnv* env, jobject, jlong handle, jshortArray pcm, jint sampleRate) {
    auto* h = reinterpret_cast<Handle*>(handle);
    jsize n = env->GetArrayLength(pcm);
    jshort* buf = env->GetShortArrayElements(pcm, nullptr);
    h->det->feedPcm16(buf, static_cast<std::size_t>(n), sampleRate);
    env->ReleaseShortArrayElements(pcm, buf, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
Java_com_quranrecite_sdk_QuranReciteDetector_nativeReset(JNIEnv*, jobject, jlong handle) {
    reinterpret_cast<Handle*>(handle)->det->reset();
}

extern "C" JNIEXPORT void JNICALL
Java_com_quranrecite_sdk_QuranReciteDetector_nativeDestroy(JNIEnv* env, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    if (h->listener) env->DeleteGlobalRef(h->listener);
    delete h;
}
