// JNI bridge: thin glue between the Kotlin QuranReciteDetector and the C++ Detector.
// Owns a Detector*, forwards feed/reset, and posts events back to Kotlin by calling the
// detector object's emitDetected/emitAdvance methods. Keep logic in the C++ core; this
// file is plumbing only.
#include <jni.h>

#include <memory>
#include <string>

#include "quranrecite/detector.h"

using namespace quranrecite;

namespace {
struct Handle {
    std::unique_ptr<Detector> det;
    JavaVM* vm = nullptr;
    jobject self = nullptr;         // global ref to the Kotlin QuranReciteDetector
    jmethodID midDetected = nullptr;  // emitDetected(surah, ayah, confidence)
    jmethodID midAdvance = nullptr;   // emitAdvance(fromS, fromA, toS, toA)
};

// Resolve a JNIEnv for the current thread, attaching if this is a foreign (non-Java)
// thread. Returns whether we attached (caller must detach iff true).
bool envFor(Handle* h, JNIEnv** env) {
    if (h->vm->GetEnv(reinterpret_cast<void**>(env), JNI_VERSION_1_6) == JNI_OK) return false;
    h->vm->AttachCurrentThread(env, nullptr);
    return true;
}

void postEvent(Handle* h, const AyahEvent& e) {
    if (!h->self) return;
    JNIEnv* env = nullptr;
    bool attached = envFor(h, &env);
    if (env) {
        if (e.type == EventType::Advance && h->midAdvance) {
            env->CallVoidMethod(h->self, h->midAdvance, e.from.surah, e.from.ayah,
                                e.ayah.surah, e.ayah.ayah);
        } else if (h->midDetected) {  // Detect / Jump -> a (re)detection of e.ayah
            env->CallVoidMethod(h->self, h->midDetected, e.ayah.surah, e.ayah.ayah,
                                static_cast<jfloat>(e.confidence));
        }
        if (env->ExceptionCheck()) env->ExceptionClear();
    }
    if (attached) h->vm->DetachCurrentThread();
}

std::string jstr(JNIEnv* env, jstring s) {
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string r(c ? c : "");
    if (c) env->ReleaseStringUTFChars(s, c);
    return r;
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_quranrecite_sdk_QuranReciteDetector_nativeCreate(
        JNIEnv* env, jobject thiz, jstring modelPath, jstring lexiconPath,
        jstring tokensPath, jstring filterbankPath, jstring hannPath) {
    auto* h = new Handle();
    env->GetJavaVM(&h->vm);
    h->self = env->NewGlobalRef(thiz);
    jclass cls = env->GetObjectClass(thiz);
    h->midDetected = env->GetMethodID(cls, "emitDetected", "(IIF)V");
    h->midAdvance = env->GetMethodID(cls, "emitAdvance", "(IIII)V");

    Config cfg;
    cfg.modelPath = jstr(env, modelPath);
    cfg.lexiconPath = jstr(env, lexiconPath);
    cfg.tokensPath = jstr(env, tokensPath);
    cfg.melFilterbankPath = jstr(env, filterbankPath);
    cfg.hannWindowPath = jstr(env, hannPath);

    h->det = std::make_unique<Detector>(cfg);
    h->det->setEventCallback([h](const AyahEvent& e) { postEvent(h, e); });
    return reinterpret_cast<jlong>(h);
}

extern "C" JNIEXPORT void JNICALL
Java_com_quranrecite_sdk_QuranReciteDetector_nativeFeed(
        JNIEnv* env, jobject, jlong handle, jshortArray pcm, jint sampleRate) {
    auto* h = reinterpret_cast<Handle*>(handle);
    jsize n = env->GetArrayLength(pcm);
    jshort* buf = env->GetShortArrayElements(pcm, nullptr);
    h->det->feedPcm16(buf, static_cast<std::size_t>(n), sampleRate);
    env->ReleaseShortArrayElements(pcm, buf, JNI_ABORT);  // read-only: don't copy back
}

extern "C" JNIEXPORT void JNICALL
Java_com_quranrecite_sdk_QuranReciteDetector_nativeReset(JNIEnv*, jobject, jlong handle) {
    reinterpret_cast<Handle*>(handle)->det->reset();
}

extern "C" JNIEXPORT void JNICALL
Java_com_quranrecite_sdk_QuranReciteDetector_nativeDestroy(JNIEnv* env, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    h->det.reset();  // stop the engine (and its callback) before releasing the global ref
    if (h->self) env->DeleteGlobalRef(h->self);
    delete h;
}
