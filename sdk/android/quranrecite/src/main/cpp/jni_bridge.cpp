// JNI bridge: thin glue between the Kotlin QuranReciteDetector and the C++ Detector.
// Owns a Detector*, forwards feed/reset, and posts events back to Kotlin by calling the
// detector object's emitDetected/emitAdvance methods. Keep logic in the C++ core; this
// file is plumbing only.
#include <jni.h>

#include <memory>
#include <string>
#include <vector>

#include "quranrecite/detector.h"

using namespace quranrecite;

namespace {
struct Handle {
    std::unique_ptr<Detector> det;
    JavaVM* vm = nullptr;
    jobject self = nullptr;         // global ref to the Kotlin QuranReciteDetector
    jmethodID midDetected = nullptr;  // emitDetected(surah, ayah, confidence)
    jmethodID midAdvance = nullptr;   // emitAdvance(fromS, fromA, toS, toA)
    jmethodID midHighlight = nullptr; // emitHighlight(json) — the centralized snapshot
};

std::string ayahKey(const AyahId& a) {
    return std::to_string(a.surah) + ":" + std::to_string(a.ayah);
}

// Serialize the snapshot to the same JSON shape as conformance/golden/highlight (parsed in
// Kotlin). Keeping the marshalling as one string keeps the JNI surface trivial.
std::string snapshotJson(const HighlightSnapshot& s) {
    std::string j = "{\"confirmed\":[";
    for (std::size_t i = 0; i < s.confirmed.size(); ++i)
        j += (i ? ",\"" : "\"") + ayahKey(s.confirmed[i]) + "\"";
    j += "],\"pending\":";
    if (s.hasPending) {
        j += "{\"ayah\":";
        j += s.pending.hasAyah ? "\"" + ayahKey(s.pending.ayah) + "\"" : "null";
        j += ",\"options\":[";
        for (std::size_t i = 0; i < s.pending.options.size(); ++i)
            j += (i ? ",\"" : "\"") + ayahKey(s.pending.options[i]) + "\"";
        j += "],\"reason\":\"";
        j += s.pending.reason == PendingReason::NeedsChoice ? "needs_choice" : "await_successor";
        j += "\"}";
    } else {
        j += "null";
    }
    j += ",\"active\":";
    j += s.hasActive ? "\"" + ayahKey(s.active) + "\"" : "null";
    j += ",\"upNext\":";
    j += s.hasUpNext ? "\"" + ayahKey(s.upNext) + "\"" : "null";
    j += ",\"activeSegment\":" + std::to_string(s.activeSegment);
    j += ",\"activeSegmentCount\":" + std::to_string(s.activeSegmentCount);
    j += "}";
    return j;
}

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

void postHighlight(Handle* h, const HighlightSnapshot& snap) {
    if (!h->self || !h->midHighlight) return;
    JNIEnv* env = nullptr;
    bool attached = envFor(h, &env);
    if (env) {
        jstring js = env->NewStringUTF(snapshotJson(snap).c_str());
        env->CallVoidMethod(h->self, h->midHighlight, js);
        env->DeleteLocalRef(js);
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
        jstring tokensPath, jstring filterbankPath, jstring hannPath, jstring ambiguousPath,
        jstring vadPath, jint mode, jstring unitPhonemesPath, jfloat chainCost,
        jfloat chainSubMin, jstring streamConvPath, jstring streamEncoderPath,
        jboolean chainVadReset, jfloat chainResetMaxGap,
        jstring suffixModelPath, jfloat chainSuffixSec, jfloat normRms, jfloat chainPageBonus) {
    auto* h = new Handle();
    env->GetJavaVM(&h->vm);
    h->self = env->NewGlobalRef(thiz);
    jclass cls = env->GetObjectClass(thiz);
    h->midDetected = env->GetMethodID(cls, "emitDetected", "(IIF)V");
    h->midAdvance = env->GetMethodID(cls, "emitAdvance", "(IIII)V");
    h->midHighlight = env->GetMethodID(cls, "emitHighlight", "(Ljava/lang/String;)V");

    Config cfg;
    cfg.modelPath = jstr(env, modelPath);
    cfg.lexiconPath = jstr(env, lexiconPath);
    cfg.tokensPath = jstr(env, tokensPath);
    cfg.melFilterbankPath = jstr(env, filterbankPath);
    cfg.hannWindowPath = jstr(env, hannPath);
    cfg.ambiguousPath = jstr(env, ambiguousPath);   // empty -> deferral disabled (confirm all)
    cfg.vadPath = jstr(env, vadPath);               // empty -> no VAD (energy gate only)
    cfg.mode = static_cast<Mode>(mode);             // Kotlin Mode ordinal == types.h Mode
    cfg.unitPhonemesPath = jstr(env, unitPhonemesPath);  // required for Mode::Chain
    cfg.chainCost = chainCost;                      // fire threshold (phone mic ~0.45)
    cfg.chainSubMin = chainSubMin;                  // Phase-2 soft scoring (~0 for phones)
    // True streaming acoustics (both empty -> windowed re-decode, the default). When set,
    // Mode::Chain decodes only the new audio each hop (battery/latency) — must be the SAME
    // weights as modelPath.
    cfg.streamConvPath = jstr(env, streamConvPath);
    cfg.streamEncoderPath = jstr(env, streamEncoderPath);
    // Focused-window VAD reset (windowed Chain only — safe no-op in streaming). Gated: reset
    // only when the pause closely follows a commit (see types.h chainResetMaxGap).
    cfg.chainVadReset = chainVadReset == JNI_TRUE;
    cfg.chainResetMaxGap = chainResetMaxGap;
    // v13 fresh-context suffix decode (windowed Chain; empty path or 0 sec -> off).
    cfg.chainSuffixModelPath = jstr(env, suffixModelPath);
    cfg.chainSuffixSec = chainSuffixSec;
    cfg.normRms = normRms;                          // gain-normalize target (demo: 0.15 for quiet mics)
    cfg.chainPageBonus = chainPageBonus;            // page-context prior: off-page penalty (0 = off)

    h->det = std::make_unique<Detector>(cfg);
    h->det->setEventCallback([h](const AyahEvent& e) { postEvent(h, e); });
    h->det->setHighlightCallback([h](const HighlightSnapshot& s) { postHighlight(h, s); });
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

// Page-context prior: `keys` packs each ayah as surah*1000 + ayah (see Kotlin setPageContext).
extern "C" JNIEXPORT void JNICALL
Java_com_quranrecite_sdk_QuranReciteDetector_nativeSetPageContext(
        JNIEnv* env, jobject, jlong handle, jintArray keys) {
    auto* h = reinterpret_cast<Handle*>(handle);
    jsize n = env->GetArrayLength(keys);
    jint* k = env->GetIntArrayElements(keys, nullptr);
    std::vector<AyahId> page;
    page.reserve(n);
    for (jsize i = 0; i < n; ++i) page.push_back({k[i] / 1000, k[i] % 1000});
    env->ReleaseIntArrayElements(keys, k, JNI_ABORT);   // read-only
    h->det->setPageContext(page);
}

extern "C" JNIEXPORT void JNICALL
Java_com_quranrecite_sdk_QuranReciteDetector_nativeSetDebug(JNIEnv*, jobject, jlong handle, jboolean on) {
    reinterpret_cast<Handle*>(handle)->det->setDebug(on == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL
Java_com_quranrecite_sdk_QuranReciteDetector_nativeDestroy(JNIEnv* env, jobject, jlong handle) {
    auto* h = reinterpret_cast<Handle*>(handle);
    h->det.reset();  // stop the engine (and its callback) before releasing the global ref
    if (h->self) env->DeleteGlobalRef(h->self);
    delete h;
}
