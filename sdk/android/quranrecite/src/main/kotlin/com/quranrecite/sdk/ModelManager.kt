package com.quranrecite.sdk

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlin.concurrent.thread

/** Resolved on-device asset paths the native engine needs. */
data class ModelAssets(
    val modelPath: String,
    val lexiconPath: String,
    val tokensPath: String,
    val filterbankPath: String,
    val hannPath: String,
    val ambiguousPath: String,   // Stage-3 confusable map; "" if not bundled (deferral off)
    val vadPath: String,         // Silero VAD; "" if not bundled (no paused-recitation reset)
    val unitPhonemesPath: String,// waqf-segment unit lexicon; "" if not bundled (Chain mode off)
    val streamConvPath: String = "",    // streaming Conv2dSubsampling ONNX; "" -> windowed decode
    val streamEncoderPath: String = "", // streaming Emformer-step ONNX; "" -> windowed decode
)

/** A hosted file + its expected sha256 (a manifest sub-asset). */
private data class RemoteAsset(val url: String, val sha256: String)

/** A released model, from the remote manifest (see [ModelManager.MODEL_MANIFEST_URL]).
 *  `description` is the optional "what's new" note shown to the user on an update. `streamConv`
 *  / `streamEncoder` are the OPTIONAL true-streaming graphs (version-coupled to this model);
 *  present => the SDK downloads them and Mode.CHAIN decodes incrementally, null => windowed. */
private data class ModelRelease(
    val version: String, val url: String, val sha256: String, val description: String,
    val streamConv: RemoteAsset?, val streamEncoder: RemoteAsset?,
)

/**
 * Asset delivery. The small assets (lexicon, tokens, mel filterbank, Hann window, + optional
 * ambiguity map / VAD / unit lexicon) ship inside the APK/.aar under assets/quranrecite/ and are
 * extracted to internal files (keyed by [ASSETS_VERSION], bumped when the bundled corpus changes).
 *
 * The ~13 MB ONNX model is NOT shipped in the default build — it is fetched once and cached in
 * EXTERNAL files (survives app updates), driven by a small remote MANIFEST:
 *
 *     GET MODEL_MANIFEST_URL -> {"version","url","sha256","description",
 *                                "streamConv":{"url","sha256"}, "streamEncoder":{"url","sha256"}}
 *
 * On each launch (with network) the app reads the manifest and compares `version` to what it has
 * cached: same -> use the cache; different (a NEW model was released) -> download + verify + cache,
 * then prune the old one. This lets a new model be pushed by uploading it and updating the manifest,
 * with no app update. Offline, the newest cached model is used; the very first launch needs a
 * connection unless the model is bundled.
 *
 * The optional `streamConv`/`streamEncoder` graphs (true-streaming Mode.CHAIN, version-coupled to
 * the model) are delivered the same way — downloaded into models/stream/, sha256-verified, and used
 * when Config.streaming is on (the default). Their download failing is non-fatal (windowed fallback).
 *
 * Dev/offline: a build with `-PbundleModel` puts the model in the APK at
 * assets/quranrecite/model.int8.onnx; [resolveModel] then uses it directly (no manifest, no network).
 */
class ModelManager(private val context: Context, private val corpus: Corpus) {

    private val root = File(context.filesDir, "quranrecite").apply { mkdirs() }
    private val assetsDir = File(root, "assets-$ASSETS_VERSION").apply { mkdirs() }
    // Downloaded models live in EXTERNAL files so they survive app updates (like the page fonts).
    private val modelsDir = File(context.getExternalFilesDir(null), "quranrecite/models").apply { mkdirs() }
    // Streaming graphs in a SUBDIR so the model prune / newest-model scan (top-level *.onnx) never
    // touch them (they are version-coupled to the model but delivered as separate files).
    private val streamDir = File(modelsDir, "stream").apply { mkdirs() }

    /** Resolve all assets off the main thread; callbacks fire on the worker thread.
     *  [onModelUpdate] fires with (version, description) when a NEW model replaces a previously
     *  cached one (a genuine update — not the first-launch download). */
    fun ensureAsync(
        onProgress: (Float) -> Unit,
        onReady: (ModelAssets) -> Unit,
        onError: (Throwable) -> Unit,
        onModelUpdate: (version: String, description: String) -> Unit = { _, _ -> },
    ) {
        thread(name = "quranrecite-model") {
            try {
                val lexicon = extractBundled("ayah_phonemes.json")
                val tokens = extractBundled("tokens.txt")
                val filterbank = extractBundled("mel_filterbank.bin")
                val hann = extractBundled("hann_window.bin")
                val ambiguous = if (assetExists("quranrecite/ambiguous_ayat.json"))
                    extractBundled("ambiguous_ayat.json") else ""
                val vad = if (assetExists("quranrecite/silero_vad.onnx"))
                    extractBundled("silero_vad.onnx") else ""
                val units = if (assetExists("quranrecite/unit_phonemes.json"))
                    extractBundled("unit_phonemes.json") else ""
                // True streaming acoustics (version-coupled to the model). Bundled by
                // -PbundleStreaming takes precedence (dev/offline); otherwise downloaded via the
                // manifest alongside the model. Absent both -> "" -> windowed re-decode.
                var streamConv = if (assetExists("quranrecite/$STREAM_CONV"))
                    extractBundled(STREAM_CONV, into = streamDir) else ""
                var streamEnc = if (assetExists("quranrecite/$STREAM_ENCODER"))
                    extractBundled(STREAM_ENCODER, into = streamDir) else ""
                // Fetch the manifest ONCE (skipped entirely for a fully-bundled model — no network).
                val bundledModel = assetExists("quranrecite/$BUNDLED_MODEL")
                val release = if (bundledModel) null else fetchManifest()
                val model = resolveModel(bundledModel, release, onProgress, onModelUpdate)
                if (streamConv.isEmpty() && release != null)
                    resolveStreaming(release, onProgress)?.let { streamConv = it.first; streamEnc = it.second }
                onReady(ModelAssets(model, lexicon, tokens, filterbank, hann, ambiguous, vad, units,
                    streamConv, streamEnc))
            } catch (t: Throwable) {
                onError(t)
            }
        }
    }

    /** Bundled dev model → else the manifest's current release (cached or downloaded) → else the
     *  newest cached model (offline). Takes the pre-fetched [release] (fetched once in ensureAsync). */
    private fun resolveModel(
        bundled: Boolean,
        release: ModelRelease?,
        onProgress: (Float) -> Unit,
        onModelUpdate: (version: String, description: String) -> Unit,
    ): String {
        // 1. Bundled dev/offline model (-PbundleModel): use it directly, no network.
        if (bundled) return extractBundled(BUNDLED_MODEL, into = modelsDir)

        // 2. Remote manifest = the currently released model. A different version than we have
        //    cached means a new release -> download it (so updates need no app update).
        release?.let { release ->
            val cached = File(modelsDir, "${release.version}.onnx")
            if (cached.exists() && (release.sha256.isEmpty() || sha256(cached) == release.sha256))
                return cached.absolutePath
            // A prior cached model means this download is an UPDATE (not the first install).
            val isUpdate = newestCachedModel() != null
            download(release.url, cached, onProgress)
            if (release.sha256.isNotEmpty() && sha256(cached) != release.sha256) {
                cached.delete()
                error("Downloaded model failed sha256 verification (${release.version})")
            }
            modelsDir.listFiles { f -> f.extension == "onnx" && f != cached }?.forEach { it.delete() }
            if (isUpdate) onModelUpdate(release.version, release.description)
            return cached.absolutePath
        }

        // 3. Offline with no manifest: use the newest model we have cached, if any.
        newestCachedModel()?.let { return it.absolutePath }
        error("No bundled model, the model manifest is unreachable, and none is cached — the first " +
            "launch needs a network connection (or build the app with -PbundleModel).")
    }

    /** Download (or reuse the cached) streaming graphs for [release], into [streamDir] keyed by
     *  version + sha256-verified; prune older versions. Returns (conv, encoder) absolute paths, or
     *  null if the manifest carries no streaming graphs OR the download fails (-> windowed fallback:
     *  streaming is an optimization, never a hard dependency). Offline reuses the cache if present. */
    private fun resolveStreaming(release: ModelRelease, onProgress: (Float) -> Unit): Pair<String, String>? {
        val conv = release.streamConv ?: return null
        val enc = release.streamEncoder ?: return null
        return runCatching {
            val convFile = File(streamDir, "${release.version}.stream_conv.onnx")
            val encFile = File(streamDir, "${release.version}.stream_encoder.onnx")
            for ((asset, dest) in listOf(conv to convFile, enc to encFile)) {
                if (dest.exists() && (asset.sha256.isEmpty() || sha256(dest) == asset.sha256)) continue
                download(asset.url, dest, onProgress)
                if (asset.sha256.isNotEmpty() && sha256(dest) != asset.sha256) {
                    dest.delete(); error("Streaming graph failed sha256 verification (${dest.name})")
                }
            }
            streamDir.listFiles { f -> f.extension == "onnx" && f != convFile && f != encFile }
                ?.forEach { it.delete() }
            convFile.absolutePath to encFile.absolutePath
        }.getOrNull()
    }

    /** GET the manifest and parse it; null on any failure (offline / unset / malformed). */
    private fun fetchManifest(): ModelRelease? {
        if (MODEL_MANIFEST_URL.isEmpty()) return null
        return runCatching {
            val conn = (URL(MODEL_MANIFEST_URL).openConnection() as HttpURLConnection).apply {
                connectTimeout = 10_000; readTimeout = 10_000
            }
            val json = try {
                conn.inputStream.use { it.readBytes().decodeToString() }
            } finally {
                conn.disconnect()
            }
            val o = JSONObject(json)
            fun asset(key: String): RemoteAsset? = o.optJSONObject(key)?.let {
                RemoteAsset(it.getString("url"), it.optString("sha256", ""))
            }
            ModelRelease(o.getString("version"), o.getString("url"),
                o.optString("sha256", ""), o.optString("description", ""),
                asset("streamConv"), asset("streamEncoder"))
        }.getOrNull()
    }

    private fun newestCachedModel(): File? =
        modelsDir.listFiles { f -> f.extension == "onnx" }?.maxByOrNull { it.lastModified() }

    private fun download(url: String, dest: File, onProgress: (Float) -> Unit) {
        dest.parentFile?.mkdirs()
        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 15_000; readTimeout = 30_000
        }
        try {
            val total = conn.contentLengthLong
            val tmp = File(dest.parentFile, dest.name + ".part")
            conn.inputStream.use { input ->
                tmp.outputStream().use { output ->
                    val buf = ByteArray(64 * 1024)
                    var read = 0L
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        output.write(buf, 0, n)
                        read += n
                        if (total > 0) onProgress((read.toDouble() / total).toFloat())
                    }
                }
            }
            if (!tmp.renameTo(dest)) { tmp.copyTo(dest, overwrite = true); tmp.delete() }
        } finally {
            conn.disconnect()
        }
    }

    private fun assetExists(path: String): Boolean =
        runCatching { context.assets.open(path).close() }.isSuccess

    /** Copy a file from the APK's assets/quranrecite/ into [into] (once). */
    private fun extractBundled(name: String, into: File = assetsDir): String {
        val out = File(into, name)
        if (!out.exists()) {
            into.mkdirs()
            context.assets.open("quranrecite/$name").use { i -> out.outputStream().use { o -> i.copyTo(o) } }
        }
        return out.absolutePath
    }

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { s ->
            val buf = ByteArray(64 * 1024)
            while (true) { val n = s.read(buf); if (n < 0) break; md.update(buf, 0, n) }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        // Bundled small-asset version — bump when the shipped lexicon/tokens/etc change (corpus
        // change). Independent of the model version, which comes from the manifest.
        const val ASSETS_VERSION = "s123-v1"
        // Remote manifest of the current released model: {"version","url","sha256"}. Publish a new
        // model by uploading the .onnx and updating this JSON (see `./gradlew :demo:modelManifest`).
        // Empty -> no download (rely on a bundled or cached model).
        const val MODEL_MANIFEST_URL =
            "https://github.com/haztaj/QtAR/releases/download/model/model_manifest.json"
        // Name of the model when bundled in the APK (-PbundleModel) — takes precedence, no network.
        const val BUNDLED_MODEL = "model.int8.onnx"
        // True streaming graphs, bundled only by -PbundleStreaming (paired with -PbundleModel; the
        // same checkpoint). Present -> Mode.CHAIN can decode incrementally (Config.streaming).
        const val STREAM_CONV = "stream_conv.onnx"
        const val STREAM_ENCODER = "stream_encoder.int8.onnx"
    }
}
