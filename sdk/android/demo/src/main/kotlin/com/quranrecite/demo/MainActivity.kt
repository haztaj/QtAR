package com.quranrecite.demo

import android.Manifest
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.quranrecite.demo.mushaf.HighlightInfo
import com.quranrecite.demo.mushaf.MushafFonts
import com.quranrecite.demo.mushaf.MushafRepository
import com.quranrecite.demo.mushaf.MushafScreen
import com.quranrecite.demo.mushaf.ayahKey
import com.quranrecite.sdk.Config
import com.quranrecite.sdk.Mode
import com.quranrecite.sdk.HighlightState
import com.quranrecite.sdk.PendingReason
import com.quranrecite.sdk.QuranReciteDetector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Mushaf demo: renders real Quran pages (KFGQPC V2 glyph fonts), swipe RTL between pages,
 * jump to any page, and start/stop live ayah detection. The detected ayah highlights on the
 * page and the view auto-advances — exercising the full SDK surface (prepare → listen →
 * onHighlightState). The whole thing re-fits on foldable/orientation changes (see the
 * Activity's configChanges + the page's BoxWithConstraints sizing).
 */
class MainActivity : ComponentActivity() {

    private lateinit var detector: QuranReciteDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Keep the screen on while reciting, and run fullscreen (hide status + nav bars).
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        enterImmersive()
        // chainSubMin = 0 enables Phase-2 posterior-aware scoring (the ~30% PER phone-mic win).
        // streaming = true prefers the incremental StreamingModel when its graphs are bundled
        // (-PbundleStreaming); it safely falls back to the windowed re-decode otherwise.
        detector = QuranReciteDetector(this,
            Config(mode = Mode.CHAIN, chainSubMin = 0.0f, streaming = true))

        setContent {
            MaterialTheme {
                var status by remember { mutableStateOf("Preparing model…") }
                var modelReady by remember { mutableStateOf(false) }
                var listening by remember { mutableStateOf(false) }
                var highlight by remember { mutableStateOf(HighlightInfo()) }
                var modelUpdate by remember { mutableStateOf<Pair<String, String>?>(null) }  // (version, what's-new)

                // Debug toggles (persisted) — control logcat + session recording at runtime from
                // the top control panel, so the instrumentation stays in the build, off by default.
                val prefs = remember { getSharedPreferences("qr_debug", MODE_PRIVATE) }
                var debugLogging by remember { mutableStateOf(prefs.getBoolean("logging", false)) }
                var recording by remember { mutableStateOf(prefs.getBoolean("recording", false)) }

                // Ensure the page fonts (downloaded once, ~199 MB — survives app updates) then open
                // the mushaf DBs — all off the main thread. First launch shows download progress;
                // failures surface a retryable error rather than an endless spinner.
                var loadMsg by remember { mutableStateOf("Loading…") }
                var loadFrac by remember { mutableStateOf<Float?>(null) }   // 0..1 while downloading, else null
                var loadFailed by remember { mutableStateOf(false) }
                var retryKey by remember { mutableStateOf(0) }
                val repo by produceState<MushafRepository?>(null, retryKey) {
                    loadFailed = false; loadFrac = null; loadMsg = "Loading…"
                    value = withContext(Dispatchers.IO) {
                        try {
                            val fonts = MushafFonts.ensure(this@MainActivity) { p ->
                                loadMsg = "Downloading text (one time)"; loadFrac = p
                            }
                            loadFrac = null; loadMsg = "Opening mushaf…"
                            MushafRepository.open(this@MainActivity, fonts)
                        } catch (t: Throwable) {
                            loadMsg = "Couldn’t download text: ${t.message}"; loadFailed = true
                            null
                        }
                    }
                }

                val mic = rememberLauncherForPermission { granted ->
                    if (granted) {
                        // Fresh session: clear the engine's rolling buffer + matcher/context/
                        // highlight state (else session 2 inherits session 1's commits) and
                        // clear the on-screen highlight.
                        detector.reset()
                        highlight = HighlightInfo()
                        startSessionLog()             // fresh detection trace for this session
                        detector.start()
                        listening = true
                        status = "Listening…"
                    } else status = "Mic permission denied"
                }

                LaunchedEffect(Unit) {
                    detector.setListener(object : QuranReciteDetector.Listener {
                        override fun onModelDownloadProgress(fraction: Float) {
                            status = "Downloading model ${(fraction * 100).toInt()}%"
                        }
                        override fun onModelUpdated(version: String, description: String) {
                            if (debugLogging) Log.i("QRDemo", "onModelUpdated $version: $description")
                            modelUpdate = version to description
                        }
                        override fun onModelReady() {
                            if (debugLogging) Log.i("QRDemo", "onModelReady")
                            modelReady = true; status = "Ready — tap Start"
                        }
                        override fun onHighlightState(state: HighlightState) {
                            if (debugLogging) Log.i("QRDemo", "highlight active=${state.active?.let { "${it.surah}:${it.ayah}" }}" +
                                " upNext=${state.upNext?.let { "${it.surah}:${it.ayah}" }}" +
                                " confirmed=${state.confirmed.joinToString(",") { "${it.surah}:${it.ayah}" }}" +
                                " pending=${state.pending?.let { p -> "${p.reason}:${p.options.joinToString("/") { "${it.surah}:${it.ayah}" }}" }}")
                            logHighlight(state)                 // into the shareable debug trace
                            highlight = state.toInfo()
                            status = state.pending?.let { p ->
                                val opts = p.options.joinToString(" / ") { "${it.surah}:${it.ayah}" }
                                if (p.reason == PendingReason.NEEDS_CHOICE) "Choose: $opts" else "Deciding… ($opts)"
                            } ?: state.active?.let {
                                // append waqf-segment progress "N/M" for segmented ayat (Mode.CHAIN)
                                val seg = if (state.activeSegmentCount > 1)
                                    " · part ${state.activeSegment}/${state.activeSegmentCount}" else ""
                                "Listening — ${it.surah}:${it.ayah}$seg"
                            } ?: "Listening…"
                        }
                        override fun onError(error: Throwable) {
                            if (debugLogging) Log.e("QRDemo", "onError", error)
                            logSession("ERROR: ${error.message}")
                            status = "Error: ${error.message}"
                        }
                    })
                    detector.setDebugLogging(debugLogging)   // carried to the engine at onReady
                    detector.setRecording(recording)
                    detector.prepare()
                }

                val r = repo
                if (r == null) {
                    Column(
                        Modifier.fillMaxSize().padding(32.dp),
                        verticalArrangement = Arrangement.Center,
                        horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        when {
                            loadFailed -> {
                                Text(loadMsg, textAlign = TextAlign.Center)
                                Spacer(Modifier.height(16.dp))
                                Button(onClick = { retryKey++ }) { Text("Retry") }
                            }
                            loadFrac != null -> {   // determinate download progress
                                LinearProgressIndicator(
                                    progress = { loadFrac ?: 0f },
                                    modifier = Modifier.fillMaxWidth(0.7f))
                                Spacer(Modifier.height(12.dp))
                                Text("$loadMsg — ${((loadFrac ?: 0f) * 100).toInt()}%",
                                     textAlign = TextAlign.Center)
                            }
                            else -> {               // indeterminate (opening DBs, etc.)
                                CircularProgressIndicator()
                                Spacer(Modifier.height(16.dp))
                                Text(loadMsg)
                            }
                        }
                    }
                } else {
                    MushafScreen(
                        repo = r,
                        highlight = highlight,
                        status = status,
                        modelReady = modelReady,
                        listening = listening,
                        onToggleListen = {
                            if (listening) {
                                if (debugLogging) Log.i("QRDemo", "STOP pressed")
                                logSession("STOP")
                                detector.stop(); listening = false; status = "Stopped"
                            } else {
                                if (debugLogging) Log.i("QRDemo", "START pressed")
                                mic.launch(Manifest.permission.RECORD_AUDIO)
                            }
                        },
                        debugLogging = debugLogging,
                        onDebugLoggingChange = {
                            debugLogging = it; detector.setDebugLogging(it)
                            prefs.edit().putBoolean("logging", it).apply()
                        },
                        recording = recording,
                        onRecordingChange = {
                            recording = it; detector.setRecording(it)
                            prefs.edit().putBoolean("recording", it).apply()
                        },
                        onShareRecording = { shareRecording() },
                    )
                }

                // "What's new" — shown once when a newly-released model was downloaded (not silent).
                modelUpdate?.let { (version, desc) ->
                    AlertDialog(
                        onDismissRequest = { modelUpdate = null },
                        title = { Text("Detection model updated") },
                        text = {
                            Column {
                                if (desc.isNotBlank()) Text(desc)
                                Spacer(Modifier.height(8.dp))
                                Text(version, style = MaterialTheme.typography.bodySmall,
                                     color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        },
                        confirmButton = { TextButton(onClick = { modelUpdate = null }) { Text("OK") } },
                    )
                }
            }
        }
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) enterImmersive()   // re-hide the bars after returning from background/dialogs
    }

    /** Hide the system bars; a swipe from the edge shows them transiently (sticky immersive). */
    private fun enterImmersive() {
        WindowInsetsControllerCompat(window, window.decorView).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    override fun onDestroy() { detector.release(); super.onDestroy() }

    // --- session debug trace (accumulated from detection callbacks; bundled by shareRecording) ---
    private val sessionLog = StringBuilder()
    private var sessionStartMs = 0L
    private var lastHlLine = ""

    /** Begin a fresh detection trace (called when a session starts). */
    fun startSessionLog() {
        synchronized(sessionLog) { sessionLog.setLength(0) }
        lastHlLine = ""
        sessionStartMs = System.currentTimeMillis()
        logSession("START")
    }

    /** Append a timestamped line to the current session trace. */
    fun logSession(line: String) {
        val t = if (sessionStartMs == 0L) 0.0 else (System.currentTimeMillis() - sessionStartMs) / 1000.0
        synchronized(sessionLog) { sessionLog.append("+%6.1fs  %s%n".format(t, line)) }
    }

    /** Log a highlight snapshot, de-duplicated against the previous line. */
    private fun logHighlight(state: HighlightState) {
        val line = "active=${state.active?.let { "${it.surah}:${it.ayah}" } ?: "-"}" +
            " upNext=${state.upNext?.let { "${it.surah}:${it.ayah}" } ?: "-"}" +
            " confirmed=[${state.confirmed.joinToString(",") { "${it.surah}:${it.ayah}" }}]" +
            (state.pending?.let { p -> " pending=${p.reason}:${p.options.joinToString("/") { "${it.surah}:${it.ayah}" }}" } ?: "")
        if (line != lastHlLine) { lastHlLine = line; logSession(line) }
    }

    /**
     * Share the last debug session as a ZIP bundling everything useful for debugging: the session
     * WAV (exact 16 kHz PCM the engine heard), an `info.txt` with device / app / model / config,
     * and the timestamped detection trace. Record a session first ('Record session audio').
     */
    private fun shareRecording() {
        val wav = detector.lastRecording()?.let { java.io.File(it) }
        if (wav == null || !wav.exists()) {
            android.widget.Toast.makeText(this, "No recording yet — enable 'Record session audio' and run detection",
                android.widget.Toast.LENGTH_LONG).show()
            return
        }
        val zip = java.io.File(getExternalFilesDir(null), "qtar_debug_${wav.nameWithoutExtension}.zip")
        try {
            java.util.zip.ZipOutputStream(zip.outputStream().buffered()).use { zos ->
                zos.putNextEntry(java.util.zip.ZipEntry("info.txt"))
                zos.write(buildDebugInfo().toByteArray()); zos.closeEntry()
                zos.putNextEntry(java.util.zip.ZipEntry(wav.name))
                wav.inputStream().use { it.copyTo(zos) }; zos.closeEntry()
            }
        } catch (t: Throwable) {
            android.widget.Toast.makeText(this, "Couldn't bundle debug zip: ${t.message}",
                android.widget.Toast.LENGTH_LONG).show()
            return
        }
        val uri = androidx.core.content.FileProvider.getUriForFile(
            this, "$packageName.fileprovider", zip)
        val send = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
            type = "application/zip"
            putExtra(android.content.Intent.EXTRA_STREAM, uri)
            addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(android.content.Intent.createChooser(send, "Share debug session"))
    }

    /** Human-readable device / app / model / config context for the debug bundle. */
    private fun buildDebugInfo(): String {
        val pm = runCatching { packageManager.getPackageInfo(packageName, 0) }.getOrNull()
        @Suppress("DEPRECATION") val vcode = pm?.versionCode
        return buildString {
            appendLine("QtAR debug session")
            appendLine("generated: " + java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.US)
                .format(java.util.Date()))
            appendLine()
            appendLine("== Device ==")
            appendLine("manufacturer: ${android.os.Build.MANUFACTURER}")
            appendLine("model: ${android.os.Build.MODEL} (${android.os.Build.DEVICE})")
            appendLine("android: ${android.os.Build.VERSION.RELEASE} (SDK ${android.os.Build.VERSION.SDK_INT})")
            appendLine("abis: ${android.os.Build.SUPPORTED_ABIS.joinToString()}")
            appendLine()
            appendLine("== App ==")
            appendLine("package: $packageName")
            appendLine("version: ${pm?.versionName} ($vcode)")
            appendLine()
            appendLine("== Detector ==")
            appendLine("mode: CHAIN")
            appendLine("model: ${detector.modelName() ?: "(not ready)"}")
            appendLine("recording: ${detector.lastRecording()?.substringAfterLast('/')}")
            appendLine()
            appendLine("== Session detection trace ==")
            synchronized(sessionLog) { append(sessionLog.toString().ifBlank { "(no detections)\n" }) }
        }
    }
}

/** Map the SDK snapshot to the renderer's highlight sets (keys are "surah:ayah"). Two-phase:
 *  only the detected ayah (lighter) and the predicted next (darker) are shown — no trail. */
private fun HighlightState.toInfo() = HighlightInfo(
    active = active?.let { ayahKey(it.surah, it.ayah) },
    upNext = upNext?.let { ayahKey(it.surah, it.ayah) },
    options = pending?.options?.map { ayahKey(it.surah, it.ayah) }?.toSet() ?: emptySet(),
    segment = activeSegment,
    segmentCount = activeSegmentCount,
)

@Composable
private fun rememberLauncherForPermission(onResult: (Boolean) -> Unit) =
    androidx.activity.compose.rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(), onResult)
