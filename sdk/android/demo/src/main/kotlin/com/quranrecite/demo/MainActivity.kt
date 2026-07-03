package com.quranrecite.demo

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.quranrecite.demo.mushaf.HighlightInfo
import com.quranrecite.demo.mushaf.MushafRepository
import com.quranrecite.demo.mushaf.MushafScreen
import com.quranrecite.demo.mushaf.ayahKey
import com.quranrecite.sdk.Config
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
        detector = QuranReciteDetector(this, Config())

        setContent {
            MaterialTheme {
                var status by remember { mutableStateOf("Preparing model…") }
                var modelReady by remember { mutableStateOf(false) }
                var listening by remember { mutableStateOf(false) }
                var highlight by remember { mutableStateOf(HighlightInfo()) }

                // Load the mushaf DBs/fonts off the main thread.
                val repo by produceState<MushafRepository?>(null) {
                    value = withContext(Dispatchers.IO) { MushafRepository.open(this@MainActivity) }
                }

                val mic = rememberLauncherForPermission { granted ->
                    if (granted) {
                        // Fresh session: clear the engine's rolling buffer + matcher/context/
                        // highlight state (else session 2 inherits session 1's commits) and
                        // clear the on-screen highlight.
                        detector.reset()
                        highlight = HighlightInfo()
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
                        override fun onModelReady() { modelReady = true; status = "Ready — tap Start" }
                        override fun onHighlightState(state: HighlightState) {
                            highlight = state.toInfo()
                            status = state.pending?.let { p ->
                                val opts = p.options.joinToString(" / ") { "${it.surah}:${it.ayah}" }
                                if (p.reason == PendingReason.NEEDS_CHOICE) "Choose: $opts" else "Deciding… ($opts)"
                            } ?: state.active?.let { "Listening — ${it.surah}:${it.ayah}" } ?: "Listening…"
                        }
                        override fun onError(error: Throwable) { status = "Error: ${error.message}" }
                    })
                    detector.prepare()
                }

                val r = repo
                if (r == null) {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator()
                    }
                } else {
                    MushafScreen(
                        repo = r,
                        highlight = highlight,
                        status = status,
                        modelReady = modelReady,
                        listening = listening,
                        onToggleListen = {
                            if (listening) { detector.stop(); listening = false; status = "Stopped" }
                            else mic.launch(Manifest.permission.RECORD_AUDIO)
                        },
                    )
                }
            }
        }
    }

    override fun onDestroy() { detector.release(); super.onDestroy() }
}

/** Map the SDK snapshot to the renderer's highlight sets (keys are "surah:ayah"). */
private fun HighlightState.toInfo() = HighlightInfo(
    active = active?.let { ayahKey(it.surah, it.ayah) },
    confirmed = confirmed.map { ayahKey(it.surah, it.ayah) }.toSet(),
    options = pending?.options?.map { ayahKey(it.surah, it.ayah) }?.toSet() ?: emptySet(),
)

@Composable
private fun rememberLauncherForPermission(onResult: (Boolean) -> Unit) =
    androidx.activity.compose.rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(), onResult)
