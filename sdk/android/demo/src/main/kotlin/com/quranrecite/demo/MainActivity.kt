package com.quranrecite.demo

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.quranrecite.sdk.AyahId
import com.quranrecite.sdk.Config
import com.quranrecite.sdk.HighlightState
import com.quranrecite.sdk.PendingReason
import com.quranrecite.sdk.QuranReciteDetector

/**
 * Minimal SDK demo: a mushaf-style list of ayat; the current ayah highlights and the view
 * auto-advances as you recite. Shows the whole SDK surface — prepare → listen → events.
 */
class MainActivity : ComponentActivity() {

    private lateinit var detector: QuranReciteDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        detector = QuranReciteDetector(this, Config())

        setContent {
            MaterialTheme {
                var status by remember { mutableStateOf("Preparing model…") }
                var current by remember { mutableStateOf<AyahId?>(null) }

                val mic = rememberLauncherForPermission { granted ->
                    if (granted) detector.start() else status = "Mic permission denied"
                }

                LaunchedEffect(Unit) {
                    detector.setListener(object : QuranReciteDetector.Listener {
                        override fun onModelDownloadProgress(fraction: Float) {
                            status = "Downloading model ${(fraction * 100).toInt()}%"
                        }
                        override fun onModelReady() { status = "Ready — tap Listen"; }
                        // The centralized snapshot is all the UI needs: render `active`, and
                        // show the deferral (no highlight, options surfaced) while pending.
                        override fun onHighlightState(state: HighlightState) {
                            current = state.active
                            status = state.pending?.let { p ->
                                val opts = p.options.joinToString(" / ") { "${it.surah}:${it.ayah}" }
                                if (p.reason == PendingReason.NEEDS_CHOICE) "Choose: $opts" else "Deciding… ($opts)"
                            } ?: "Listening…"
                        }
                        override fun onError(error: Throwable) { status = "Error: ${error.message}" }
                    })
                    detector.prepare()
                }

                MushafScreen(
                    status = status,
                    current = current,
                    onListen = { mic.launch(Manifest.permission.RECORD_AUDIO) },
                )
            }
        }
    }

    override fun onDestroy() { detector.release(); super.onDestroy() }
}

@Composable
private fun rememberLauncherForPermission(onResult: (Boolean) -> Unit) =
    androidx.activity.compose.rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(), onResult)

@Composable
private fun MushafScreen(status: String, current: AyahId?, onListen: () -> Unit) {
    // Demo dataset: An-Nas (114) for illustration; a real app renders the mushaf text.
    val surah = 114
    val ayatText = listOf(
        "قُلْ أَعُوذُ بِرَبِّ النَّاسِ",
        "مَلِكِ النَّاسِ",
        "إِلَٰهِ النَّاسِ",
        "مِن شَرِّ الْوَسْوَاسِ الْخَنَّاسِ",
        "الَّذِي يُوَسْوِسُ فِي صُدُورِ النَّاسِ",
        "مِنَ الْجِنَّةِ وَالنَّاسِ",
    )
    Column(Modifier.fillMaxSize().padding(16.dp)) {
        Text("QuranRecite SDK demo", style = MaterialTheme.typography.titleLarge)
        Text(status, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.height(12.dp))
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) {
            ayatText.forEachIndexed { i, text ->
                val isCurrent = current?.surah == surah && current?.ayah == i + 1
                Surface(
                    color = if (isCurrent) MaterialTheme.colorScheme.primaryContainer else Color.Transparent,
                    modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                ) {
                    Text("${i + 1}. $text", Modifier.padding(12.dp),
                        style = MaterialTheme.typography.titleMedium)
                }
            }
        }
        Button(onClick = onListen, Modifier.fillMaxWidth()) { Text("Listen") }
    }
}
