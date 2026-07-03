@file:OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)

package com.quranrecite.demo.mushaf

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * The mushaf reader: a right-to-left page pager, a jump-to-page control, and a start/stop
 * detection button. The page auto-advances to follow the detected ayah (from [highlight]).
 */
@Composable
fun MushafScreen(
    repo: MushafRepository,
    highlight: HighlightInfo,
    status: String,
    modelReady: Boolean,
    listening: Boolean,
    onToggleListen: () -> Unit,
) {
    val pagerState = rememberPagerState(pageCount = { repo.pageCount })
    val scope = rememberCoroutineScope()
    var showJump by remember { mutableStateOf(false) }

    // Follow the reciter: when the active ayah changes, page to where it lives.
    LaunchedEffect(highlight.active) {
        val a = highlight.active ?: return@LaunchedEffect
        val (s, y) = a.split(":").map { it.toInt() }
        val pg = withContext(Dispatchers.IO) { repo.pageForAyah(s, y) } ?: return@LaunchedEffect
        if (pg - 1 != pagerState.currentPage) pagerState.animateScrollToPage(pg - 1)
    }

    Column(Modifier.fillMaxSize()) {
        // Top bar — page indicator + jump.
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Page ${pagerState.currentPage + 1} / ${repo.pageCount}",
                style = MaterialTheme.typography.labelLarge)
            Spacer(Modifier.weight(1f))
            TextButton(onClick = { showJump = true }) { Text("Jump to page") }
        }

        // Pages — RTL so it reads like a physical mushaf (swipe right→left to advance).
        CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) {
            HorizontalPager(state = pagerState, modifier = Modifier.weight(1f).fillMaxWidth()) { index ->
                val loaded by produceState<Pair<MushafPage, android.graphics.Typeface>?>(null, index) {
                    value = withContext(Dispatchers.IO) {
                        repo.loadPage(index + 1) to repo.typefaceForPage(index + 1)
                    }
                }
                val data = loaded
                if (data == null) {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator()
                    }
                } else {
                    MushafPageView(data.first, data.second, highlight)
                }
            }
        }

        // Bottom — status + start/stop detection.
        Column(Modifier.fillMaxWidth().padding(12.dp)) {
            Text(status, style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(6.dp))
            Button(
                onClick = onToggleListen,
                enabled = modelReady,
                modifier = Modifier.fillMaxWidth(),
            ) { Text(if (listening) "Stop detection" else "Start detection") }
        }
    }

    if (showJump) {
        JumpDialog(
            max = repo.pageCount,
            onDismiss = { showJump = false },
            onJump = { pg -> showJump = false; scope.launch { pagerState.scrollToPage(pg - 1) } },
        )
    }
}

@Composable
private fun JumpDialog(max: Int, onDismiss: () -> Unit, onJump: (Int) -> Unit) {
    var text by remember { mutableStateOf("") }
    val pg = text.toIntOrNull()
    val valid = pg != null && pg in 1..max
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Jump to page") },
        text = {
            OutlinedTextField(
                value = text,
                onValueChange = { text = it.filter(Char::isDigit).take(3) },
                label = { Text("Page (1–$max)") },
                singleLine = true,
                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                    keyboardType = KeyboardType.Number),
            )
        },
        confirmButton = { TextButton(enabled = valid, onClick = { onJump(pg!!) }) { Text("Go") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}
