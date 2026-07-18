@file:OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)

package com.quranrecite.demo.mushaf

import com.quranrecite.sdk.AyahId

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.Typeface
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private val ARABIC_DIGITS = charArrayOf('٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩')
private fun Int.easternArabic(): String = toString().map { ARABIC_DIGITS[it - '0'] }.joinToString("")

private const val PREVIEW_LINES = 4        // next-page preview: how many top lines to show

/**
 * The mushaf reader. Chrome is a printed-mushaf frame: the top strip shows the page's surah name
 * (left) and juz number (right); the page number sits at the bottom, mirrored by parity (odd →
 * right, even → left). Tapping the page toggles two control panels — top (jump button + a ☰
 * menu of debug controls, right), bottom (status + start/stop on one line) — that slide in over
 * the frame. The page auto-advances to follow the detected ayah (from [highlight]).
 */
@Composable
fun MushafScreen(
    repo: MushafRepository,
    highlight: HighlightInfo,
    status: String,
    modelReady: Boolean,
    listening: Boolean,
    onToggleListen: () -> Unit,
    debugLogging: Boolean,
    onDebugLoggingChange: (Boolean) -> Unit,
    recording: Boolean,
    onRecordingChange: (Boolean) -> Unit,
    blacklist: Boolean,
    onBlacklistChange: (Boolean) -> Unit,
    onShareRecording: () -> Unit,
    onPageContext: (List<AyahId>) -> Unit = {},
    initialPage: Int = 1,                       // 1-based page to open on (last-viewed, persisted)
    onPageChanged: (Int) -> Unit = {},          // called with the 1-based page as the reader settles
    dark: Boolean = false,                      // render the fonts' dark palette (see MushafRepository)
    onDarkChange: (Boolean) -> Unit = {},
) {
    val pagerState = rememberPagerState(
        initialPage = (initialPage - 1).coerceIn(0, repo.pageCount - 1),
        pageCount = { repo.pageCount })
    // Persist the last-viewed page (as it settles) so the reader reopens there next launch.
    LaunchedEffect(pagerState) {
        snapshotFlow { pagerState.settledPage }.collect { onPageChanged(it + 1) }
    }
    val scope = rememberCoroutineScope()
    var showJump by remember { mutableStateOf(false) }
    var chrome by remember { mutableStateOf(false) }   // both control panels visible
    var contentWidth by remember { mutableStateOf(0.dp) }   // rendered mushaf width (chrome aligns to it)
    var pageFontSize by remember { mutableStateOf(0.sp) }   // the page's fitted font size (preview reuses it)

    val surahNameFamily = remember(repo) { FontFamily(Typeface(repo.surahNameTypeface)) }
    val quranCommonFamily = remember(repo) { FontFamily(Typeface(repo.quranCommonTypeface)) }

    // Word-level highlight: resolve the ACTIVE waqf segment's global word-id range from the
    // segment->word map (exact by construction — data/build_segment_words.py). Falls back to
    // the plain whole-ayah highlight whenever unresolved (no segment info, unmapped, etc.).
    val enriched by produceState(highlight, highlight) {
        value = withContext(Dispatchers.IO) {
            val a = highlight.active
            if (a != null && highlight.segmentCount > 1 &&
                highlight.segment in 1..highlight.segmentCount) {
                val range = repo.segmentWordRange(a + "#%02d".format(highlight.segment))
                val parts = a.split(":")
                val first = if (range != null)
                    repo.firstWordId(parts[0].toInt(), parts[1].toInt()) else null
                if (range != null && first != null)
                    highlight.copy(activeWordIds = (first + range[0])..(first + range[1] - 1))
                else highlight
            } else highlight
        }
    }

    val page = pagerState.currentPage + 1
    val topSurah by produceState(1, page) { value = withContext(Dispatchers.IO) { repo.pageTopSurah(page) } }
    val juz = repo.pageJuz(page)

    // Detection page-context prior: as the reader flips, tell the detector which ayat are on the
    // current page + the next one so on-page ayat win twin ambiguities and off-page jumps are
    // suppressed (see QuranReciteDetector.setPageContext / Config.chainPageBonus).
    LaunchedEffect(page, repo) {
        val keys = withContext(Dispatchers.IO) {
            (repo.pageAyat(page) + if (page < repo.pageCount) repo.pageAyat(page + 1) else emptyList())
        }
        onPageContext(keys.distinct().mapNotNull { k ->
            k.split(":").let { if (it.size == 2) AyahId(it[0].toInt(), it[1].toInt()) else null }
        })
    }

    // Next-page preview: once the reciter reaches the last two ayat of the page, peek at the next
    // page's first lines (bordered overlay over the top of the current page). Dismissable per page.
    val lastAyat by produceState<List<String>>(emptyList(), page) {
        value = withContext(Dispatchers.IO) { repo.pageLastAyat(page, 2) }
    }
    var previewDismissed by remember(page) { mutableStateOf(false) }
    val showPreview = listening && !previewDismissed &&
        highlight.active != null && highlight.active in lastAyat && page < repo.pageCount
    val previewData by produceState<Pair<MushafPage, android.graphics.Typeface>?>(null, showPreview, page, dark) {
        value = if (showPreview) withContext(Dispatchers.IO) {
            repo.loadPage(page + 1) to repo.typefaceForPage(page + 1, dark)
        } else null
    }

    // Follow the reciter: when the active ayah changes, page to where it lives.
    LaunchedEffect(highlight.active) {
        val a = highlight.active ?: return@LaunchedEffect
        val (s, y) = a.split(":").map { it.toInt() }
        val pg = withContext(Dispatchers.IO) { repo.pageForAyah(s, y) } ?: return@LaunchedEffect
        if (pg - 1 != pagerState.currentPage) pagerState.animateScrollToPage(pg - 1)
    }

    Box(Modifier.fillMaxSize()) {
        Column(Modifier.fillMaxSize()) {
            // Top strip: surah name (left) + juz (right), aligned to the mushaf page width.
            Box(Modifier.fillMaxWidth().padding(vertical = 2.dp), contentAlignment = Alignment.Center) {
                Row(
                    if (contentWidth > 0.dp) Modifier.width(contentWidth).padding(horizontal = 12.dp)
                    else Modifier.fillMaxWidth().padding(horizontal = 16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(repo.surahNameGlyph(topSurah), fontFamily = surahNameFamily, fontSize = 26.sp,
                         maxLines = 1, softWrap = false)
                    Spacer(Modifier.weight(1f))
                    // Waqf-segment progress of the active ayah (Mode.CHAIN), centered in the strip.
                    if (highlight.segmentCount > 1) {
                        SegmentProgress(highlight.segment, highlight.segmentCount)
                        Spacer(Modifier.weight(1f))
                    }
                    Text(repo.juzGlyph(juz), fontFamily = quranCommonFamily, fontSize = 22.sp,
                         maxLines = 1, softWrap = false)
                }
            }

            // Page area: RTL pager + the parity-placed page number at the bottom.
            BoxWithConstraints(Modifier.weight(1f).fillMaxWidth()) {
                val pageAreaHeight = maxHeight
                CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) {
                    HorizontalPager(state = pagerState, modifier = Modifier.fillMaxSize()) { index ->
                        val loaded by produceState<Pair<MushafPage, android.graphics.Typeface>?>(null, index, dark) {
                            value = withContext(Dispatchers.IO) {
                                repo.loadPage(index + 1) to repo.typefaceForPage(index + 1, dark)
                            }
                        }
                        val data = loaded
                        Box(
                            Modifier.fillMaxSize()
                                .pointerInput(Unit) { detectTapGestures { chrome = !chrome } },
                            contentAlignment = Alignment.Center,
                        ) {
                            if (data == null) CircularProgressIndicator()
                            else MushafPageView(data.first, data.second, enriched,
                                                repo.surahHeaderTypeface(dark), repo.quranCommonTypeface,
                                                repo::surahHeaderGlyph,
                                                onContentWidth = { contentWidth = it },
                                                onFontSize = { pageFontSize = it })
                        }
                    }
                }
                // Page number — odd pages on the right, even on the left, aligned to the page width.
                Box(
                    Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(bottom = 6.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Row(if (contentWidth > 0.dp) Modifier.width(contentWidth).padding(horizontal = 28.dp)
                        else Modifier.fillMaxWidth().padding(horizontal = 28.dp)) {
                        val num: @Composable () -> Unit = {
                            Text(page.easternArabic(), fontSize = 15.sp,
                                 color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        if (page % 2 == 1) { Spacer(Modifier.weight(1f)); num() }
                        else { num(); Spacer(Modifier.weight(1f)) }
                    }
                }

                // Next-page preview overlay — only the next page's first lines, rendered at the SAME
                // font size as the page (so scale matches) and top-aligned. Height wraps the lines.
                previewData?.let { (nextPage, nextTf) ->
                    if (pageFontSize > 0.sp) {
                        val shape = RoundedCornerShape(bottomStart = 14.dp, bottomEnd = 14.dp)
                        Box(
                            Modifier
                                .align(Alignment.TopCenter)
                                .fillMaxWidth()
                                .clip(shape)
                                .background(MaterialTheme.colorScheme.surface)
                                .border(BorderStroke(2.dp, MaterialTheme.colorScheme.primary), shape)
                                .pointerInput(Unit) { detectTapGestures { previewDismissed = true } }
                                .padding(bottom = 4.dp),
                        ) {
                            MushafPagePreview(nextPage, nextTf, repo.surahHeaderTypeface(dark),
                                              repo.quranCommonTypeface, repo::surahHeaderGlyph,
                                              fontSize = pageFontSize, lineCount = PREVIEW_LINES)
                            Text("▾ ${(page + 1).easternArabic()}",
                                 fontSize = 13.sp, color = MaterialTheme.colorScheme.primary,
                                 modifier = Modifier.align(Alignment.BottomCenter))
                        }
                    }
                }
            }
        }

        // Top panel (jump + debug) slides down over the top strip.
        AnimatedVisibility(
            visible = chrome,
            enter = slideInVertically { -it }, exit = slideOutVertically { -it },
            modifier = Modifier.align(Alignment.TopCenter),
        ) {
            TopControls(
                onJump = { showJump = true },
                dark = dark, onDarkChange = onDarkChange,
                debugLogging = debugLogging, onDebugLoggingChange = onDebugLoggingChange,
                recording = recording, onRecordingChange = onRecordingChange,
                blacklist = blacklist, onBlacklistChange = onBlacklistChange,
                onShareRecording = onShareRecording,
            )
        }

        // Bottom panel (detection) slides up over the page number.
        AnimatedVisibility(
            visible = chrome,
            enter = slideInVertically { it }, exit = slideOutVertically { it },
            modifier = Modifier.align(Alignment.BottomCenter),
        ) {
            Surface(tonalElevation = 3.dp, shadowElevation = 8.dp) {
                Row(
                    Modifier.fillMaxWidth().padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(status, style = MaterialTheme.typography.bodyMedium,
                         maxLines = 2, modifier = Modifier.weight(1f))
                    Spacer(Modifier.width(12.dp))
                    Button(
                        onClick = {
                            if (!listening) chrome = false   // hide both panels when starting
                            onToggleListen()
                        },
                        enabled = modelReady,
                    ) {
                        Text(if (listening) "Stop detection" else "Start detection")
                    }
                }
            }
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
private fun TopControls(
    onJump: () -> Unit,
    dark: Boolean, onDarkChange: (Boolean) -> Unit,
    debugLogging: Boolean, onDebugLoggingChange: (Boolean) -> Unit,
    recording: Boolean, onRecordingChange: (Boolean) -> Unit,
    blacklist: Boolean, onBlacklistChange: (Boolean) -> Unit,
    onShareRecording: () -> Unit,
) {
    var menuOpen by remember { mutableStateOf(false) }
    Surface(tonalElevation = 3.dp, shadowElevation = 8.dp) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(onClick = onJump) { Text("Jump to page") }
            Spacer(Modifier.weight(1f))
            // Hamburger (☰) → debug controls menu, anchored top-right.
            Box {
                IconButton(onClick = { menuOpen = true }) {
                    Text("☰", fontSize = 24.sp)
                }
                DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                    DropdownMenuItem(
                        text = { MenuToggle("Dark mode", dark) },
                        onClick = { onDarkChange(!dark) },
                    )
                    DropdownMenuItem(
                        text = { MenuToggle("Debug logging", debugLogging) },
                        onClick = { onDebugLoggingChange(!debugLogging) },
                    )
                    DropdownMenuItem(
                        text = { MenuToggle("Collision blacklist", blacklist) },
                        onClick = { onBlacklistChange(!blacklist) },
                    )
                    DropdownMenuItem(
                        text = { MenuToggle("Record session audio", recording) },
                        onClick = { onRecordingChange(!recording) },
                    )
                    DropdownMenuItem(
                        text = { Text("Share last recording") },
                        onClick = { menuOpen = false; onShareRecording() },
                    )
                }
            }
        }
    }
}

/** Waqf-segment progress of the active ayah — "which part of a long verse am I on". Dots for a
 *  handful of segments; a filled bar + "N/M" for long ayat (many waqf stops) so it stays compact. */
@Composable
private fun SegmentProgress(current: Int, count: Int) {
    val on = MaterialTheme.colorScheme.primary
    val off = MaterialTheme.colorScheme.outlineVariant
    if (count <= 8) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            for (i in 1..count) {
                Box(Modifier.padding(horizontal = 2.dp).size(9.dp).clip(CircleShape)
                    .background(if (i <= current) on else off))
            }
        }
    } else {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.width(70.dp).height(6.dp).clip(RoundedCornerShape(3.dp)).background(off)) {
                Box(Modifier.fillMaxWidth(current.coerceIn(0, count) / count.toFloat())
                    .fillMaxHeight().clip(RoundedCornerShape(3.dp)).background(on))
            }
            Spacer(Modifier.width(6.dp))
            Text("$current/$count", fontSize = 12.sp,
                 color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun MenuToggle(label: String, checked: Boolean) {
    // Display-only switch (onCheckedChange = null) — the DropdownMenuItem's onClick toggles.
    Row(Modifier.width(220.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
        Spacer(Modifier.width(12.dp))
        Switch(checked = checked, onCheckedChange = null)
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
