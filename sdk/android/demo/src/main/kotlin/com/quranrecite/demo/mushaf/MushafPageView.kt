package com.quranrecite.demo.mushaf

import android.graphics.Typeface as AndroidTypeface
import android.text.TextPaint
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.PlatformTextStyle
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.Typeface
import androidx.compose.ui.text.style.LineHeightStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.zIndex
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** What the detector wants highlighted right now (keys are "surah:ayah"). Two-phase:
 *  [active] is the just-detected ayah (lighter); [upNext] is the predicted next ayah,
 *  shown darker only once [active] nears completion (the one being verified now). */
data class HighlightInfo(
    val active: String? = null,
    val upNext: String? = null,
    val options: Set<String> = emptySet(),
    // Waqf-segment progress within [active] (Mode.CHAIN): "part [segment] of [segmentCount]".
    // segmentCount <= 1 means no sub-ayah progress to show.
    val segment: Int = 0,
    val segmentCount: Int = 0,
    // WORD-LEVEL highlight: global words.db ids of the ACTIVE waqf segment's words (from
    // data/lang/segment_words.json via MushafScreen). Null -> whole-ayah highlight only.
    val activeWordIds: IntRange? = null,
)

private const val BASMALLAH_GLYPH = "﷽"   // U+FDFD, rendered in quran-common.ttf
private const val LINES_PER_PAGE = 15
// Inter-word space advance at textSize=1000. The font's own space glyph is inconsistent (40 on most
// pages, 400 on the 6 fat-space fonts), so every inter-word space is scaled to THIS constant width —
// words separate cleanly on every page and the fat-space lines don't overflow. 40 = the value the
// 597 well-behaved fonts already used, which read fine before.
private const val NORMAL_SPACE = 40f
// The single widest line in the whole Quran at textSize=1000, measured with each page's own font
// (glyph ink + NORMAL_SPACE per inter-word gap). One global number drives one font size + one column
// width for EVERY page, so the surah/juz markers never shift page-to-page.
// Source: scratchpad measure — max 17546.8 (page 585, 15 words); pages cluster ~16.6k–17.5k.
// The constant sits just above that max, so the widest line fills ~the full column and everything
// narrower is inset naturally — that small headroom is the only guard against edge glyph-collapse
// (no extra width margin; a flat 0.9 factor looked too inset in portrait, where width is binding).
private const val GLOBAL_WIDEST = 17600f
// The line-INK height (TextPaint.fontMetrics bottom-top @ textSize=1000) that should fill one of the
// 15 fixed slots. The page is laid out as 15 EQUAL-height slots (not natural-height lines spaced by
// SpaceBetween), so lines always tile the full page height at an even pitch regardless of each
// font's metrics — and the font is sized so a ~median line fills its slot. Rare taller glyphs
// overlap into the neighbouring slot (never clipped — slots always tile the page). Per-page ink
// (on-device scan): p50 2001, p90 2130, max 2531 (page 534). 2100 ≈ p85 → most pages ~95% full,
// only the few tallest overlap a little. Lower = larger text / more overlap.
private const val SLOT_INK = 1800f
// Surah headers are scaled per-line so their ink height == one ayah line (see MushafLineItem), so
// every page is exactly LINES_PER_PAGE slots — no per-page-header vertical band, no size drift.
private const val HEADER_TARGET = 0.9f          // header ink height as a fraction of one line
// Recitation-highlight height. The highlight is a span background behind the ayah glyphs, so its
// height = the ayah line box. By default that hugs the tight font metrics and reads short against
// these tall glyphs; set the line box to this multiple of the font size so the highlight fills more
// of the slot. Glyphs stay centred (LineHeightStyle.Center) — only the highlight height changes.
private const val HIGHLIGHT_LINE_HEIGHT = 1.5f

/**
 * Renders one mushaf page. Sizes the font from ONE global width (the widest ink line in the whole
 * Quran, [GLOBAL_WIDEST]) and the 15 line slots, so EVERY page shares the same font size and column
 * width — the chrome pinned to that column never shifts page-to-page. Still re-fits per viewport, so
 * foldable postures and orientation changes flow straight through [BoxWithConstraints]. Highlighting
 * recolors word spans without resizing.
 */
@Composable
fun MushafPageView(
    page: MushafPage,
    typeface: AndroidTypeface,
    highlight: HighlightInfo,
    headerTypeface: AndroidTypeface,
    commonTypeface: AndroidTypeface,
    headerGlyph: (Int) -> String,
    onContentWidth: (androidx.compose.ui.unit.Dp) -> Unit = {},
    onFontSize: (TextUnit) -> Unit = {},
    onSlotHeight: (androidx.compose.ui.unit.Dp) -> Unit = {},
    modifier: Modifier = Modifier,
) {
    val family = remember(typeface) { FontFamily(Typeface(typeface)) }
    val headerFamily = remember(headerTypeface) { FontFamily(Typeface(headerTypeface)) }
    val commonFamily = remember(commonTypeface) { FontFamily(Typeface(commonTypeface)) }
    val activeBg = MaterialTheme.colorScheme.primary.copy(alpha = 0.22f)   // detected (lighter)
    val upNextBg = MaterialTheme.colorScheme.primary.copy(alpha = 0.48f)   // being verified (darker)
    val optionBg = MaterialTheme.colorScheme.tertiary.copy(alpha = 0.22f)

    BoxWithConstraints(modifier.fillMaxSize().padding(start = 14.dp, end = 14.dp, top=2.dp, bottom = 30.dp)) {
        val density = LocalDensityValue()
        val wPx = maxWidth.value * density
        val hPx = maxHeight.value * density

        // One global font size = the smaller of: the width fit (widest Quran line fills the column)
        // and the slot fit (a median line fills one of the 15 equal slots). BOTH use corpus-wide
        // constants, so fp is IDENTICAL on every page — the column, and the chrome pinned to it,
        // never shift. Keyed on the viewport only, so it re-fits on foldable/orientation changes.
        val (fontPx, frameWidthPx) = remember(wPx, hPx) {
            val widthDriven = 1000f * wPx / GLOBAL_WIDEST
            val slotDriven = 1000f * (hPx / LINES_PER_PAGE) / SLOT_INK
            val fp = minOf(widthDriven, slotDriven)
            // The fixed column frame (widest-line width at this size) — steady across pages.
            fp to (GLOBAL_WIDEST * fp / 1000f)
        }
        val fontSize = (fontPx / density).sp
        val slotHeight = maxHeight / LINES_PER_PAGE       // the pitch of one of the 15 line slots
        LaunchedEffect(frameWidthPx, density) { onContentWidth((frameWidthPx / density).dp) }
        LaunchedEffect(fontSize) { onFontSize(fontSize) }
        LaunchedEffect(slotHeight) { onSlotHeight(slotHeight) }

        // 15 EQUAL-height slots (weight 1f each) that tile the page — lines sit at an even pitch and
        // always fill the full height, independent of each font's metric quirks. Each line's glyphs
        // are centred in their slot; the rare tall glyph overflows into the neighbouring slot's gap
        // (never clipped — the slots always sum to the page). zIndex descends with the line index so
        // each line PAINTS OVER the one below it: when boxes overlap, an upper line's low glyphs are
        // drawn on top of the lower line's (empty) top padding instead of being covered by it.
        // Full pages (602 of them) tile the height with 15 equal slots. The two short opening pages
        // (Al-Fatiha, Al-Baqarah's start — 8 lines each) instead use normal-pitch slots CENTRED
        // vertically, so their few lines sit as a centred block rather than spread over the whole page.
        val shortPage = page.lines.size < LINES_PER_PAGE
        Column(Modifier.fillMaxSize(),
               verticalArrangement = if (shortPage) Arrangement.Center else Arrangement.Top) {
            page.lines.forEachIndexed { index, line ->
                val slot = if (shortPage) Modifier.height(slotHeight) else Modifier.weight(1f)
                Box(Modifier.fillMaxWidth().then(slot).zIndex(-index.toFloat()),
                    contentAlignment = Alignment.Center) {
                    // unbounded height: let the line keep its full natural height and OVERFLOW its
                    // slot rather than being clipped to it (tall marks/descenders would otherwise be
                    // cut off at the slot edge). The zIndex above makes the upper line win the overlap.
                    MushafLineItem(line, fontSize, family, headerFamily, commonFamily,
                                   typeface, headerTypeface, headerGlyph, highlight, activeBg, upNextBg, optionBg,
                                   modifier = Modifier.wrapContentHeight(unbounded = true))
                }
            }
        }
    }
}

/** A single mushaf line (surah header / basmalah / ayah), rendered at [fontSize]. Shared by the
 *  full page and the next-page preview so both use identical styling. */
@Composable
private fun MushafLineItem(
    line: MushafLine,
    fontSize: TextUnit,
    family: FontFamily,
    headerFamily: FontFamily,
    commonFamily: FontFamily,
    ayahTypeface: AndroidTypeface,
    headerTypeface: AndroidTypeface,
    headerGlyph: (Int) -> String,
    highlight: HighlightInfo,
    activeBg: Color,
    upNextBg: Color,
    optionBg: Color,
    modifier: Modifier = Modifier,
) {
    when (line.type) {
        LineType.SURAH_NAME -> line.surahNumber?.let { s ->
            // Scale the banner so its ink height == one ayah line (HEADER_TARGET of it), measured
            // from this glyph's own bounds — it fills exactly one of the 15 slots on every page (no
            // oversized banner, no per-page vertical band). Trim the line box to the banner ink.
            val headerScale = remember(ayahTypeface, headerTypeface, s) {
                val tp = TextPaint().apply { this.typeface = ayahTypeface; textSize = 1000f }
                val fm = tp.fontMetrics; val lineH = fm.bottom - fm.top
                val htp = TextPaint().apply { this.typeface = headerTypeface; textSize = 1000f }
                val r = android.graphics.Rect(); val g = headerGlyph(s)
                htp.getTextBounds(g, 0, g.length, r)
                if (r.height() > 0) HEADER_TARGET * lineH / r.height() else 1f
            }
            Text(headerGlyph(s), fontFamily = headerFamily,
                 fontSize = fontSize * headerScale, textAlign = TextAlign.Center,
                 maxLines = 1, softWrap = false, modifier = modifier,
                 style = TextStyle(
                     platformStyle = PlatformTextStyle(includeFontPadding = false),
                     lineHeightStyle = LineHeightStyle(
                         alignment = LineHeightStyle.Alignment.Center,
                         trim = LineHeightStyle.Trim.Both)))
        }
        LineType.BASMALLAH -> Text(
            BASMALLAH_GLYPH, fontFamily = commonFamily, fontSize = fontSize,
            textAlign = TextAlign.Center, maxLines = 1, softWrap = false, modifier = modifier,
        )
        LineType.AYAH -> {
            // Scale this page-font's space glyph to the constant NORMAL_SPACE so words separate the
            // same amount everywhere (the 6 fat-space fonts otherwise blow the gaps out ~10x).
            val spaceScale = remember(ayahTypeface) {
                val tp = TextPaint().apply { this.typeface = ayahTypeface; textSize = 1000f }
                val sa = tp.measureText(" ")
                if (sa > 0f) NORMAL_SPACE / sa else 1f
            }
            val segIds = highlight.activeWordIds
            // Highlight tint for one word: within the ACTIVE ayah the waqf segment's words go darker
            // (word-level); otherwise the whole ayah shares one tint (active / up-next / option).
            fun bgOf(wd: WordGlyph): Color {
                val key = ayahKey(wd.surah, wd.ayah)
                if (key == highlight.active && segIds != null)
                    return if (wd.wordId in segIds) upNextBg else activeBg
                return when {
                    key == highlight.upNext -> upNextBg
                    key == highlight.active -> activeBg
                    key in highlight.options -> optionBg
                    else -> Color.Unspecified
                }
            }
            Text(
                text = buildAnnotatedString {
                    val ws = line.words
                    for (k in ws.indices) {
                        val bg = bgOf(ws[k])
                        if (bg != Color.Unspecified) withStyle(SpanStyle(background = bg)) { append(ws[k].glyph) }
                        else append(ws[k].glyph)
                        if (k + 1 < ws.size) {
                            // The inter-word space, normalized in width and tinted only when both
                            // neighbours share a highlight so a run reads as one continuous box.
                            val nbg = bgOf(ws[k + 1])
                            val sbg = if (bg == nbg) bg else Color.Unspecified
                            withStyle(SpanStyle(background = sbg, fontSize = fontSize * spaceScale)) { append(" ") }
                        }
                    }
                },
                fontFamily = family, fontSize = fontSize,
                textAlign = TextAlign.Center, maxLines = 1, softWrap = false, modifier = modifier,
                // Taller line box so the span-background highlight reads consistently with the
                // glyphs; centred + untrimmed so the glyphs stay put and the fill spans the height.
                style = TextStyle(
                    lineHeight = fontSize * HIGHLIGHT_LINE_HEIGHT,
                    platformStyle = PlatformTextStyle(includeFontPadding = false),
                    lineHeightStyle = LineHeightStyle(
                        alignment = LineHeightStyle.Alignment.Center,
                        trim = LineHeightStyle.Trim.None)),
            )
        }
    }
}

/**
 * Next-page preview: the first [lineCount] lines of [page], rendered at the full page's [fontSize]
 * and [slotHeight] pitch — the SAME fixed-slot layout as the live page (equal-height slots, glyphs
 * centred and unbounded, upper line drawn on top) so it lines up exactly with the page underneath.
 * Top-aligned; height = lineCount slots. Meant to be placed in a bordered overlay.
 */
@Composable
fun MushafPagePreview(
    page: MushafPage,
    typeface: AndroidTypeface,
    headerTypeface: AndroidTypeface,
    commonTypeface: AndroidTypeface,
    headerGlyph: (Int) -> String,
    fontSize: TextUnit,
    slotHeight: androidx.compose.ui.unit.Dp,
    lineCount: Int,
    modifier: Modifier = Modifier,
) {
    val family = remember(typeface) { FontFamily(Typeface(typeface)) }
    val headerFamily = remember(headerTypeface) { FontFamily(Typeface(headerTypeface)) }
    val commonFamily = remember(commonTypeface) { FontFamily(Typeface(commonTypeface)) }
    val empty = HighlightInfo()
    Column(
        modifier.fillMaxWidth().padding(start = 14.dp, end = 14.dp, top = 2.dp, bottom = 6.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        page.lines.take(lineCount).forEachIndexed { index, line ->
            Box(Modifier.fillMaxWidth().height(slotHeight).zIndex(-index.toFloat()),
                contentAlignment = Alignment.Center) {
                MushafLineItem(line, fontSize, family, headerFamily, commonFamily,
                               typeface, headerTypeface, headerGlyph, empty,
                               Color.Unspecified, Color.Unspecified, Color.Unspecified,
                               modifier = Modifier.wrapContentHeight(unbounded = true))
            }
        }
    }
}

@Composable
private fun LocalDensityValue(): Float = androidx.compose.ui.platform.LocalDensity.current.density
