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
import androidx.compose.ui.layout.layout
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
// Safety margin on the auto-fit. The fit measures line widths with TextPaint.measureText, which
// under-measures vs Compose's actual RTL layout, so the widest (justified) line could overflow and
// its glyphs collapse/overlap — worst on wide/landscape screens. 0.90 keeps enough headroom.
private const val FIT = 0.90f
// Extra vertical room per line so a full 15-line page fits without clipping the bottom.
private const val LINE_H_SAFETY = 0.85f
// Each line shrinks its own footprint by this fraction of the ayah font size, absorbing the font's
// built-in vertical padding so lines sit closer (0 = none). Per line type — the ayah page font, the
// basmalah (quran-common) and the surah-name header each have different padding. Tune to taste.
private const val LINE_OVERLAP = .8f        // ayah lines (page font)
private const val BASMALLAH_OVERLAP = 0.0f   // basmalah (quran-common.ttf)
private const val HEADER_OVERLAP = 2.5f       // surah-name header (already trimmed)
private const val HEADER_SCALE = 4.5f           // surah-name glyph size, x the ayah font (3x prior)

/**
 * Renders one mushaf page. Auto-sizes a single font size to fit BOTH the available width
 * (widest line's baked advance) and height (15 line slots), so it re-fits on any
 * screen — foldable postures and orientation changes flow straight through
 * [BoxWithConstraints]. Highlighting recolors word spans without resizing.
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

        // Fit font size to the tighter of width (baked line advance) and height (15 slots), and the
        // rendered content width (widest line at that size) so the chrome can align to the page.
        val (fontPx, contentWidthPx) = remember(page.pageNumber, typeface, headerTypeface, wPx, hPx) {
            val tp = TextPaint().apply { this.typeface = typeface; textSize = 1000f }
            val widest = page.lines.filter { it.type == LineType.AYAH }
                .maxOfOrNull { tp.measureText(lineString(it)) } ?: 1f
            val widthDriven = 1000f * wPx / widest
            val fm = tp.fontMetrics
            val lineH = (fm.bottom - fm.top)
            // Reserve each header's *measured* banner-ink height (the trimmed line box) instead of a
            // full HEADER_SCALE em, so the page fits without leaving big gaps around headers.
            val headerLines = page.lines.count { it.type == LineType.SURAH_NAME }
            val headerBand = page.lines.firstOrNull { it.type == LineType.SURAH_NAME }
                ?.surahNumber?.let { s ->
                    val htp = TextPaint().apply { this.typeface = headerTypeface; textSize = 1000f * HEADER_SCALE }
                    val r = android.graphics.Rect()
                    val g = headerGlyph(s); htp.getTextBounds(g, 0, g.length, r)
                    r.height().toFloat() * 1.2f   // + small breathing margin
                } ?: 0f
            val extraPerHeader = maxOf(0f, headerBand / lineH - 1f)
            val slots = LINES_PER_PAGE + headerLines * extraPerHeader
            val heightDriven = 1000f * hPx / (slots * lineH * LINE_H_SAFETY)
            val fp = minOf(widthDriven, heightDriven) * FIT
            fp to (widest * fp / 1000f)
        }
        val fontSize = (fontPx / density).sp
        LaunchedEffect(contentWidthPx, density) { onContentWidth((contentWidthPx / density).dp) }
        LaunchedEffect(fontSize) { onFontSize(fontSize) }

        Column(
            Modifier.fillMaxSize(),
            // First line at the top, last at the bottom, free space distributed between lines — so
            // every page starts/ends at the same position and sparse pages spread to fill.
            verticalArrangement = Arrangement.SpaceBetween,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            page.lines.forEach { line ->
                MushafLineItem(line, fontSize, family, headerFamily, commonFamily,
                               headerGlyph, highlight, activeBg, upNextBg, optionBg)
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
    headerGlyph: (Int) -> String,
    highlight: HighlightInfo,
    activeBg: Color,
    upNextBg: Color,
    optionBg: Color,
) {
    when (line.type) {
        LineType.SURAH_NAME -> line.surahNumber?.let { s ->
            // Trim the header's line box to the banner ink (no font padding / leading),
            // else a 4.8x em box wraps the short banner glyph in large empty space.
            Text(headerGlyph(s), fontFamily = headerFamily,
                 fontSize = fontSize * HEADER_SCALE, textAlign = TextAlign.Center,
                 maxLines = 1, softWrap = false,
                 modifier = Modifier.tighten((fontSize.value * HEADER_OVERLAP).dp),
                 style = TextStyle(
                     platformStyle = PlatformTextStyle(includeFontPadding = false),
                     lineHeightStyle = LineHeightStyle(
                         alignment = LineHeightStyle.Alignment.Center,
                         trim = LineHeightStyle.Trim.Both)))
        }
        LineType.BASMALLAH -> Text(
            BASMALLAH_GLYPH, fontFamily = commonFamily, fontSize = fontSize,
            textAlign = TextAlign.Center, maxLines = 1, softWrap = false,
            modifier = Modifier.tighten((fontSize.value * BASMALLAH_OVERLAP).dp),
        )
        LineType.AYAH -> Text(
            text = buildAnnotatedString {
                var i = 0
                while (i < line.words.size) {
                    val w = line.words[i]
                    val key = ayahKey(w.surah, w.ayah)
                    val bg = when {
                        key == highlight.upNext -> upNextBg
                        key == highlight.active -> activeBg
                        key in highlight.options -> optionBg
                        else -> Color.Unspecified
                    }
                    // Coalesce the run of consecutive words in this ayah into ONE span
                    // (spaces included) so the highlight is continuous, not per-word boxes.
                    var j = i
                    while (j + 1 < line.words.size &&
                           line.words[j + 1].surah == w.surah && line.words[j + 1].ayah == w.ayah) j++
                    val segIds = highlight.activeWordIds
                    if (key == highlight.active && segIds != null) {
                        // Word-level: the ACTIVE ayah's run, sub-split so the words of the
                        // waqf segment being recited get the darker tint. Sub-runs stay
                        // coalesced (spaces inside) so each region is one continuous box.
                        var k = i
                        while (k <= j) {
                            val inSeg = line.words[k].wordId in segIds
                            var m = k
                            while (m + 1 <= j && (line.words[m + 1].wordId in segIds) == inSeg) m++
                            val sub = buildString {
                                for (q in k..m) { append(line.words[q].glyph); if (q != m) append(" ") }
                                if (m != j) append(" ")   // joint space carries this sub-run's tint
                            }
                            withStyle(SpanStyle(background = if (inSeg) upNextBg else activeBg)) {
                                append(sub)
                            }
                            k = m + 1
                        }
                    } else {
                        val run = buildString {
                            for (k in i..j) { append(line.words[k].glyph); if (k != j) append(" ") }
                        }
                        if (bg != Color.Unspecified) withStyle(SpanStyle(background = bg)) { append(run) }
                        else append(run)
                    }
                    if (j + 1 < line.words.size) append(" ")   // gap between ayat: unhighlighted
                    i = j + 1
                }
            },
            fontFamily = family, fontSize = fontSize,
            textAlign = TextAlign.Center, maxLines = 1, softWrap = false,
            modifier = Modifier.tighten((fontSize.value * LINE_OVERLAP).dp),
        )
    }
}

/**
 * Next-page preview: the first [lineCount] lines of [page], rendered at a fixed [fontSize] (the
 * full page's size, so scale matches) and top-aligned. Height wraps the lines — no clipping, so it
 * unambiguously shows the TOP of the page. Meant to be placed in a bordered overlay.
 */
@Composable
fun MushafPagePreview(
    page: MushafPage,
    typeface: AndroidTypeface,
    headerTypeface: AndroidTypeface,
    commonTypeface: AndroidTypeface,
    headerGlyph: (Int) -> String,
    fontSize: TextUnit,
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
        page.lines.take(lineCount).forEach { line ->
            MushafLineItem(line, fontSize, family, headerFamily, commonFamily,
                           headerGlyph, empty, Color.Unspecified, Color.Unspecified, Color.Unspecified)
        }
    }
}

private fun lineString(line: MushafLine): String =
    line.words.joinToString(" ") { it.glyph }

/** Shrink this composable's laid-out height by [dy] (content stays centred), so neighbours in a
 *  Column move in and overlap it by [dy] — used to absorb a font's built-in vertical padding. */
private fun Modifier.tighten(dy: androidx.compose.ui.unit.Dp) = layout { measurable, constraints ->
    val placeable = measurable.measure(constraints)
    val d = dy.roundToPx()
    val h = (placeable.height - d).coerceAtLeast(0)
    layout(placeable.width, h) { placeable.place(0, -d / 2) }
}

@Composable
private fun LocalDensityValue(): Float = androidx.compose.ui.platform.LocalDensity.current.density
