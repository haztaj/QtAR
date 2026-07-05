package com.quranrecite.demo.mushaf

import android.graphics.Typeface as AndroidTypeface
import android.text.TextPaint
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
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
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** What the detector wants highlighted right now (keys are "surah:ayah"). Two-phase:
 *  [active] is the just-detected ayah (lighter); [upNext] is the predicted next ayah,
 *  shown darker only once [active] nears completion (the one being verified now). */
data class HighlightInfo(
    val active: String? = null,
    val upNext: String? = null,
    val options: Set<String> = emptySet(),
)

private const val BASMALLAH_GLYPH = "ﭐ"   // page fonts map this codepoint to the basmalah glyph
private const val LINES_PER_PAGE = 15
private const val FIT = 0.98f                   // leave a hair of margin so nothing clips
private const val HEADER_SCALE = 4.8f           // surah-name glyph size, x the ayah font (3x prior)

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
    headerGlyph: (Int) -> String,
    modifier: Modifier = Modifier,
) {
    val family = remember(typeface) { FontFamily(Typeface(typeface)) }
    val headerFamily = remember(headerTypeface) { FontFamily(Typeface(headerTypeface)) }
    val activeBg = MaterialTheme.colorScheme.primary.copy(alpha = 0.22f)   // detected (lighter)
    val upNextBg = MaterialTheme.colorScheme.primary.copy(alpha = 0.48f)   // being verified (darker)
    val optionBg = MaterialTheme.colorScheme.tertiary.copy(alpha = 0.22f)

    BoxWithConstraints(modifier.fillMaxSize().padding(horizontal = 14.dp, vertical = 8.dp)) {
        val density = LocalDensityValue()
        val wPx = maxWidth.value * density
        val hPx = maxHeight.value * density

        // Fit font size to the tighter of width (baked line advance) and height (15 slots).
        val fontPx = remember(page.pageNumber, typeface, headerTypeface, wPx, hPx) {
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
            val heightDriven = 1000f * hPx / (slots * lineH)
            minOf(widthDriven, heightDriven) * FIT
        }
        val fontSize = (fontPx / density).sp

        Column(
            Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.SpaceEvenly,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            page.lines.forEach { line ->
                when (line.type) {
                    LineType.SURAH_NAME -> line.surahNumber?.let { s ->
                        // Trim the header's line box to the banner ink (no font padding / leading),
                        // else a 4.8x em box wraps the short banner glyph in large empty space.
                        Text(headerGlyph(s), fontFamily = headerFamily,
                             fontSize = fontSize * HEADER_SCALE, textAlign = TextAlign.Center,
                             maxLines = 1, softWrap = false,
                             style = TextStyle(
                                 platformStyle = PlatformTextStyle(includeFontPadding = false),
                                 lineHeightStyle = LineHeightStyle(
                                     alignment = LineHeightStyle.Alignment.Center,
                                     trim = LineHeightStyle.Trim.Both)))
                    }
                    LineType.BASMALLAH -> Text(
                        BASMALLAH_GLYPH, fontFamily = family, fontSize = fontSize,
                        textAlign = TextAlign.Center, maxLines = 1, softWrap = false,
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
                                val run = buildString {
                                    for (k in i..j) { append(line.words[k].glyph); if (k != j) append(" ") }
                                }
                                if (bg != Color.Unspecified) withStyle(SpanStyle(background = bg)) { append(run) }
                                else append(run)
                                if (j + 1 < line.words.size) append(" ")   // gap between ayat: unhighlighted
                                i = j + 1
                            }
                        },
                        fontFamily = family, fontSize = fontSize,
                        textAlign = TextAlign.Center, maxLines = 1, softWrap = false,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }
        }
    }
}

private fun lineString(line: MushafLine): String =
    line.words.joinToString(" ") { it.glyph }

@Composable
private fun LocalDensityValue(): Float = androidx.compose.ui.platform.LocalDensity.current.density
