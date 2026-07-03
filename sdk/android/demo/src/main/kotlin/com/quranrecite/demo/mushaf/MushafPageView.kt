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
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.Typeface
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** What the detector wants highlighted right now (keys are "surah:ayah"). */
data class HighlightInfo(
    val active: String? = null,
    val confirmed: Set<String> = emptySet(),
    val options: Set<String> = emptySet(),
)

private const val BASMALLAH_GLYPH = "ﭐ"   // page fonts map this codepoint to the basmalah glyph
private const val LINES_PER_PAGE = 15
private const val FIT = 0.98f                   // leave a hair of margin so nothing clips

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
    modifier: Modifier = Modifier,
) {
    val family = remember(typeface) { FontFamily(Typeface(typeface)) }
    val activeBg = MaterialTheme.colorScheme.primary.copy(alpha = 0.30f)
    val confirmedBg = MaterialTheme.colorScheme.secondary.copy(alpha = 0.14f)
    val optionBg = MaterialTheme.colorScheme.tertiary.copy(alpha = 0.22f)

    BoxWithConstraints(modifier.fillMaxSize().padding(horizontal = 14.dp, vertical = 8.dp)) {
        val density = LocalDensityValue()
        val wPx = maxWidth.value * density
        val hPx = maxHeight.value * density

        // Fit font size to the tighter of width (baked line advance) and height (15 slots).
        val fontPx = remember(page.pageNumber, typeface, wPx, hPx) {
            val tp = TextPaint().apply { this.typeface = typeface; textSize = 1000f }
            val widest = page.lines.filter { it.type == LineType.AYAH }
                .maxOfOrNull { tp.measureText(lineString(it)) } ?: 1f
            val widthDriven = 1000f * wPx / widest
            val fm = tp.fontMetrics
            val lineH = (fm.bottom - fm.top)
            val heightDriven = 1000f * hPx / (LINES_PER_PAGE * lineH)
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
                    LineType.SURAH_NAME -> SurahHeader(line.surahNumber)
                    LineType.BASMALLAH -> Text(
                        BASMALLAH_GLYPH, fontFamily = family, fontSize = fontSize,
                        textAlign = TextAlign.Center, maxLines = 1, softWrap = false,
                    )
                    LineType.AYAH -> Text(
                        text = buildAnnotatedString {
                            line.words.forEachIndexed { i, w ->
                                val key = ayahKey(w.surah, w.ayah)
                                val bg = when {
                                    key == highlight.active -> activeBg
                                    key in highlight.confirmed -> confirmedBg
                                    key in highlight.options -> optionBg
                                    else -> Color.Unspecified
                                }
                                if (bg != Color.Unspecified) withStyle(SpanStyle(background = bg)) { append(w.glyph) }
                                else append(w.glyph)
                                if (i != line.words.lastIndex) append(" ")
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
private fun SurahHeader(surahNumber: Int?) {
    // Cosmetic placeholder — a real mushaf frames the Arabic surah name (KFGQPC surah-name
    // font, not bundled here). The ayah text below is the authoritative content.
    Box(
        Modifier.fillMaxWidth(0.6f)
            .background(MaterialTheme.colorScheme.surfaceVariant, MaterialTheme.shapes.small),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            "﴿ Sūrah ${surahNumber ?: "?"} ﴾",
            Modifier.padding(vertical = 4.dp),
            style = MaterialTheme.typography.titleSmall,
        )
    }
}

@Composable
private fun LocalDensityValue(): Float = androidx.compose.ui.platform.LocalDensity.current.density
