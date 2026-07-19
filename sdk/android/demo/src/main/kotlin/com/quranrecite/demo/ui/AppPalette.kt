package com.quranrecite.demo.ui

import androidx.compose.material3.ColorScheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color

/**
 * Single source of truth for the app's color palette — the design-spec "Design Tokens"
 * (see sdk/design/UI_SPEC.md). Edit the [LightPalette] / [DarkPalette] values below to
 * retheme the whole app; nothing else hard-codes chrome colors.
 *
 * IMPORTANT — the Quran glyph text is NOT coloured by this palette. The ayah body, the
 * surah-name banners, and the ayah-end markers are drawn from the KFGQPC fonts' own
 * COLR/CPAL colour palette (light / dark, patched in [MushafRepository]); those `Text`
 * composables set no `color=`. So the spec's `ink` token, and the "header / ayah markers"
 * uses the spec lists under `accent`, are deliberately left OFF the glyphs. This palette
 * only drives the app CHROME: page/preview backgrounds, the recitation highlight, the
 * progress bar, the page number, borders, and (later) the preview depth cues.
 */
data class AppPalette(
    val paper: Color,     // current page background
    val paper2: Color,    // secondary paper
    val preview: Color,   // preview band background (distinct from paper)
    val edge: Color,      // paper edge tone
    val ink: Color,       // general chrome text (NOT the Quran glyphs — font-coloured)
    val accent: Color,    // chrome accent, progress, highlight base
    val accent2: Color,   // recitation highlight, progress track
    val seam: Color,      // seam cast shadow + top hint (depth cue)
    val edgeHi: Color,    // lit top edge of the current page (depth cue)
    val chrome: Color,    // page number
)

/** Light mode — spec "Design Tokens · Light mode". */
val LightPalette = AppPalette(
    paper   = Color(0xFFF6F2E8),
    paper2  = Color(0xFFEFE9DB),
    preview = Color(0xFFEEE7D7),
    edge    = Color(0xFFE2D9C4),
    ink     = Color(0xFF2C2823),
    accent  = Color(0xFF6F61A8),
    accent2 = Color(0xFFEBE7F6),
    seam    = Color(0x33463414),   // rgba(70,52,20,.20)
    edgeHi  = Color(0x8CFFFFFF),   // rgba(255,255,255,.55)
    chrome  = Color(0xFF5B5348),
)

/** Dark mode — spec "Design Tokens · Dark mode". */
val DarkPalette = AppPalette(
    paper   = Color(0xFF17151D),
    paper2  = Color(0xFF211F2B),
    preview = Color(0xFF2A2736),
    edge    = Color(0xFF3A3648),
    ink     = Color(0xFFE9E3D5),
    accent  = Color(0xFFA99CE4),
    accent2 = Color(0x38A99CE4),   // rgba(169,156,228,.22)
    seam    = Color(0xD1000000),   // rgba(0,0,0,.82)
    edgeHi  = Color(0x6BBEB4DE),   // rgba(190,180,222,.42)
    chrome  = Color(0xFF8A8496),
)

/**
 * Map the palette onto the Material 3 [ColorScheme] so stock M3 components and existing
 * `MaterialTheme.colorScheme.*` references pick up the tokens. Only the chrome-relevant
 * slots are overridden; the rest keep sensible M3 defaults.
 */
fun AppPalette.toColorScheme(dark: Boolean): ColorScheme {
    val base = if (dark) darkColorScheme() else lightColorScheme()
    return base.copy(
        primary = accent,                                   // highlight base, progress, borders, buttons
        onPrimary = if (dark) paper else Color.White,       // button label on accent
        tertiary = accent,                                  // ambiguity "option" highlight base
        background = paper,
        onBackground = ink,
        surface = paper,
        onSurface = ink,
        surfaceVariant = paper2,
        onSurfaceVariant = chrome,                          // page number, progress label, dialog subtitle
        outlineVariant = accent2,                           // progress track (off)
    )
}

/** App-wide access to the raw palette tokens that have no Material slot (preview band, depth cues). */
val LocalAppPalette = staticCompositionLocalOf { LightPalette }
