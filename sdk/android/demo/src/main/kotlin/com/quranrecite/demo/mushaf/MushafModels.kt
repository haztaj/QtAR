package com.quranrecite.demo.mushaf

/** A single rendered word: its glyph text (page-font PUA codepoints) + which ayah it belongs to. */
data class WordGlyph(val wordId: Int, val surah: Int, val ayah: Int, val glyph: String)

enum class LineType { AYAH, SURAH_NAME, BASMALLAH }

/** One line of a mushaf page. AYAH lines carry words; SURAH_NAME/BASMALLAH are centered headers. */
data class MushafLine(
    val lineNumber: Int,
    val type: LineType,
    val centered: Boolean,
    val surahNumber: Int?,        // set for SURAH_NAME lines
    val words: List<WordGlyph>,   // empty for SURAH_NAME / BASMALLAH
)

/** A full mushaf page: its 1-based number and its lines (already broken to mushaf line layout). */
data class MushafPage(val pageNumber: Int, val lines: List<MushafLine>)

/** Ayah key helper — matches the SDK's AyahId string form ("surah:ayah"). */
fun ayahKey(surah: Int, ayah: Int) = "$surah:$ayah"
