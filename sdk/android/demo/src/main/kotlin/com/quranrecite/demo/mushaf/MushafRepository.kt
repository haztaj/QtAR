package com.quranrecite.demo.mushaf

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.graphics.Typeface
import android.util.LruCache
import java.io.File

/**
 * Reads the KFGQPC V2 glyph mushaf assets bundled under assets/mushaf/:
 *   - layout.db  : `pages` table — per (page, line): line_type, is_centered, word-id range.
 *   - words.db   : `words` table — per word: id (global, 1..83668), surah, ayah, text (the
 *                  page-font PUA glyph, "code_v2").
 *   - fonts/pN.ttf : one font per page; word glyphs are addressed by the PUA codepoints in
 *                    `words.text` and are page-local (each page font redefines the block).
 *
 * Rendering a line = concatenate `words.text` for the line's id range, drawn in that page's
 * font. Highlighting = colour the words whose (surah,ayah) matches. See docs in the demo.
 */
class MushafRepository private constructor(
    private val layout: SQLiteDatabase,
    private val words: SQLiteDatabase,
    private val assetManager: android.content.res.AssetManager,
    private val pageFont: (Int) -> Typeface,   // page fonts: from assets (dev) or downloaded files
) {
    val pageCount: Int by lazy {
        layout.rawQuery("select max(page_number) from pages", null).use {
            if (it.moveToFirst()) it.getInt(0) else 604
        }
    }

    // Page fonts are ~50–200 KB each; keep a small window of recently-used ones resident.
    private val fontCache = object : LruCache<Int, Typeface>(12) {}

    fun typefaceForPage(page: Int): Typeface = fontCache.get(page) ?: run {
        val tf = pageFont(page)
        fontCache.put(page, tf); tf
    }

    /** KFGQPC ornate surah-header font (COLR/CPAL color glyphs) + the "surah-N" -> glyph map. */
    val surahHeaderTypeface: Typeface by lazy {
        Typeface.createFromAsset(assetManager, "mushaf/fonts/surah-header.ttf")
    }
    private val surahHeaderGlyphs: Map<Int, String> by lazy {
        val txt = assetManager.open("mushaf/surah-header-ligatures.json").bufferedReader().use { it.readText() }
        val obj = org.json.JSONObject(txt)
        (1..114).associateWith { obj.optString("surah-$it").trim() }
    }

    /** The single header glyph to render (in [surahHeaderTypeface]) for a surah, or "" if unknown. */
    fun surahHeaderGlyph(surah: Int): String = surahHeaderGlyphs[surah] ?: ""

    /** Compact surah-name font (top bar): render [surahNameGlyph] in this. Ligature "surahNNN". */
    val surahNameTypeface: Typeface by lazy {
        Typeface.createFromAsset(assetManager, "mushaf/fonts/surah-name.ttf")
    }
    fun surahNameGlyph(surah: Int): String = "surah%03d".format(surah)

    /** Common Quran font (top bar juz number): render [juzGlyph] in this. Ligature "juzNNN". */
    val quranCommonTypeface: Typeface by lazy {
        Typeface.createFromAsset(assetManager, "mushaf/fonts/quran-common.ttf")
    }
    fun juzGlyph(juz: Int): String = "juz%03d".format(juz)

    /** The surah at the top of a page (surah of its first word) — for the top-bar name. */
    fun pageTopSurah(page: Int): Int {
        val wid = layout.rawQuery(
            "select min(cast(first_word_id as integer)) from pages " +
                "where page_number=? and first_word_id is not null and first_word_id!=''",
            arrayOf(page.toString()),
        ).use { if (it.moveToFirst() && !it.isNull(0)) it.getInt(0) else return 1 }
        return words.rawQuery("select surah from words where id=?", arrayOf(wid.toString()))
            .use { if (it.moveToFirst()) it.getInt(0) else 1 }
    }

    /** All distinct ayat on a page ("surah:ayah" keys, in page order). Empty if the page has no
     *  words. Used for the last-N end-of-page check and the detector's page-context prior. */
    fun pageAyat(page: Int): List<String> {
        val lo = layout.rawQuery(
            "select min(cast(first_word_id as integer)) from pages " +
                "where page_number=? and first_word_id is not null and first_word_id!=''",
            arrayOf(page.toString()),
        ).use { if (it.moveToFirst() && !it.isNull(0)) it.getInt(0) else return emptyList() }
        val hi = layout.rawQuery(
            "select max(cast(last_word_id as integer)) from pages " +
                "where page_number=? and last_word_id is not null and last_word_id!=''",
            arrayOf(page.toString()),
        ).use { if (it.moveToFirst() && !it.isNull(0)) it.getInt(0) else return emptyList() }
        val keys = ArrayList<String>()
        words.rawQuery(
            "select surah, ayah from words where id between ? and ? " +
                "group by surah, ayah order by min(id)",
            arrayOf(lo.toString(), hi.toString()),
        ).use { c -> while (c.moveToNext()) keys += "${c.getInt(0)}:${c.getInt(1)}" }
        return keys
    }

    /** The last [count] distinct ayat on a page — to detect when the reader nears the page end. */
    fun pageLastAyat(page: Int, count: Int): List<String> = pageAyat(page).takeLast(count)

    /** The juz a page falls in (1..30), from the standard 604-page Madani juz start pages. */
    fun pageJuz(page: Int): Int {
        var j = 1
        for (i in JUZ_START_PAGES.indices) if (page >= JUZ_START_PAGES[i]) j = i + 1
        return j
    }

    fun loadPage(page: Int): MushafPage {
        val lines = ArrayList<MushafLine>()
        layout.rawQuery(
            "select line_number, line_type, is_centered, first_word_id, last_word_id, surah_number " +
                "from pages where page_number=? order by line_number",
            arrayOf(page.toString()),
        ).use { c ->
            while (c.moveToNext()) {
                val lineNo = c.getInt(0)
                val type = when (c.getString(1)) {
                    "surah_name" -> LineType.SURAH_NAME
                    "basmallah" -> LineType.BASMALLAH
                    else -> LineType.AYAH
                }
                val centered = c.getInt(2) == 1
                val first = c.getString(3).toIntOrNull()
                val last = c.getString(4).toIntOrNull()
                val surahNo = c.getString(5).toIntOrNull()
                val ws = if (type == LineType.AYAH && first != null && last != null)
                    wordsInRange(first, last) else emptyList()
                lines += MushafLine(lineNo, type, centered, surahNo, ws)
            }
        }
        return MushafPage(page, lines)
    }

    private fun wordsInRange(first: Int, last: Int): List<WordGlyph> {
        val out = ArrayList<WordGlyph>(last - first + 1)
        words.rawQuery(
            "select id, surah, ayah, text from words where id between ? and ? order by id",
            arrayOf(first.toString(), last.toString()),
        ).use { c ->
            while (c.moveToNext())
                out += WordGlyph(c.getInt(0), c.getInt(1), c.getInt(2), c.getString(3))
        }
        return out
    }

    // ---- word-level highlighting (segment -> word map; see data/build_segment_words.py) ----
    private val segmentWords: Map<String, IntArray> by lazy {
        val txt = assetManager.open("quranrecite/segment_words.json").bufferedReader().use { it.readText() }
        val obj = org.json.JSONObject(txt)
        buildMap {
            for (k in obj.keys()) {
                if (k.startsWith("_")) continue
                val arr = obj.getJSONArray(k)
                put(k, intArrayOf(arr.getInt(0), arr.getInt(1)))
            }
        }
    }
    private val firstWordIds = HashMap<String, Int?>()

    /** Content-word ordinal range [w0, w1) of a unit key ("s:a#NN" or "s:a"), or null. */
    fun segmentWordRange(unitKey: String): IntArray? = segmentWords[unitKey]

    /** Global words.db id of the ayah's first word (word ordinals offset from it). */
    fun firstWordId(surah: Int, ayah: Int): Int? = synchronized(firstWordIds) {
        firstWordIds.getOrPut("$surah:$ayah") {
            words.rawQuery(
                "select min(id) from words where surah=? and ayah=?",
                arrayOf(surah.toString(), ayah.toString()),
            ).use { if (it.moveToFirst() && !it.isNull(0)) it.getInt(0) else null }
        }
    }

    /** Page holding the first word of the given ayah (drives jump + auto-advance). */
    fun pageForAyah(surah: Int, ayah: Int): Int? {
        val wid = words.rawQuery(
            "select min(id) from words where surah=? and ayah=?",
            arrayOf(surah.toString(), ayah.toString()),
        ).use { if (it.moveToFirst() && !it.isNull(0)) it.getInt(0) else return null }
        return layout.rawQuery(
            "select page_number from pages where line_type='ayah' " +
                "and cast(first_word_id as integer)<=? and cast(last_word_id as integer)>=? " +
                "order by page_number limit 1",
            arrayOf(wid.toString(), wid.toString()),
        ).use { if (it.moveToFirst()) it.getInt(0) else null }
    }

    companion object {
        /** Copies the two DBs out of assets (once) and opens them read-only, and wires the page-font
         *  loader to [fonts] (bundled assets for dev, or the downloaded dir). Call off the main thread. */
        fun open(context: Context, fonts: FontSource): MushafRepository {
            val layout = openAssetDb(context, "mushaf/layout.db")
            val words = openAssetDb(context, "mushaf/words.db")
            val pageFont: (Int) -> Typeface = when (fonts) {
                is FontSource.Bundled ->
                    { p -> Typeface.createFromAsset(context.assets, "mushaf/fonts/p$p.ttf") }
                is FontSource.Downloaded ->
                    { p -> Typeface.createFromFile(File(fonts.dir, "p$p.ttf")) }
            }
            return MushafRepository(layout, words, context.assets, pageFont)
        }

        private const val ASSET_VERSION = 2   // bump to force a re-copy when the DBs change

        // Standard 604-page Madani mushaf: first page of each juz (index 0 = juz 1).
        private val JUZ_START_PAGES = intArrayOf(
            1, 22, 42, 62, 82, 102, 121, 142, 162, 182, 201, 222, 242, 262, 282,
            302, 322, 342, 362, 382, 402, 422, 442, 462, 482, 502, 522, 542, 562, 582)

        private fun openAssetDb(context: Context, assetPath: String): SQLiteDatabase {
            val name = assetPath.substringAfterLast('/')
            val out = File(context.filesDir, "mushaf/$name").apply { parentFile?.mkdirs() }
            val stamp = File(out.parentFile, "$name.v")
            // Copy out of assets (once). `assets.open` transparently decompresses — unlike
            // `openFd`, which fails on AAPT-compressed .db assets. Re-copy on version bump.
            if (!out.exists() || stamp.readTextOrNull() != ASSET_VERSION.toString()) {
                context.assets.open(assetPath).use { i -> out.outputStream().use { o -> i.copyTo(o) } }
                stamp.writeText(ASSET_VERSION.toString())
            }
            return SQLiteDatabase.openDatabase(out.path, null, SQLiteDatabase.OPEN_READONLY)
        }

        private fun File.readTextOrNull(): String? = if (exists()) readText() else null
    }
}
