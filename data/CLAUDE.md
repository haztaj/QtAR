# data/ — dataset prep pipeline

Turns raw HuggingFace datasets into lhotse manifests + a phoneme lexicon for
icefall CTC training. All scripts are idempotent and safe to re-run.

## Run order

```bash
python data/prepare.py          # download parquets, report coverage  -> COVERAGE.md
python data/derive_aya_ids.py   # RetaSy Arabic text -> numeric aya IDs
python data/extract_audio.py    # parquet audio blobs -> mp3 files + manifest.csv
python data/build_manifests.py  # mp3 metadata + splits -> lhotse jsonl.gz + ayah_text.json
python data/build_lexicon.py    # G2P over ayat -> lang/ayah_phonemes.json + tokens.txt

# Waqf segmentation (research: segment-level detection; audition-approved 2026-07-06)
python data/segment_waqf.py     # audition sample: segment wavs + audition.html (by-ear QA)
python data/build_segments.py   # full corpus: lang/segment_phonemes.json + raw/segments/segment_spans.csv
```

`prepare.py` and `derive_aya_ids.py` are independent; everything else is sequential.

## Key facts (verified by the scripts)

- **Primary corpus:** `Buraaq/quran-md-ayahs` — **31,710 clips, 30 Hafs reciters, 153.9 h**
  over surahs 1–3 + Juz Amma (1,057 ayat; was 16,920 clips / 30.8 h / 564 Juz-Amma ayat before
  the 2026-07-05 expansion). Audio is MP3 at **mixed sample rates** (16k / 22050 / 24k / 44.1k /
  48k) — resample to 16 kHz at load time. **Long ayat:** surahs 1–3 add very long verses (max
  2:282 = 878 phonemes vs Juz Amma's ~105; 148 ayat > 150 phonemes) — matters for the training
  frame budget and the fixed-window export/detection tuning (sized for short Juz-Amma ayat).
- **Reciter split** (deterministic, alphabetical): 24 train / 3 val / 3 test.
  val = nasser_alqatami, saood_ash_shuraym, tunaiji;
  test = warsh_husary, warsh_yassin, yasser_ad_dussary.
  Split is **by reciter** so eval measures speaker generalization.
- **Learner set:** `RetaSy/quranic_audio_dataset` — only the short final surahs.
  Aya IDs derived by normalized-Arabic text join against quran-md-ayahs
  (**84.8% matched**, 2235 clips). RetaSy `Surah` is a transliterated *name*
  (Al-Falaq…), `Aya` is Arabic text — neither is numeric. Remaining 15% unmatched
  is multi-ayah recordings + spelling variants across mushaf editions.

## Gotchas baked into the scripts

- **Boundary surah split:** surah 78 spans two parquet shards. `prepare.py` grabs
  one extra shard after first full coverage so it isn't undercounted (was 85, real
  is 1200).
- **Uthmani vs simplified text:** RetaSy uses Uthmani orthography (superscript alef
  U+0670, alef maksura+dagger, Arabic Extended-A tajweed marks) and Persian/Urdu
  keyboard codepoints (U+06CC, U+06A9). `derive_aya_ids.normalize()` reconciles all
  of these. U+0670 is *replaced with* alef, not stripped (it's a real letter there).
- **Windows + MP3:** torchaudio 2.x needs torchcodec/FFmpeg which isn't set up here,
  so `build_manifests.py` reads MP3 headers via **mutagen** (no decode).
- **Arabic in JSONL:** lhotse's writer uses cp1252 on Windows and chokes on Arabic.
  Ayah text is kept out of the supervisions — stored in `manifests/ayah_text.json`
  (UTF-8) and referenced by `ayah_text_key` ("surah:ayah").

## G2P (`quran_g2p.py`)

Deterministic Hafs grapheme→phoneme. 34-phoneme ASCII inventory (Buckwalter-ish).
Models: short vowels, sukun, shadda (gemination), tanwin, long vowels via carrier
letters, dagger alef, hamza forms, ta-marbuta, **definite-article sun/moon
assimilation**, and **waqf** (ayah-final pausal vowel dropping; fathatan→aa). Two
deliberate exceptions: the divine name الله is forced to long aa (unwritten in this
text), and final case-vowels are dropped because clips are per-ayah (reciters pause).

**Not modeled:** cross-word tajweed (idghaam, ikhfaa, iqlab, special madd lengths).
Intentional — the G2P is used for *both* the CTC target and the matcher index, so
internal consistency beats tajweed-perfection, and the fuzzy matcher absorbs the rest.

## Outputs (git-ignored except small text assets)

- `raw/quran-md-ayahs/*.parquet` — source shards (never commit)
- `raw/audio/<reciter>/sNNN_aNNN.mp3` + `raw/audio/manifest.csv` (never commit)
- `raw/retasy_juz_amma_with_ids.parquet`
- `manifests/{train,val,test}_{recordings,supervisions}.jsonl.gz`
- `manifests/ayah_text.json`
- `lang/ayah_phonemes.json`, `lang/tokens.txt`
- `lang/segment_phonemes.json` — waqf-segment refs ("s:a#NN" -> phonemes/text; pausal form at
  segment ends). 1,029 segments over 345 ayat (median 8.5 s — sliding-window-sized units).
- `raw/segments/segment_spans.csv` — per-clip aligned segment boundaries (16 kHz sample offsets)
  via CTC forced alignment with our own model; 30,870 spans over all 10,350 clips, 0 failures.
  Cut on the fly — no segment wav copies. Long-clip forwards need fp16 (fp32 attention on a
  390 s clip is ~2.4 GB/layer and kills the process).
- `COVERAGE.md` (auto-generated per-surah counts)
