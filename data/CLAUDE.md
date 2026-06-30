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
```

`prepare.py` and `derive_aya_ids.py` are independent; everything else is sequential.

## Key facts (verified by the scripts)

- **Primary corpus:** `Buraaq/quran-md-ayahs` — 16,920 Juz Amma clips, 30 Hafs
  reciters, **30.8 h** total. Audio is MP3 at **mixed sample rates** (16k / 22050 /
  24k / 44.1k / 48k) — icefall must resample to 16 kHz at load time.
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
- `COVERAGE.md` (auto-generated per-surah counts)
