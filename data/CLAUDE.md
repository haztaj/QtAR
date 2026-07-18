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

# RetaSy learner-data cleanup (junk/mislabel triage; feeds make_phase2_splits)
python data/retasy_flag.py      # auto-flag -> raw/retasy_audio/flags.csv (buckets + signals)
python data/retasy_review.py    # by-ear review page -> raw/retasy_audio/review.html
#   review, Download verdicts, save to data/retasy_verdicts.json, then re-run splits
```

`prepare.py` and `derive_aya_ids.py` are independent; everything else is sequential.

## Key facts (verified by the scripts)

- **Primary corpus:** `Buraaq/quran-md-ayahs` — **187,080 clips, 30 Hafs reciters, 984.4 h**
  over the **FULL QURAN (all 114 surahs, 6,236 ayat)** as of the 2026-07-13 expansion (was
  31,710 clips / 153.9 h / 1,057 ayat for surahs 1-3 + Juz Amma; 16,920 / 30.8 h / 564 ayat
  Juz-Amma-only). All 71 shards are downloaded and coverage is complete (0 INCOMPLETE surahs
  in `prepare.py`'s per-surah check). Audio is MP3 at **mixed sample rates** (now 8k / 11025 /
  12k / 16k / 22050 / 24k / 32k / 44.1k / 48k) — resample to 16 kHz at load time. **Long ayat:**
  the full Quran has very long verses (max 2:282 = 878 phonemes vs Juz Amma's ~105; mean 83.2) —
  matters for the training frame budget and the fixed-window export/detection tuning (originally
  sized for short Juz-Amma ayat).
- **Reciter split** (deterministic, alphabetical): 24 train / 3 val / 3 test.
  val = nasser_alqatami, saood_ash_shuraym, tunaiji;
  test = warsh_husary, warsh_yassin, yasser_ad_dussary.
  Split is **by reciter** so eval measures speaker generalization.
- **Primary learner set:** `tusers` (full-Quran). Source:
  **https://archive.org/download/quran-speech-dataset** (licensing cleared, user-confirmed
  2026-07-16). 17,837 clips / 17,811 distinct voices / 45.6 h, all 114 surahs, 93.2% of ayat
  (median 3 clips/ayah). Raw CSV `data/raw/tusers/tusers_filtered.csv` → manifest
  `data/raw/phase2/tusers_manifest.csv` (columns: recording_id, reciter_id `tuser_<uid>`,
  surah_id, ayah_id, path, duration=(filesize-44)/32000, source="tusers"). Mixed at 8.5% into
  `data/raw/phase3/combined_train_tusers.csv` to train the shipped `best_full_tu`. Audio not committed.
- **Legacy learner set:** `RetaSy/quranic_audio_dataset` — only the short final surahs.
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

## RetaSy cleanup (`retasy_flag.py` -> `retasy_review.py` -> `make_phase2_splits.py`)

RetaSy is crowd-sourced learner audio: many clips are silence, mic-noise, false starts, or
recitations of a DIFFERENT ayah than labeled. Only 252 of 2,235 carry a human `final_label`;
the other 1,983 were unjudged (counted as misses in the learner eval, poisoned training).

- **`retasy_flag.py`** — one model pass, three signal tiers: energy/VAD (silent / noise_only /
  too_short), decode sanity (phoneme count), and infix cost of the `best_s123_mic` decode vs
  (a) the LABELED ayah and (b) the whole unit index. Buckets each clip: ok / silent /
  noise_only / too_short / borderline / garbage / **possible_mislabel** (decode matches a
  DIFFERENT ayah better — recoverable, suggests the correction). Writes `raw/retasy_audio/
  flags.csv` + a distribution and a cross-check vs the 252 human labels (validation: 0–8%
  false-flag on `correct`, ~94–100% catch on `not_related_quran`/`not_match_aya`).
  **Full-corpus result (2026-07-08, best_s123_mic): 70% ok, ~30% flagged** (noise_only 11%,
  too_short 5.5%, borderline 4.8%, garbage 4.4%, silent 3.2%, possible_mislabel 1.1%).
- **`retasy_review.py`** — static by-ear page (waqf-audition pattern, `file://` audio, no
  copies). Full-review band = possible_mislabel + borderline + too_short (~255 clips, ~1 h;
  too_short shown in full because that's where good-but-brief learner clips leak into
  auto-discard). Extremes (ok=keep, silent/garbage/noise_only=discard) are pre-verdicted with
  a spot sample surfaced. Exports `review_verdicts.json`.
- Save verdicts to **`data/retasy_verdicts.json`** (committed — reproducible cleanup, like
  `BAD_LABELS`; audio stays uncommitted). `make_phase2_splits.py` resolves ONE decision per
  clip with human priority: relabel > explicit discard > explicit keep > baseline
  (BAD_LABELS ∪ flag auto-discard buckets). The reciter split is computed on the STABLE
  BAD_LABELS-filtered universe so the holdout is unchanged by cleaning (comparable to prior
  runs). A human keep/relabel overrides the old `final_label` blacklist — a `not_match_aya`
  clip the reviewer re-identified is rescued, not dropped.

**Applied (2026-07-08):** 2,235 → 1,678 clips (557 dropped, 8 relabeled); cleaned
`retasy_test.csv` = 530 clips / 57 reciters (35 reciters were entirely junk). **The dirty
test set deflated the learner number ~16 pts** — `best_s123_mic` reads 66.0% dirty vs 81.9%
cleaned (unjudgeable clips counted as misses). **Use the cleaned `retasy_test.csv` as the
learner reference.** Retrain on cleaned data (`best_s123_mic_clean.pt`) → 83.8% learner /
96.0% clean (RetaSy is ~4% of training, so the retrain effect is small; the eval correction
is the win).

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
- `lang/short_unit_blacklist.json` — **committed** collision blacklist: the 110 decoder units with
  collision `eff>100` (18 of them basmala-matchers), cold-fire-suppressed by the chain decoder so
  they fire only with page/sequence context. Generated by `research/collision_rank.py` (which ranks
  every unit by how many ayah contexts + the basmala fuzzy-match it); copied into
  `conformance/assets/` by `conformance/generate.py` and bundled into the `.aar` by the gradle copy
  task. Worst offenders are short COMMON phrases (كلّا, قل الله, بلى) and basmala-matchers (55:1),
  NOT the muqattaʿāt — see research/CLAUDE.md "collision_rank.py".
- `raw/segments/segment_spans.csv` — per-clip aligned segment boundaries (16 kHz sample offsets)
  via CTC forced alignment with our own model; 30,870 spans over all 10,350 clips, 0 failures.
  Cut on the fly — no segment wav copies. Long-clip forwards need fp16 (fp32 attention on a
  390 s clip is ~2.4 GB/layer and kills the process).
- `COVERAGE.md` (auto-generated per-surah counts)
