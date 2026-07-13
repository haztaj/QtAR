#!/usr/bin/env python3
"""
Derive numeric ayah IDs for RetaSy rows by matching their Arabic text (Aya field)
against the reference text from already-downloaded quran-md-ayahs parquets.

Lookup key: (surah_id, normalize(ayah_ar)) to avoid cross-surah collisions.
Normalization: strip diacritics, tatweel, normalize alef variants — minimum needed
for reliable matching without losing phonemically meaningful differences.

Writes: data/raw/retasy_juz_amma_with_ids.parquet
        (unmatched rows are logged and excluded)
"""

import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATA_DIR = Path(__file__).parent
QMD_DIR = DATA_DIR / "raw" / "quran-md-ayahs"

# FULL QURAN corpus (expanded 2026-07-13; was Juz Amma range(78,115)). Kept as the
# RetaSy corpus filter — RetaSy only contains short surahs + Al-Fatiha, and widening to
# the full corpus rescues surah-1 (Al-Fatiha) learner clips that the Juz-Amma filter dropped.
CORPUS_SURAHS = set(range(1, 115))

SURAH_NAME_TO_ID = {
    "Al-Faatihah": 1,
    "Al-NABAA": 78,
    "Al-Qadr": 97,
    "Al-Asr": 103,
    "Al-Humazah": 104,
    "Al-Fil": 105,
    "Quraish": 106,
    "Al-Maaoon": 107,
    "Al-Kauthar": 108,
    "Al-Kafiroon": 109,
    "An-Nasr": 110,
    "Al-Masad": 111,
    "Al-Ikhlas": 112,
    "Al-Falaq": 113,
    "An-Nas": 114,
}

# Diacritics regex: tashkeel (U+064B-U+065F), Arabic signs (U+0610-U+061A),
# Quranic annotation signs (U+06D6-U+06ED), Superscript Alef U+0670 (Uthmani
# orthography marker), Arabic Extended-A U+08A0-U+08FF (RetaSy tajweed marks).
_DIACRITICS = re.compile(
    "[ً-ٟؐ-ؚۖ-ۜ۟-۪ۤۧۨ-ٰۭࢠ-ࣿ]"
)
_TATWEEL = re.compile("ـ")  # tatweel / kashida
# U+0649 alef maksura -> U+0627 alef: Uthmani u0649+u0670 = simplified alef.
_ALEF = str.maketrans("أإآٱى", "ااااا")
# Persian/Urdu codepoints used by non-Arabic keyboard layouts in RetaSy
_PERSIAN_ARABIC = str.maketrans(
    "یک",  # Farsi Yeh (ی), Keheh (ک)
    "يك",  # Arabic Yeh (ي), Kaf  (ك)
)


def normalize(text: str) -> str:
    # Uthmani superscript alef (U+0670) represents a full alef letter, not a diacritic.
    # Must handle BEFORE the diacritics strip so the alef is preserved in the output.
    # Rule 1: alef-maksura + superscript-alef (U+0649 U+0670) -> plain alef (single char).
    text = re.sub("ىٰ", "ا", text)
    # Rule 2: superscript-alef on any other letter -> insert alef after that letter.
    text = re.sub("ٰ", "ا", text)
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = text.translate(_ALEF)
    text = text.translate(_PERSIAN_ARABIC)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Build reference: (surah_id, normalized_text) -> ayah_id
# ---------------------------------------------------------------------------

def build_reference() -> dict[tuple[int, str], int]:
    ref: dict[tuple[int, str], int] = {}
    parquets = sorted(QMD_DIR.glob("train-*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquets found in {QMD_DIR}")

    print(f"Building reference from {len(parquets)} parquet(s) ...")
    for path in parquets:
        df = pd.read_parquet(path, columns=["surah_id", "ayah_id", "ayah_ar"])
        juz = df[df["surah_id"].isin(CORPUS_SURAHS)].drop_duplicates(["surah_id", "ayah_id"])
        for row in juz.itertuples(index=False):
            key = (int(row.surah_id), normalize(row.ayah_ar))
            ref[key] = int(row.ayah_id)

    print(f"  Reference size: {len(ref)} unique (surah, ayah) entries")
    return ref


# ---------------------------------------------------------------------------
# Load and filter RetaSy
# ---------------------------------------------------------------------------

def load_retasy() -> pd.DataFrame:
    print("Loading RetaSy ...")
    ds = load_dataset("RetaSy/quranic_audio_dataset", split="train")
    if "audio" in ds.column_names:
        ds = ds.remove_columns(["audio"])
    df = ds.to_pandas()

    df["surah_id"] = df["Surah"].map(SURAH_NAME_TO_ID)
    df = df.dropna(subset=["surah_id"])
    df["surah_id"] = df["surah_id"].astype(int)
    df = df[df["surah_id"].isin(CORPUS_SURAHS)].copy()
    print(f"  RetaSy corpus rows: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

def derive_ids(retasy: pd.DataFrame, ref: dict[tuple[int, str], int]) -> pd.DataFrame:
    retasy = retasy.copy()
    retasy["norm_aya"] = retasy["Aya"].astype(str).apply(normalize)
    retasy["ayah_id"] = retasy.apply(
        lambda r: ref.get((r["surah_id"], r["norm_aya"])), axis=1
    )

    matched = retasy["ayah_id"].notna()
    print(f"\nMatch results: {matched.sum()} / {len(retasy)} rows matched "
          f"({matched.mean():.1%})")

    unmatched = retasy[~matched][["Surah", "surah_id", "norm_aya"]].copy()
    if len(unmatched):
        print(f"\nUnmatched samples (first 10):")
        for _, row in unmatched.head(10).iterrows():
            char_len = len(row["norm_aya"])
            # Print length and first few Unicode codepoints (safe for any terminal)
            codepoints = " ".join(f"U+{ord(c):04X}" for c in row["norm_aya"][:6])
            print(f"  surah {row['surah_id']:>3} ({row['Surah']}): "
                  f"{char_len} chars, starts: {codepoints}")

        unmatched_out = DATA_DIR / "raw" / "retasy_unmatched.txt"
        unmatched["norm_aya"].to_csv(unmatched_out, index=False, header=False)
        print(f"\n  Full unmatched list -> {unmatched_out}")

    retasy = retasy[matched].copy()
    retasy["ayah_id"] = retasy["ayah_id"].astype(int)
    return retasy.drop(columns=["norm_aya"])


# ---------------------------------------------------------------------------

def main():
    ref = build_reference()
    retasy = load_retasy()
    retasy = derive_ids(retasy, ref)

    out = DATA_DIR / "raw" / "retasy_juz_amma_with_ids.parquet"
    retasy.to_parquet(out, index=False)
    print(f"\nWrote {len(retasy)} rows -> {out}")

    print("\nPer-surah breakdown (matched):")
    print(f"{'Surah':>6}  {'Ayat':>6}  {'Clips':>7}")
    for sid, g in retasy.groupby("surah_id"):
        print(f"{sid:>6}  {g['ayah_id'].nunique():>6}  {len(g):>7}")


if __name__ == "__main__":
    main()
