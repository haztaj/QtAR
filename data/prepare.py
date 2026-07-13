#!/usr/bin/env python3
"""
Pull Buraaq/quran-md-ayahs parquets (backward from the end until Juz Amma is
fully covered) and RetaSy/quranic_audio_dataset, then report per-surah clip
counts and durations.  Writes a summary table to data/CLAUDE.md.

Pre-downloaded parquets: place them in  data/raw/quran-md-ayahs/  (flat).
The script also accepts the nested  data/raw/quran-md-ayahs/data/  layout
produced by hf_hub_download, so either location works.
"""

import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
QMD_DIR = RAW_DIR / "quran-md-ayahs"

CORPUS_SURAHS = set(range(1, 115))       # FULL QURAN (all 114 surahs), expanded 2026-07-13
# Per-surah total ayah counts (Hafs) — completeness reference for the coverage check only.
# NOT a label source: training ayah IDs still come from the parquet surah_id/ayah_id fields
# (standing rule). This table just flags surahs whose clips don't reach the known ayah total.
SURAH_AYAH_COUNTS = {
    1: 7, 2: 286, 3: 200, 4: 176, 5: 120, 6: 165, 7: 206, 8: 75, 9: 129, 10: 109,
    11: 123, 12: 111, 13: 43, 14: 52, 15: 99, 16: 128, 17: 111, 18: 110, 19: 98, 20: 135,
    21: 112, 22: 78, 23: 118, 24: 64, 25: 77, 26: 227, 27: 93, 28: 88, 29: 69, 30: 60,
    31: 34, 32: 30, 33: 73, 34: 54, 35: 45, 36: 83, 37: 182, 38: 88, 39: 75, 40: 85,
    41: 54, 42: 53, 43: 89, 44: 59, 45: 37, 46: 35, 47: 38, 48: 29, 49: 18, 50: 45,
    51: 60, 52: 49, 53: 62, 54: 55, 55: 78, 56: 96, 57: 29, 58: 22, 59: 24, 60: 13,
    61: 14, 62: 11, 63: 11, 64: 18, 65: 12, 66: 12, 67: 30, 68: 52, 69: 52, 70: 44,
    71: 28, 72: 28, 73: 20, 74: 56, 75: 40, 76: 31, 77: 50, 78: 40, 79: 46, 80: 42,
    81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17, 87: 19, 88: 26, 89: 30, 90: 20,
    91: 15, 92: 21, 93: 11, 94: 8, 95: 8, 96: 19, 97: 5, 98: 8, 99: 8, 100: 11,
    101: 11, 102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3, 109: 6, 110: 3,
    111: 5, 112: 4, 113: 5, 114: 6,
}
QMD_REPO = "Buraaq/quran-md-ayahs"
QMD_TOTAL = 71


# ---------------------------------------------------------------------------
# quran-md-ayahs
# ---------------------------------------------------------------------------

def _parquet_path(index: int) -> Path:
    """Return local path for a parquet shard, checking both flat and nested."""
    fname = f"train-{index:05d}-of-{QMD_TOTAL:05d}.parquet"
    flat = QMD_DIR / fname
    nested = QMD_DIR / "data" / fname
    if flat.exists():
        return flat
    if nested.exists():
        return nested
    return flat  # target path for a new download


def _download_parquet(index: int) -> Path:
    fname = f"train-{index:05d}-of-{QMD_TOTAL:05d}.parquet"
    print(f"  Downloading {fname} ...")
    QMD_DIR.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=QMD_REPO,
        filename=f"data/{fname}",
        repo_type="dataset",
        local_dir=str(QMD_DIR),
    )
    # hf_hub_download places the file under QMD_DIR/data/fname
    nested = QMD_DIR / "data" / fname
    flat = QMD_DIR / fname
    if nested.exists() and not flat.exists():
        nested.rename(flat)
    return flat


def load_corpus_qmd() -> pd.DataFrame:
    """Slice the corpus surahs (1-3 + Juz Amma) from the **locally-present** parquet shards.

    Juz Amma lives in the final shards (64-70) and surahs 1-3 in the first (0-5); both ranges
    are pre-downloaded. We scan only shards that exist on disk and never download the ~34 GB of
    middle shards — corpus expansion added surahs 1-3 from already-downloaded shards, not a full
    dataset pull (see CLAUDE.md, 2026-07-05).
    """
    print("Loading quran-md-ayahs (corpus slices: surahs 1-3 + Juz Amma) ...")
    frames: list[pd.DataFrame] = []
    covered: set[int] = set()

    for i in range(QMD_TOTAL):
        path = _parquet_path(i)
        if not path.exists():
            continue  # skip non-downloaded shards (no download)
        df = pd.read_parquet(path, columns=["surah_id", "ayah_id", "reciter_name"])
        sl = df[df["surah_id"].isin(CORPUS_SURAHS)]
        if sl.empty:
            continue
        frames.append(sl)
        covered |= set(sl["surah_id"].unique())
        print(f"  [{i:02d}] {path.name}: {len(sl):>5} corpus rows "
              f"(surahs {sorted(sl['surah_id'].unique())[:6]})")

    missing = CORPUS_SURAHS - covered
    if missing:
        print(f"WARNING: corpus surahs with no rows in local shards: {sorted(missing)}",
              file=sys.stderr)
    qmd = pd.concat(frames, ignore_index=True)
    # Completeness check on the added surahs (Juz-Amma coverage was verified previously).
    for sid, total in SURAH_AYAH_COUNTS.items():
        got = qmd[qmd["surah_id"] == sid]["ayah_id"].nunique()
        flag = "OK" if got == total else f"INCOMPLETE ({got}/{total})"
        print(f"  surah {sid}: {got}/{total} ayat -> {flag}")
    return qmd


# ---------------------------------------------------------------------------
# RetaSy
# ---------------------------------------------------------------------------

def load_retasy_corpus() -> pd.DataFrame:
    print("\nLoading RetaSy/quranic_audio_dataset ...")
    ds = load_dataset("RetaSy/quranic_audio_dataset", split="train")

    # Drop the audio column — we only need metadata here.
    if "audio" in ds.column_names:
        ds = ds.remove_columns(["audio"])

    df = ds.to_pandas()
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Surah is a transliterated name, not a number.
    # Map known Quran surahs; any unmapped value (Adhan, adhkar, duas, etc.) is dropped.
    # Aya contains Arabic text — numeric aya IDs must be derived via Tanzil (separate step).
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
    before = len(df)
    df["surah_id"] = df["Surah"].map(SURAH_NAME_TO_ID)
    df = df.dropna(subset=["surah_id"])
    df["surah_id"] = df["surah_id"].astype(int)
    print(f"  Dropped {before - len(df)} non-Quran rows -> {len(df)} remaining")

    df = df[df["surah_id"].isin(CORPUS_SURAHS)]
    print(f"  After corpus filter (1-3 + Juz Amma): {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _duration_col(df: pd.DataFrame) -> str | None:
    # Prefer duration_ms; fall back to any column containing "duration".
    if "duration_ms" in df.columns:
        return "duration_ms"
    return next((c for c in df.columns if "duration" in c.lower()), None)


def _duration_hours(df: pd.DataFrame, col: str) -> float:
    # duration_ms -> hours; plain seconds column -> hours.
    divisor = 3_600_000 if col == "duration_ms" else 3600
    return df[col].sum() / divisor


def print_report(qmd: pd.DataFrame, retasy: pd.DataFrame) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("COVERAGE REPORT — corpus (surahs 1-3 + Juz Amma 78–114)")
    print(sep)

    print(f"\n--- quran-md-ayahs ({len(qmd)} clips, {qmd['reciter_name'].nunique()} reciters) ---")
    print(f"{'Surah':>6}  {'Clips':>7}")
    for sid, g in qmd.groupby("surah_id"):
        print(f"{sid:>6}  {len(g):>7}")

    dur = _duration_col(retasy)
    print(f"\n--- RetaSy Juz Amma slice ({len(retasy)} clips) ---")
    header = f"{'Surah':>6}  {'Clips':>7}"
    if dur:
        header += f"  {'Dur (h)':>8}"
    print(header)
    for sid, g in retasy.groupby("surah_id"):
        row = f"{sid:>6}  {len(g):>7}"
        if dur:
            row += f"  {_duration_hours(g, dur):>8.2f}"
        print(row)


def write_data_claude_md(qmd: pd.DataFrame, retasy: pd.DataFrame) -> None:
    dur = _duration_col(retasy)

    lines = [
        "# Corpus — dataset coverage (surahs 1-3 + Juz Amma 78-114)\n\n",
        "Auto-generated by `data/prepare.py`. Re-run to refresh. "
        "Do not edit by hand.\n\n",
        "## quran-md-ayahs — corpus\n\n",
        f"- Total clips: {len(qmd)}\n",
        f"- Reciters: {qmd['reciter_name'].nunique()}\n\n",
        "| Surah | Clips |\n",
        "|------:|------:|\n",
    ]
    for sid, g in qmd.groupby("surah_id"):
        lines.append(f"| {sid} | {len(g)} |\n")

    lines += [
        "\n## RetaSy — Juz Amma slice\n\n",
        f"- Total clips: {len(retasy)}\n",
        f"- Note: aya-number derivation via Tanzil is a separate step.\n\n",
        "| Surah | Clips |" + (" Duration (h) |\n" if dur else "\n"),
        "|------:|------:|" + ("-------------:|\n" if dur else "\n"),
    ]
    for sid, g in retasy.groupby("surah_id"):
        row = f"| {sid} | {len(g)} |"
        if dur:
            row += f" {_duration_hours(g, dur):.2f} |"
        lines.append(row + "\n")

    out = DATA_DIR / "COVERAGE.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    qmd = load_corpus_qmd()
    retasy = load_retasy_corpus()
    print_report(qmd, retasy)
    write_data_claude_md(qmd, retasy)
