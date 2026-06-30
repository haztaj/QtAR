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

JUZ_AMMA = set(range(78, 115))
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


def load_juz_amma_qmd() -> pd.DataFrame:
    print("Loading quran-md-ayahs (Juz Amma slices) ...")
    frames: list[pd.DataFrame] = []
    covered: set[int] = set()
    fully_covered = False

    for i in range(QMD_TOTAL - 1, -1, -1):
        path = _parquet_path(i)
        if path.exists():
            print(f"  [{i:02d}] Using cached {path.name}")
        else:
            path = _download_parquet(i)

        df = pd.read_parquet(path, columns=["surah_id", "ayah_id", "reciter_name"])
        juz_slice = df[df["surah_id"].isin(JUZ_AMMA)]
        frames.append(juz_slice)
        covered |= set(juz_slice["surah_id"].unique())

        missing = JUZ_AMMA - covered
        print(f"       {len(juz_slice):>5} Juz Amma rows | "
              f"surahs covered: {len(covered)}/37"
              + (f" | still missing: {sorted(missing)[:8]}" if missing else ""))

        if not missing:
            if not fully_covered:
                # First full-coverage hit: grab one extra parquet in case the
                # boundary surah is split across two shards (e.g. surah 78).
                fully_covered = True
                continue
            print("  Juz Amma fully covered — stopping.")
            break
    else:
        missing = JUZ_AMMA - covered
        if missing:
            print(f"WARNING: exhausted all parquets; missing surahs: {sorted(missing)}", file=sys.stderr)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# RetaSy
# ---------------------------------------------------------------------------

def load_retasy_juz_amma() -> pd.DataFrame:
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

    df = df[df["surah_id"].isin(JUZ_AMMA)]
    print(f"  After Juz Amma filter: {len(df)} rows")
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
    print("COVERAGE REPORT — Juz Amma (surahs 78–114)")
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
        "# Juz Amma — dataset coverage\n\n",
        "Auto-generated by `data/prepare.py`. Re-run to refresh. "
        "Do not edit by hand.\n\n",
        "## quran-md-ayahs — Juz Amma\n\n",
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
    qmd = load_juz_amma_qmd()
    retasy = load_retasy_juz_amma()
    print_report(qmd, retasy)
    write_data_claude_md(qmd, retasy)
