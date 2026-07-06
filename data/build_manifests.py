#!/usr/bin/env python3
"""
Build lhotse RecordingSet + SupervisionSet manifests from manifest.csv.

Steps:
  1. Load manifest.csv (output of extract_audio.py)
  2. Get per-file audio metadata (sampling_rate, duration) via mutagen
  3. Save ayah Arabic text to data/manifests/ayah_text.json (sidecar — avoids
     encoding issues in JSONL and keeps manifests ASCII-safe)
  4. Split by reciter: train (24) / val (3) / test (3)
  5. Write data/manifests/{train,val,test}_{recordings,supervisions}.jsonl.gz

Note: audio is 22050 Hz MP3; icefall resamples to 16 kHz during data loading.

Run after: extract_audio.py
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from mutagen.mp3 import MP3
from lhotse import AudioSource, Recording, RecordingSet, SupervisionSegment, SupervisionSet

DATA_DIR = Path(__file__).parent
QMD_DIR = DATA_DIR / "raw" / "quran-md-ayahs"
AUDIO_DIR = DATA_DIR / "raw" / "audio"
MANIFEST_DIR = DATA_DIR / "manifests"

CORPUS_SURAHS = {1, 2, 3} | set(range(78, 115))   # surahs 1-3 added 2026-07-05
WORKERS = 16

VAL_RECITERS = 3
TEST_RECITERS = 3


# ---------------------------------------------------------------------------

def get_audio_info(path: str) -> tuple[int, int, float]:
    """Return (sampling_rate, num_samples, duration) via MP3 header (no decode)."""
    audio = MP3(path)
    sr = audio.info.sample_rate
    duration = audio.info.length
    num_samples = int(duration * sr)
    return sr, num_samples, duration


def load_ayah_text() -> dict[str, str]:
    """Build 'surah_id:ayah_id' -> ayah_ar from downloaded parquets."""
    print("Loading ayah text from parquets ...")
    ref: dict[str, str] = {}
    for path in sorted(QMD_DIR.glob("train-*.parquet")):
        df = pd.read_parquet(path, columns=["surah_id", "ayah_id", "ayah_ar"])
        sub = df[df["surah_id"].isin(CORPUS_SURAHS)].drop_duplicates(["surah_id", "ayah_id"])
        for row in sub.itertuples(index=False):
            ref[f"{int(row.surah_id)}:{int(row.ayah_id)}"] = row.ayah_ar
    print(f"  {len(ref)} unique (surah, ayah) entries")
    return ref


# ---------------------------------------------------------------------------

def main():
    manifest_csv = AUDIO_DIR / "manifest.csv"
    if not manifest_csv.exists():
        print("manifest.csv not found — run extract_audio.py first", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(manifest_csv)
    print(f"Loaded {len(df)} recordings from manifest")

    # --- Ayah text sidecar ---
    ayah_text = load_ayah_text()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    ayah_text_path = MANIFEST_DIR / "ayah_text.json"
    ayah_text_path.write_text(
        json.dumps(ayah_text, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Wrote {ayah_text_path.name}")

    # --- Audio metadata via mutagen (parallel) ---
    print(f"Reading MP3 headers for {len(df)} files ({WORKERS} workers) ...")
    results: dict[int, tuple] = {}
    errors = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(get_audio_info, row["path"]): i
                   for i, row in df.iterrows()}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(df)} ...", flush=True)
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = (22050, 0, 0.0)
                errors += 1
                if errors <= 5:
                    print(f"  WARN [{idx}] {e}", file=sys.stderr)

    sr_col, ns_col, dur_col = [], [], []
    for i in range(len(df)):
        sr, ns, dur = results[i]
        sr_col.append(sr); ns_col.append(ns); dur_col.append(dur)

    df["sampling_rate"] = sr_col
    df["num_samples"] = ns_col
    df["duration"] = dur_col

    if errors:
        print(f"  {errors} files had metadata errors", file=sys.stderr)

    print(f"  Sample rates: {sorted(df['sampling_rate'].unique())}")
    print(f"  Total duration: {df['duration'].sum() / 3600:.1f} h")

    # --- Reciter split ---
    reciters = sorted(df["reciter_id"].unique())
    test_set  = set(reciters[-TEST_RECITERS:])
    val_set   = set(reciters[-(TEST_RECITERS + VAL_RECITERS):-TEST_RECITERS])
    train_set = set(reciters[:-(TEST_RECITERS + VAL_RECITERS)])

    print(f"\nSplit — {len(train_set)} train / {len(val_set)} val / {len(test_set)} test reciters")
    print(f"  val : {sorted(val_set)}")
    print(f"  test: {sorted(test_set)}")

    df["split"] = df["reciter_id"].map(
        lambda r: "test" if r in test_set else ("val" if r in val_set else "train")
    )

    # --- Build and write lhotse manifests ---
    for split in ("train", "val", "test"):
        split_df = df[df["split"] == split]
        print(f"\n{split}: {len(split_df)} recordings | "
              f"{split_df['duration'].sum() / 3600:.2f} h | "
              f"{split_df['reciter_id'].nunique()} reciters")

        recordings, supervisions = [], []
        for row in split_df.itertuples(index=False):
            r = row._asdict()
            sid, aid = int(r["surah_id"]), int(r["ayah_id"])

            rec = Recording(
                id=r["recording_id"],
                sources=[AudioSource(type="file", channels=[0], source=r["path"])],
                sampling_rate=int(r["sampling_rate"]),
                num_samples=int(r["num_samples"]),
                duration=float(r["duration"]),
            )
            sup = SupervisionSegment(
                id=r["recording_id"],
                recording_id=r["recording_id"],
                start=0.0,
                duration=float(r["duration"]),
                channel=0,
                custom={
                    "surah_id": sid,
                    "ayah_id": aid,
                    "reciter_id": r["reciter_id"],
                    "reciter_name": r["reciter_name"],
                    # ayah_text omitted here; load from ayah_text.json at training time
                    "ayah_text_key": f"{sid}:{aid}",
                },
            )
            recordings.append(rec)
            supervisions.append(sup)

        rec_set = RecordingSet.from_recordings(recordings)
        sup_set = SupervisionSet.from_segments(supervisions)

        rec_path = MANIFEST_DIR / f"{split}_recordings.jsonl.gz"
        sup_path = MANIFEST_DIR / f"{split}_supervisions.jsonl.gz"
        rec_set.to_file(str(rec_path))
        sup_set.to_file(str(sup_path))
        print(f"  -> {rec_path.name}")
        print(f"  -> {sup_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
