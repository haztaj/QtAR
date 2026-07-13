#!/usr/bin/env python3
"""
Extract MP3 audio bytes from quran-md-ayahs parquets to files on disk.
Processes the detection corpus (FULL QURAN, surahs 1-114 as of 2026-07-13).  Safe to
re-run: skips files that already exist (so re-running after the full-Quran expansion only
writes the newly-covered surahs; the existing 1-3 + Juz Amma files are untouched).

Output layout:
    data/raw/audio/<reciter_id>/s<surah>_a<ayah>.mp3

Also writes data/raw/audio/manifest.csv with one row per file:
    recording_id, reciter_id, reciter_name, surah_id, ayah_id, path
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
QMD_DIR = DATA_DIR / "raw" / "quran-md-ayahs"
AUDIO_DIR = DATA_DIR / "raw" / "audio"

CORPUS_SURAHS = set(range(1, 115))   # FULL QURAN, expanded 2026-07-13 (was 1-3 + Juz Amma)
WORKERS = 8  # parallel writers


# ---------------------------------------------------------------------------

def recording_id(reciter_id: str, surah_id: int, ayah_id: int) -> str:
    return f"{reciter_id}_s{surah_id:03d}_a{ayah_id:03d}"


def extract_row(row) -> dict | None:
    """Write audio bytes to disk.  Returns manifest dict or None if skipped."""
    rec_id = str(row["reciter_id"])
    sid = int(row["surah_id"])
    aid = int(row["ayah_id"])
    rid = recording_id(rec_id, sid, aid)

    out_dir = AUDIO_DIR / rec_id
    out_path = out_dir / f"s{sid:03d}_a{aid:03d}.mp3"

    if not out_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_bytes = row["audio"]["bytes"]
        out_path.write_bytes(audio_bytes)

    return {
        "recording_id": rid,
        "reciter_id": rec_id,
        "reciter_name": row["reciter_name"],
        "surah_id": sid,
        "ayah_id": aid,
        "path": str(out_path),
    }


def process_parquet(path: Path) -> list[dict]:
    df = pd.read_parquet(
        path,
        columns=["surah_id", "ayah_id", "reciter_id", "reciter_name", "audio"],
    )
    rows = df[df["surah_id"].isin(CORPUS_SURAHS)].to_dict("records")

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(extract_row, r): r for r in rows}
        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result:
                    results.append(result)
            except Exception as e:
                r = futures[fut]
                print(
                    f"  ERROR s{r['surah_id']}:a{r['ayah_id']} "
                    f"reciter={r['reciter_id']}: {e}",
                    file=sys.stderr,
                )
    return results


# ---------------------------------------------------------------------------

def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    parquets = sorted(QMD_DIR.glob("train-*.parquet"))
    if not parquets:
        print(f"No parquets found in {QMD_DIR}", file=sys.stderr)
        sys.exit(1)

    manifest_path = AUDIO_DIR / "manifest.csv"
    existing: set[str] = set()
    if manifest_path.exists():
        existing = set(pd.read_csv(manifest_path)["recording_id"])
        print(f"Resuming — {len(existing)} recordings already in manifest.")

    all_records: list[dict] = []
    total_written = 0

    for i, path in enumerate(parquets, 1):
        # Quick check: does this parquet have any corpus rows?
        sids = pd.read_parquet(path, columns=["surah_id"])["surah_id"]
        if not sids.isin(CORPUS_SURAHS).any():
            continue

        print(f"[{i}/{len(parquets)}] {path.name} ...", end=" ", flush=True)
        records = process_parquet(path)

        new = [r for r in records if r["recording_id"] not in existing]
        all_records.extend(new)
        existing.update(r["recording_id"] for r in new)
        total_written += len(new)
        print(f"{len(records)} corpus rows | {len(new)} new files written")

    if all_records:
        new_df = pd.DataFrame(all_records)
        if manifest_path.exists():
            existing_df = pd.read_csv(manifest_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(manifest_path, index=False)

    total_in_manifest = len(pd.read_csv(manifest_path)) if manifest_path.exists() else 0
    print(f"\nDone. {total_written} new files written. "
          f"Manifest total: {total_in_manifest} recordings -> {manifest_path}")


if __name__ == "__main__":
    main()
