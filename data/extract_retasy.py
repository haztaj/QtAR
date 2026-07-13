#!/usr/bin/env python3
"""
Extract RetaSy learner audio to disk + manifest, with derived numeric ayah IDs.

RetaSy's HF Audio column decodes via torchcodec (broken here), so we read raw
bytes with Audio(decode=False) and write them out unchanged; soundfile decodes
them later. Ayah IDs are derived exactly as in derive_aya_ids.py (name->id map +
normalized-text join against quran-md-ayahs).

Output:
  data/raw/retasy_audio/<reciter_id>/<surah>_<ayah>_<row>.<ext>
  data/raw/retasy_audio/manifest.csv  (same schema as the main manifest + source col)
"""

import sys
from pathlib import Path

import pandas as pd
from datasets import Audio, load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from derive_aya_ids import (CORPUS_SURAHS, SURAH_NAME_TO_ID, build_reference, normalize)

DATA_DIR = Path(__file__).parent
OUT_DIR = DATA_DIR / "raw" / "retasy_audio"


def _ext(path: str | None) -> str:
    if path and "." in path:
        return "." + path.rsplit(".", 1)[-1].lower()
    return ".wav"


def main():
    ref = build_reference()  # (surah_id, normalized_text) -> ayah_id
    print("Loading RetaSy with raw audio bytes ...")
    ds = load_dataset("RetaSy/quranic_audio_dataset", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    matched = unmatched = non_quran = 0

    for ex in ds:
        sid = SURAH_NAME_TO_ID.get(ex["Surah"])
        if sid is None or sid not in CORPUS_SURAHS:
            non_quran += 1
            continue
        aid = ref.get((sid, normalize(str(ex["Aya"]))))
        if aid is None:
            unmatched += 1
            continue

        audio = ex["audio"]
        data = audio.get("bytes")
        if not data:
            unmatched += 1
            continue
        ext = _ext(audio.get("path"))
        rec = ex.get("reciter_id") or "unknown"
        rec_dir = OUT_DIR / str(rec)
        rec_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{sid:03d}_{aid:03d}_{matched:05d}{ext}"
        (rec_dir / fname).write_bytes(data)

        rows.append({
            "recording_id": f"retasy_{rec}_{sid:03d}_{aid:03d}_{matched:05d}",
            "reciter_id": str(rec),
            "reciter_name": str(rec),
            "surah_id": sid,
            "ayah_id": aid,
            "path": str(rec_dir / fname),
            "duration": (ex.get("duration_ms") or 0) / 1000.0,
            "source": "retasy",
            "final_label": ex.get("final_label"),
            "reciter_country": ex.get("reciter_country"),
        })
        matched += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "manifest.csv", index=False)
    print(f"\nmatched={matched}  unmatched={unmatched}  non_quran(dropped)={non_quran}")
    print(f"Wrote {len(df)} clips -> {OUT_DIR / 'manifest.csv'}")
    print(f"reciters={df['reciter_id'].nunique()}  surahs={sorted(df['surah_id'].unique())}")
    print("\nPer-surah:")
    print(df.groupby("surah_id").size().to_string())


if __name__ == "__main__":
    main()
