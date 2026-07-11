"""Download per-surah CONTINUOUS recitations for phase-3 concatenation training.

Source specs live in data/raw/continuous/sources/<reciter>.json — the quranicaudio-style
map {"N": {"surah_number", "audio_url", "duration"}, ...} the user provides per reciter.
Only MVP-scope surahs are fetched (1-3 + Juz Amma 78-114). Downloads are resumable
(skip-if-complete by size), verified (MP3/ID3 sniff + mutagen duration vs the spec's),
and land in data/raw/continuous/<reciter>/sNNN.mp3 (never committed — see .gitignore).

  python data/download_continuous.py                 # all specs in sources/
  python data/download_continuous.py --reciter huthayfi
  python data/download_continuous.py --report        # coverage/duration table only
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "data/raw/continuous"
SOURCES = ROOT / "sources"
SCOPE = {1, 2, 3} | set(range(78, 115))   # MVP corpus (locked scope)


def spec_files(only: str | None):
    for p in sorted(SOURCES.glob("*.json")):
        if only and p.stem != only:
            continue
        yield p.stem, json.loads(p.read_text(encoding="utf-8"))


def mp3_dur(path: Path) -> float | None:
    try:
        from mutagen.mp3 import MP3
        return float(MP3(str(path)).info.length)
    except Exception:
        return None


def fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "QtAR-phase3/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reciter", default=None, help="only this source spec (stem of the json)")
    ap.add_argument("--report", action="store_true", help="coverage report only, no downloads")
    args = ap.parse_args()

    grand_ok = grand_sec = 0
    for reciter, spec in spec_files(args.reciter):
        out = ROOT / reciter
        out.mkdir(parents=True, exist_ok=True)
        meta = spec.get("_meta", {})   # optional: canonical_reciter / eval_only quarantine
        rows = [v for k, v in spec.items()
                if not k.startswith("_") and int(v["surah_number"]) in SCOPE]
        rows.sort(key=lambda v: int(v["surah_number"]))
        n_ok, sec = 0, 0.0
        for v in rows:
            s = int(v["surah_number"])
            dest = out / f"s{s:03d}.mp3"
            want = float(v.get("duration") or 0)
            if not dest.exists() and not args.report:
                try:
                    fetch(v["audio_url"], dest)
                except Exception as e:
                    print(f"  {reciter} s{s:03d}: DOWNLOAD FAILED {e}", flush=True)
                    continue
            if not dest.exists():
                continue
            if dest.stat().st_size < 1024 or dest.read_bytes()[:3] not in (b"ID3", b"\xff\xfb", b"\xff\xf3"):
                print(f"  {reciter} s{s:03d}: NOT AN MP3 — deleting", flush=True)
                dest.unlink()
                continue
            got = mp3_dur(dest)
            if got is not None and want and abs(got - want) > max(10.0, 0.05 * want):
                print(f"  {reciter} s{s:03d}: duration {got:.0f}s != spec {want:.0f}s (check)", flush=True)
            n_ok += 1
            sec += got if got is not None else want
        tag = " [EVAL-ONLY — held-out reciter]" if meta.get("eval_only") else ""
        print(f"{reciter}: {n_ok}/{len(rows)} in-scope surahs, {sec/3600:.2f} h{tag}", flush=True)
        grand_ok += n_ok
        grand_sec += sec
    print(f"TOTAL: {grand_ok} files, {grand_sec/3600:.2f} h continuous audio")


if __name__ == "__main__":
    main()
