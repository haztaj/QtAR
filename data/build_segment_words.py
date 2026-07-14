"""Segment -> word map for WORD-LEVEL highlighting (data/lang/segment_words.json).

For every detector unit ("s:a#NN" waqf segment or unsegmented "s:a" ayah) emit the range of
CONTENT-WORD ordinals [w0, w1) it covers within its ayah. Word ordinals index the ayah's
content words in mushaf order — the demo's words.db rows for that ayah minus the trailing
ayah-number medallion row, so the app resolves ordinal -> word glyph directly.

Exact by construction: segment_phonemes.json carries each segment's TEXT (the very words the
segment was built from), so ranges come from consuming those words in order — no inference.
Validation, per ayah:
  1. the segments' words concatenate to exactly the ayah's content words;
  2. content-word count == words.db per-ayah rows - 1 (the medallion).
Ayat failing either check are listed in "_unmapped" — the UI falls back to whole-ayah
highlight there (expected rare; 2:282-style orthographic tokenization differences).

  python data/build_segment_words.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

DATA = Path(__file__).parent
REPO = DATA.parent
sys.path.insert(0, str(DATA))
from segment_waqf import waqf_segments   # noqa: E402  (content-word tokenizer, waqf-aware)

WORDS_DB = REPO / "sdk/android/demo/src/main/assets/mushaf/words.db"
OUT = REPO / "data/lang/segment_words.json"


def display_words(tokens: list[str]) -> list[str]:
    """Normalize content tokens to the mushaf's WORD segmentation (words.db):
    - drop the rub-el-hizb «۞» and sajda «۩» marks (tokens in our text source, not words);
    - join the standalone vocatives «يا» / «ها» to the following word (QPC Uthmani writes
      them as ONE word: «يَٰٓأَيُّهَا», «هَٰٓأَنتُمْ», ...);
    - split «بَعْدَمَا» (one token in our source, TWO QPC words «بَعْدَ مَا», 2:181).
    Phoneme refs are untouched — this only affects word COUNTING for the map."""
    out: list[str] = []
    join_next = False
    for t in tokens:
        if t in ("۞", "۩"):
            continue
        if join_next:
            out[-1] = out[-1] + " " + t   # joined for counting; text content preserved
            join_next = False
            continue
        if t == "بَعْدَمَا":
            out += ["بَعْدَ", "مَا"]
            continue
        out.append(t)
        join_next = t in ("يَا", "هَا")   # standalone vocative particles
    return out


def main():
    ayah_text = json.load(open(REPO / "data/manifests/ayah_text.json", encoding="utf-8"))
    seg = json.load(open(REPO / "data/lang/segment_phonemes.json", encoding="utf-8"))
    db = sqlite3.connect(str(WORDS_DB))

    # group segment keys per ayah, ordered by segment index
    by_ayah: dict[str, list[str]] = {}
    for k in seg:
        by_ayah.setdefault(k.split("#")[0], []).append(k)
    for v in by_ayah.values():
        v.sort(key=lambda k: int(k.split("#")[1]))

    out: dict[str, list[int]] = {}
    unmapped: list[str] = []
    n_db_mismatch = 0
    for key, text in ayah_text.items():
        s, a = map(int, key.split(":"))
        raw = [w for sg in waqf_segments(text) for w in sg]     # content tokens, mushaf order
        words = display_words(raw)                              # mushaf WORD segmentation
        n_db = db.execute("select count(*) from words where surah=? and ayah=?",
                          (s, a)).fetchone()[0]
        db_ok = n_db - 1 == len(words)     # trailing medallion row
        if not db_ok:
            n_db_mismatch += 1
        if key in by_ayah:                 # segmented ayah: consume each segment's own text
            w0, ranges, consistent = 0, [], True
            for uk in by_ayah[key]:
                seg_disp = display_words(seg[uk]["text"].split())
                w1 = w0 + len(seg_disp)
                if words[w0:w1] != seg_disp:
                    consistent = False
                    break
                ranges.append((uk, [w0, w1]))
                w0 = w1
            consistent = consistent and w0 == len(words)
            if consistent and db_ok:
                out.update(dict(ranges))
            else:
                unmapped.append(key)
        else:                              # unsegmented: the whole ayah
            if db_ok:
                out[key] = [0, len(words)]
            else:
                unmapped.append(key)

    payload = {"_meta": {"words": "content-word ordinals within the ayah (mushaf order; "
                                  "words.db rows minus the trailing ayah-number row)",
                         "unmapped": sorted(unmapped)}}
    payload.update(dict(sorted(out.items())))
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8")
    n_units = len(out)
    print(f"{OUT.name}: {n_units} units mapped "
          f"({len([k for k in out if '#' in k])} segments + "
          f"{len([k for k in out if '#' not in k])} unsegmented ayat); "
          f"unmapped ayat: {len(unmapped)} {sorted(unmapped)[:8]}")
    print(f"words.db count mismatches: {n_db_mismatch}")


if __name__ == "__main__":
    main()
