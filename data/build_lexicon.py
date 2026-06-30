#!/usr/bin/env python3
"""
Run the Quranic G2P over every Juz Amma ayah and emit the CTC training assets:

  data/lang/ayah_phonemes.json  "surah:ayah" -> "ph ph ph ..."
  data/lang/tokens.txt          icefall/sherpa-onnx token table (<blk>=0, phonemes 1..N)

Also validates that every emitted phoneme is in the declared inventory and
prints corpus statistics.

Run after: build_manifests.py (needs manifests/ayah_text.json)
"""

import json
import sys
from collections import Counter
from pathlib import Path

from quran_g2p import g2p_ayah, PHONEME_SET

DATA_DIR = Path(__file__).parent
AYAH_TEXT = DATA_DIR / "manifests" / "ayah_text.json"
LANG_DIR = DATA_DIR / "lang"


def main():
    if not AYAH_TEXT.exists():
        print("manifests/ayah_text.json not found — run build_manifests.py first",
              file=sys.stderr)
        sys.exit(1)

    text = json.loads(AYAH_TEXT.read_text(encoding="utf-8"))
    print(f"Loaded {len(text)} ayat")

    ayah_phonemes: dict[str, str] = {}
    phoneme_counts: Counter = Counter()
    lengths: list[int] = []
    oov: set[str] = set()
    valid = set(PHONEME_SET)

    for key, t in text.items():
        ph = g2p_ayah(t)
        ayah_phonemes[key] = " ".join(ph)
        phoneme_counts.update(ph)
        lengths.append(len(ph))
        oov |= {p for p in ph if p not in valid}

    if oov:
        print(f"ERROR: out-of-inventory phonemes emitted: {sorted(oov)}", file=sys.stderr)
        sys.exit(2)

    # --- Write assets ---
    LANG_DIR.mkdir(parents=True, exist_ok=True)
    (LANG_DIR / "ayah_phonemes.json").write_text(
        json.dumps(ayah_phonemes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # tokens.txt: <blk> 0, then phonemes in canonical PHONEME_SET order
    used = [p for p in PHONEME_SET if p in phoneme_counts]
    unused = [p for p in PHONEME_SET if p not in phoneme_counts]
    lines = ["<blk> 0"]
    for i, p in enumerate(PHONEME_SET, start=1):
        lines.append(f"{p} {i}")
    (LANG_DIR / "tokens.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- Stats ---
    print(f"\nWrote {LANG_DIR / 'ayah_phonemes.json'} ({len(ayah_phonemes)} ayat)")
    print(f"Wrote {LANG_DIR / 'tokens.txt'} ({len(PHONEME_SET)} phonemes + <blk>)")
    print(f"\nPhoneme-sequence length: min={min(lengths)} "
          f"max={max(lengths)} mean={sum(lengths)/len(lengths):.1f}")
    print(f"Inventory used: {len(used)}/{len(PHONEME_SET)} phonemes")
    if unused:
        print(f"  Never emitted: {unused}")
    print("\nPhoneme frequency (top 15):")
    for p, c in phoneme_counts.most_common(15):
        print(f"  {p:>3}  {c}")


if __name__ == "__main__":
    main()
