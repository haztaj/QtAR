#!/usr/bin/env python3
"""
Deterministic Quranic grapheme-to-phoneme (G2P) for Hafs ʿan ʿAsim.

Input : fully-diacritized Quranic text (one ayah).
Output: list of phoneme tokens.

Phoneme inventory (Buckwalter-ish, ASCII, space-separated tokens):
  consonants: ' b t th j H x d dh r z s sh S D T Z 3 gh f q k l m n h w y
  vowels    : a i u aa ii uu

Scope / non-goals (v1):
  Models word-internal phonology: short vowels, sukun, shadda (gemination),
  tanwin, long vowels / madd via carrier letters, dagger alef, hamza forms,
  ta-marbuta, and the definite-article sun/moon-letter assimilation.

  Does NOT model cross-word tajweed (idghaam, ikhfaa, iqlab of noon saakinah,
  special madd lengths). For a closed-corpus matcher these are deliberately
  omitted: internal consistency between the training target and the matcher
  index matters more than tajweed-perfect output, and the acoustic encoder +
  fuzzy matcher are designed to absorb that variation.
"""

from dataclasses import dataclass, field

# --- Diacritics ------------------------------------------------------------
FATHA = "َ"
KASRA = "ِ"
DAMMA = "ُ"
FATHATAN = "ً"
KASRATAN = "ٍ"
DAMMATAN = "ٌ"
SUKUN = "ْ"
SHADDA = "ّ"
DAGGER_ALEF = "ٰ"  # superscript alef -> long aa

HARAKAT = {FATHA: "a", KASRA: "i", DAMMA: "u"}
TANWIN = {FATHATAN: "a", KASRATAN: "i", DAMMATAN: "u"}
MARKS = set(HARAKAT) | set(TANWIN) | {SUKUN, SHADDA, DAGGER_ALEF}

# --- Letters ---------------------------------------------------------------
ALEF = "ا"
ALEF_WASLA = "ٱ"
ALEF_MAKSURA = "ى"
WAW = "و"
YA = "ي"
LAM = "ل"
TA_MARBUTA = "ة"

# base consonant letter -> phoneme
CONS = {
    "ء": "'",        # ء hamza
    "أ": "'",        # أ
    "ؤ": "'",        # ؤ
    "إ": "'",        # إ
    "ئ": "'",        # ئ
    "ب": "b",
    TA_MARBUTA: "t",      # ة (pausal -> h handled at ayah level)
    "ت": "t",
    "ث": "th",
    "ج": "j",
    "ح": "H",
    "خ": "x",
    "د": "d",
    "ذ": "dh",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "sh",
    "ص": "S",
    "ض": "D",
    "ط": "T",
    "ظ": "Z",
    "ع": "3",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    LAM: "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    WAW: "w",
    YA: "y",
}
ALEF_MADDA = "آ"  # آ -> ' aa

# Sun letters (lam of article assimilates) — by phoneme
SUN = {"t", "th", "d", "dh", "r", "z", "s", "sh", "S", "D", "T", "Z", "l", "n"}

PHONEME_SET = (
    ["'", "b", "t", "th", "j", "H", "x", "d", "dh", "r", "z", "s", "sh",
     "S", "D", "T", "Z", "3", "gh", "f", "q", "k", "l", "m", "n", "h", "w", "y"]
    + ["a", "i", "u", "aa", "ii", "uu"]
)


@dataclass
class Unit:
    base: str
    haraka: str | None = None      # 'a' | 'i' | 'u'
    tanwin: str | None = None      # 'a' | 'i' | 'u'
    sukun: bool = False
    shadda: bool = False
    dagger: bool = False
    raw_marks: list = field(default_factory=list)


def _parse_word(word: str) -> list[Unit]:
    units: list[Unit] = []
    cur: Unit | None = None
    for ch in word:
        if ch in MARKS:
            if cur is None:
                continue  # stray mark, ignore
            if ch in HARAKAT:
                cur.haraka = HARAKAT[ch]
            elif ch in TANWIN:
                cur.tanwin = TANWIN[ch]
            elif ch == SUKUN:
                cur.sukun = True
            elif ch == SHADDA:
                cur.shadda = True
            elif ch == DAGGER_ALEF:
                cur.dagger = True
            cur.raw_marks.append(ch)
        else:
            cur = Unit(base=ch)
            units.append(cur)
    return units


def _emit_consonant(out: list[str], phon: str, u: Unit, final: bool = False) -> None:
    """Emit a consonant (+ gemination) and its trailing vowel.

    When `final` (last sounded letter of an ayah-final word), apply waqf:
      - short case-vowel a/i/u  -> dropped (consonant becomes silent stop)
      - dammatan / kasratan     -> dropped
      - fathatan                -> long aa  (e.g. 3aliiman -> 3aliimaa)
      - dagger alef (long aa)   -> kept
    """
    out.append(phon)
    if u.shadda:
        out.append(phon)
    if u.dagger:
        out.append("aa")
    elif final:
        if u.tanwin == "a":
            out.append("aa")
        # tanwin i/u and short haraka are dropped at pause
    elif u.haraka:
        out.append(u.haraka)
    elif u.tanwin:
        out.append(u.tanwin)
        out.append("n")
    # sukun / bare -> no vowel


def _strip_marks(word: str) -> str:
    return "".join(c for c in word if c not in MARKS)


def g2p_word(word: str, ayah_initial: bool = False, word_final_in_ayah: bool = False) -> list[str]:
    units = _parse_word(word)
    if not units:
        return []

    # --- Divine name الله: long aa is unwritten in this text, force it ---
    if _strip_marks(word) == "الله":
        out = ["'", "a"] if ayah_initial else ["a"]
        out += ["l", "l", "aa", "h"]
        if not word_final_in_ayah and units[-1].haraka:
            out.append(units[-1].haraka)
        return out

    out: list[str] = []
    i = 0

    # --- Definite article: (wasl)alef + lam + root ---
    if (len(units) >= 2
            and units[0].base in (ALEF, ALEF_WASLA)
            and not units[0].haraka
            and units[1].base == LAM
            and not units[1].shadda):
        out.append("'a" if ayah_initial else "a")
        # last token is a compound "'a"/"a"; normalize to tokens
        if out[-1] == "'a":
            out[-1] = "'"
            out.append("a")
        root = units[2] if len(units) >= 3 else None
        if root is not None and CONS.get(root.base) in SUN:
            i = 2  # lam silent; sun letter (already shadda) geminates itself
        else:
            out.append("l")
            i = 2

    while i < len(units):
        u = units[i]
        base = u.base
        nxt = units[i + 1] if i + 1 < len(units) else None

        # Long-vowel carriers (madd)
        if base == ALEF and not u.haraka:
            if out and out[-1] == "a":
                out[-1] = "aa"
            else:
                out.append("aa")
            i += 1
            continue
        if base == ALEF_MAKSURA and not u.haraka:
            if out and out[-1] == "a":
                out[-1] = "aa"
            else:
                out.append("aa")
            i += 1
            continue
        if base == ALEF_WASLA and not u.haraka:
            # bare wasl alef not in article: glottal onset only if ayah-initial
            if ayah_initial and i == 0:
                out.append("'")
            i += 1
            continue
        if base == WAW and not u.haraka and not u.shadda:
            if out and out[-1] == "u":
                out[-1] = "uu"
            else:
                out.append("w")
            i += 1
            continue
        if base == YA and not u.haraka and not u.shadda:
            if out and out[-1] == "i":
                out[-1] = "ii"
            else:
                out.append("y")
            i += 1
            continue

        # Alef madda: hamza + long aa
        if base == ALEF_MADDA:
            out.append("'")
            out.append("aa")
            i += 1
            continue

        # Regular consonant
        phon = CONS.get(base)
        if phon is None:
            i += 1
            continue  # unknown char, skip

        # ta-marbuta at ayah end -> pausal /h/
        if base == TA_MARBUTA and word_final_in_ayah and nxt is None:
            out.append("h")
            i += 1
            continue

        # fatha + following bare alef -> long aa (handled when we reach alef),
        # but emit the consonant + short vowel now; alef lookahead lengthens it.
        is_final = word_final_in_ayah and nxt is None
        _emit_consonant(out, phon, u, final=is_final)
        i += 1

    return out


def g2p_ayah(text: str) -> list[str]:
    words = text.split()
    phonemes: list[str] = []
    for wi, word in enumerate(words):
        phonemes.extend(
            g2p_word(
                word,
                ayah_initial=(wi == 0),
                word_final_in_ayah=(wi == len(words) - 1),
            )
        )
    return phonemes


# --- self-test -------------------------------------------------------------
if __name__ == "__main__":
    import json
    from pathlib import Path

    samples = {
        "112:1": "Qul huwa Allahu ahad",
        "112:2": "Allahu as-samad",
        "114:1": "Qul a3udhu bi rabbi n-nas",
        "78:1": "3amma yatasaa'aluun",
    }
    p = Path(__file__).parent / "manifests" / "ayah_text.json"
    if p.exists():
        txt = json.load(open(p, encoding="utf-8"))
        for key, gloss in samples.items():
            if key in txt:
                ph = g2p_ayah(txt[key])
                print(f"{key} ({gloss}):")
                print("   ", " ".join(ph))
                print()
