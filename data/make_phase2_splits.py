#!/usr/bin/env python3
"""
Build phase-2 (learner-adaptation) manifests.

- Split RetaSy by RECITER (no leakage): ~80% reciters -> retasy_train, ~20% -> retasy_test.
- combined_train.csv = main-train clips (clean studio) + retasy_train clips (learners).
- retasy_test.csv    = held-out learner reciters, for the honest learner number.

Reciter-level holdout means the learner test set is speakers the model never trained on.

Cleanup verdicts (data/retasy_review.py -> review_verdicts.json) are merged on top of the
pre-existing `final_label` blacklist: `discard` clips are dropped, `relabel` clips have
their surah/ayah corrected. The RECITER split is computed BEFORE applying verdicts, so
the held-out learner set stays the same reciters regardless of cleaning (honest holdout).

Outputs -> data/raw/phase2/
"""

import json
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
sys.path.insert(0, str(DATA_DIR.parent / "training"))
from data import reciter_split  # main-manifest train/val/test by reciter

MAIN_MANIFEST = DATA_DIR / "raw" / "audio" / "manifest.csv"
RETASY_MANIFEST = DATA_DIR / "raw" / "retasy_audio" / "manifest.csv"
VERDICTS = DATA_DIR / "retasy_verdicts.json"   # committed human labor (not under raw/)
OUT_DIR = DATA_DIR / "raw" / "phase2"

COMMON = ["recording_id", "reciter_id", "surah_id", "ayah_id", "path", "duration"]
RETASY_TEST_FRAC = 0.20
SEED = 1234
# RetaSy clips whose audio doesn't match the labeled ayah — never use for train/eval.
BAD_LABELS = {"not_related_quran", "not_match_aya", "multiple_aya", "in_complete"}


def apply_verdicts(ret: pd.DataFrame) -> pd.DataFrame:
    """Drop `discard` clips and re-key `relabel` clips per review_verdicts.json (if present).
    Reciter is unchanged, so a clip stays in whichever split its reciter was assigned."""
    if not VERDICTS.exists():
        print(f"(no {VERDICTS.name} — using final_label blacklist only; "
              f"run data/retasy_flag.py + data/retasy_review.py to clean further)")
        return ret
    v = json.loads(VERDICTS.read_text(encoding="utf-8"))
    discard = set(v.get("discard", []))
    relabel = v.get("relabel", {})
    n0 = len(ret)
    ret = ret[~ret["recording_id"].isin(discard)].copy()
    n_rel = 0
    for rid, key in relabel.items():
        m = ret["recording_id"] == rid
        if m.any():
            s, a = key.split(":")
            ret.loc[m, "surah_id"] = int(s)
            ret.loc[m, "ayah_id"] = int(a)
            n_rel += 1
    print(f"verdicts: dropped {n0 - len(ret)} discard, relabeled {n_rel} "
          f"(from {VERDICTS.name})")
    return ret


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- main (clean) train clips ---
    main = pd.read_csv(MAIN_MANIFEST)
    smap = reciter_split(main["reciter_id"].unique().tolist())
    main_train = main[main["reciter_id"].map(smap) == "train"].copy()
    main_train = main_train[COMMON]
    main_train["source"] = "quran_md"

    # --- RetaSy: drop bad-label clips, then split by reciter, then apply cleanup verdicts ---
    ret = pd.read_csv(RETASY_MANIFEST)
    if "final_label" in ret.columns:
        ret = ret[~ret["final_label"].isin(BAD_LABELS)].copy()
    reciters = sorted(ret["reciter_id"].astype(str).unique())
    rng = __import__("random").Random(SEED)
    rng.shuffle(reciters)
    n_test = max(1, int(len(reciters) * RETASY_TEST_FRAC))
    test_reciters = set(reciters[:n_test])

    ret["reciter_id"] = ret["reciter_id"].astype(str)
    ret = apply_verdicts(ret)                 # cleanup AFTER the split is fixed
    ret_test = ret[ret["reciter_id"].isin(test_reciters)].copy()
    ret_train = ret[~ret["reciter_id"].isin(test_reciters)].copy()
    for d in (ret_train, ret_test):
        d["source"] = "retasy"

    combined = pd.concat([main_train, ret_train[COMMON + ["source"]]], ignore_index=True)

    combined.to_csv(OUT_DIR / "combined_train.csv", index=False)
    ret_test[COMMON + ["source"]].to_csv(OUT_DIR / "retasy_test.csv", index=False)

    print(f"RetaSy reciters: {len(reciters)} -> {len(test_reciters)} test / "
          f"{len(reciters) - len(test_reciters)} train")
    print(f"combined_train.csv : {len(combined)} clips "
          f"({len(main_train)} clean + {len(ret_train)} retasy-train)")
    print(f"retasy_test.csv    : {len(ret_test)} clips "
          f"({ret_test['reciter_id'].nunique()} held-out learner reciters, "
          f"{ret_test['surah_id'].nunique()} surahs)")


if __name__ == "__main__":
    main()
