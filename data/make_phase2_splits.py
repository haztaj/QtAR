#!/usr/bin/env python3
"""
Build phase-2 (learner-adaptation) manifests.

- Split RetaSy by RECITER (no leakage): ~80% reciters -> retasy_train, ~20% -> retasy_test.
- combined_train.csv = main-train clips (clean studio) + retasy_train clips (learners).
- retasy_test.csv    = held-out learner reciters, for the honest learner number.

Reciter-level holdout means the learner test set is speakers the model never trained on.

Outputs -> data/raw/phase2/
"""

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
sys.path.insert(0, str(DATA_DIR.parent / "training"))
from data import reciter_split  # main-manifest train/val/test by reciter

MAIN_MANIFEST = DATA_DIR / "raw" / "audio" / "manifest.csv"
RETASY_MANIFEST = DATA_DIR / "raw" / "retasy_audio" / "manifest.csv"
OUT_DIR = DATA_DIR / "raw" / "phase2"

COMMON = ["recording_id", "reciter_id", "surah_id", "ayah_id", "path", "duration"]
RETASY_TEST_FRAC = 0.20
SEED = 1234
# RetaSy clips whose audio doesn't match the labeled ayah — never use for train/eval.
BAD_LABELS = {"not_related_quran", "not_match_aya", "multiple_aya", "in_complete"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- main (clean) train clips ---
    main = pd.read_csv(MAIN_MANIFEST)
    smap = reciter_split(main["reciter_id"].unique().tolist())
    main_train = main[main["reciter_id"].map(smap) == "train"].copy()
    main_train = main_train[COMMON]
    main_train["source"] = "quran_md"

    # --- RetaSy: drop bad-label clips, split by reciter ---
    ret = pd.read_csv(RETASY_MANIFEST)
    if "final_label" in ret.columns:
        ret = ret[~ret["final_label"].isin(BAD_LABELS)].copy()
    reciters = sorted(ret["reciter_id"].astype(str).unique())
    rng = __import__("random").Random(SEED)
    rng.shuffle(reciters)
    n_test = max(1, int(len(reciters) * RETASY_TEST_FRAC))
    test_reciters = set(reciters[:n_test])

    ret["reciter_id"] = ret["reciter_id"].astype(str)
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
