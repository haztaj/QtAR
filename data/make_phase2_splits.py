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
FLAGS = DATA_DIR / "raw" / "retasy_audio" / "flags.csv"   # bucket baseline (retasy_flag.py)
VERDICTS = DATA_DIR / "retasy_verdicts.json"   # committed human overrides (not under raw/)
OUT_DIR = DATA_DIR / "raw" / "phase2"
# clips in these buckets are discarded by DEFAULT (retasy_review.py pre-verdicts them);
# the review page spot-samples them, so most are never explicitly touched -> the baseline
# is what actually removes them. Human verdicts override per-clip.
AUTO_DISCARD_BUCKETS = {"silent", "noise_only", "too_short", "garbage"}

COMMON = ["recording_id", "reciter_id", "surah_id", "ayah_id", "path", "duration"]
RETASY_TEST_FRAC = 0.20
SEED = 1234
# RetaSy clips whose audio doesn't match the labeled ayah — never use for train/eval.
BAD_LABELS = {"not_related_quran", "not_match_aya", "multiple_aya", "in_complete"}


def clean_retasy(ret: pd.DataFrame) -> pd.DataFrame:
    """Resolve one effective cleanup decision per clip and apply it. Priority (human wins):

      relabel   (verdicts)  -> KEEP + re-key to the corrected ayah   (highest)
      discard   (verdicts)  -> DROP
      keep      (verdicts)  -> KEEP (rescues a baseline drop)
      BAD_LABELS + flag AUTO_DISCARD_BUCKETS (baseline) -> DROP

    So a human relabel/keep overrides both the old `final_label` blacklist AND the flag
    bucket baseline — e.g. a `not_match_aya` clip the reviewer re-identified is rescued,
    not dropped (the ordering bug this replaces)."""
    v = json.loads(VERDICTS.read_text(encoding="utf-8")) if VERDICTS.exists() else {}
    keep, hdiscard, relabel = set(v.get("keep", [])), set(v.get("discard", [])), v.get("relabel", {})
    protected = keep | set(relabel)                       # never auto-dropped

    discard: set[str] = set()
    if "final_label" in ret.columns:
        bad = set(ret.loc[ret["final_label"].isin(BAD_LABELS), "recording_id"]) - protected
        discard |= bad
        print(f"final_label blacklist: {len(bad)} dropped ({len(protected & set(ret.loc[ret['final_label'].isin(BAD_LABELS), 'recording_id']))} rescued by review)")
    if FLAGS.exists():
        f = pd.read_csv(FLAGS, encoding="utf-8")
        bucket_bad = set(f.loc[f["bucket"].isin(AUTO_DISCARD_BUCKETS), "recording_id"]) - protected
        discard |= bucket_bad
        print(f"flag baseline: {len(bucket_bad)} in auto-discard buckets "
              f"(rescued {len([r for r in keep if r in set(f.loc[f['bucket'].isin(AUTO_DISCARD_BUCKETS),'recording_id'])])} by review)")
    elif not v:
        print(f"(no {FLAGS.name} / {VERDICTS.name} — final_label blacklist only; "
              f"run data/retasy_flag.py + data/retasy_review.py to clean further)")
    discard |= hdiscard                                   # explicit human discards
    discard -= protected                                 # ...but relabel/keep always win

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
    print(f"RetaSy cleanup: kept {len(ret)} / {n0} (dropped {n0 - len(ret)}), relabeled {n_rel}")
    return ret


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- main (clean) train clips ---
    main = pd.read_csv(MAIN_MANIFEST)
    smap = reciter_split(main["reciter_id"].unique().tolist())
    main_train = main[main["reciter_id"].map(smap) == "train"].copy()
    main_train = main_train[COMMON]
    main_train["source"] = "quran_md"

    # --- RetaSy ---
    ret = pd.read_csv(RETASY_MANIFEST)
    ret["reciter_id"] = ret["reciter_id"].astype(str)

    # Reciter split is computed on the STABLE universe (BAD_LABELS-filtered) so test_reciters
    # is identical to the best_s123_mic runs — cleaning must not reshuffle the holdout.
    universe = ret[~ret["final_label"].isin(BAD_LABELS)] if "final_label" in ret.columns else ret
    reciters = sorted(universe["reciter_id"].unique())
    rng = __import__("random").Random(SEED)
    rng.shuffle(reciters)
    n_test = max(1, int(len(reciters) * RETASY_TEST_FRAC))
    test_reciters = set(reciters[:n_test])

    # Clean clips (human verdicts override BAD_LABELS + flag buckets), then assign by reciter.
    ret = clean_retasy(ret)
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
