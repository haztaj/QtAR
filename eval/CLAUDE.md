# eval/ — end-to-end ayah-ID evaluation

Ties Stage 1 + Stage 2 together and reports the metric that actually matters
(ayah-ID accuracy), as opposed to the Stage-1 PER proxy.

## `evaluate.py`

```bash
python eval/evaluate.py --checkpoint training/exp/best.pt --split val --limit 300
```

Pipeline per clip: audio → log-mel → Emformer+CTC → **greedy** phoneme stream
(`greedy_phonemes`) → `PhonemeMatcher` → ranked ayat.

Reports:
- top-1 / top-3 ayah-ID accuracy
- mean time-to-detection (fraction of phonemes until the true ayah first hits top-1)
- commit rate + false-commit rate at a norm-cost margin (`--commit-margin`)

## Notes

- This is the integration harness; numbers are only meaningful once Stage 1 is
  trained to convergence. A barely-trained checkpoint will score poorly — that
  still validates the wiring.
- Greedy decoding is the v1 bridge. A posterior-aware bridge (feed CTC frame
  posteriors into the matcher beam instead of a 1-best phoneme string) is the
  obvious next accuracy lever, but greedy is the right baseline first.
- **Still TODO** per the brief: time-to-detection in seconds (not just phoneme
  fraction), robustness curves across SNR and proficiency, a dedicated
  mispronounced-but-correct vs different-ayah slice, and on-device RTF/memory.
  These come after the model trains and the RetaSy learner set is wired in.
