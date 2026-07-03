# conformance/ — C++ SDK port acceptance test

Golden fixtures generated from the Python reference, so the cross-platform C++ engine
(see `docs/sdk-architecture.md`) can be validated bit-for-bit-ish. This was the **first
deliverable** of the SDK effort — it de-risked the port before any C++ was written.

**Status: the C++ core passes this harness end-to-end.** `sdk/core/tests/conformance_runner`
emits the front-end + matcher outputs and `verify.py --candidate` reports ALL PASS; the
ORT-backed stages and full detector are checked by `test_inference` / `test_detector`. See
`sdk/README.md`.

```bash
python conformance/generate.py                       # (re)build golden from the reference
python conformance/verify.py                          # self-check (reference vs golden) — must pass
python conformance/verify.py --candidate <out_dir>    # check the C++ port's outputs (-> ALL PASS)
```

- `spec.md` — the exact computation the port must reproduce (front-end DSP, matcher /
  sliding segmenter, formats, tolerances). **This is the contract.**
- `generate.py` — produces `assets/` (exact mel filterbank + Hann window + lexicon),
  `fixtures/` (inputs), `golden/` (expected outputs), `manifest.json`.
- `verify.py` — comparison logic: front-end log-mel within `1e-2`; matcher event sequences
  exact. Self-check recomputes from the reference (currently 0.0 diff, ALL PASS).

Three port-risk stages are covered independently: **front-end** (WAV→log-mel; the C++ reuses
the provided filterbank/window constants so only STFT+matmul+log can differ),
**matcher/segmenter** (phonemes→ayah events, decoupled from the model), and the **Stage-3
highlight controller** (committed detections→render-ready state snapshots; model-independent,
the SDK's output contract — `golden/highlight/`, exact match, C++ port byte-identical). Model
inference is ONNX Runtime (shared engine, parity already established — not re-implemented).

`gen_highlight()` in `generate.py` is self-contained (needs only `data/lang/ambiguous_ayat.json`
+ the controller), so the highlight golden can be regenerated without the model/audio.

Fixtures derive from dataset audio + `best_mic.pt`; `.wav`/`.bin` are regenerated, not
committed (audio rule) — hand the generated package to the port team or regenerate where
data+model are present.
