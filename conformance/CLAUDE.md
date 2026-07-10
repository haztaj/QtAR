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
- `generate.py` — produces `assets/` (exact mel filterbank + Hann window + lexicon +
  `ambiguous_ayat.json` + the Silero VAD `silero_vad.onnx`, copied from the pip `silero-vad`
  package for the C++ core + Android build), `fixtures/` (inputs), `golden/` (expected outputs),
  `manifest.json`.
- `verify.py` — comparison logic: front-end log-mel within `1e-2`; matcher event sequences
  exact. Self-check recomputes from the reference (currently 0.0 diff, ALL PASS).

Four port-risk stages are covered independently: **front-end** (WAV→log-mel; the C++ reuses
the provided filterbank/window constants so only STFT+matmul+log can differ),
**matcher/segmenter** (phonemes→ayah events, decoupled from the model), the **Stage-3
highlight controller** (committed detections→render-ready state snapshots; model-independent,
the SDK's output contract — `golden/highlight/`, exact match, C++ port byte-identical), and
the **unit-chain decoder** (spec.md §Stage 2b; phoneme streams→unit chains, `golden/chain/`,
exact match; `--only chain` regenerates just this section; additionally cross-validated EXACT
over 200 real decoded streams). Model inference is ONNX Runtime (shared engine, parity
already established — not re-implemented).

> **Coverage limit (2026-07-11):** the chain fixtures are SHORT SYNTHETIC phoneme streams — they
> pin C++↔Python decoder fidelity, NOT decode quality or the live rolling-audio buffer. They
> cannot exhibit the 22 s rolling-window CROWDING of short units on real phone audio (a live bug
> found this session — see research/CLAUDE.md "Rolling-window CROWDING"). Golden pass ≠ good
> on-device tracking; long real-audio behaviour must be validated through the full Detector.

A fifth stage covers the **true-streaming acoustic path** (spec.md §Streaming model inference;
`sdk/core/src/streaming.*` — incremental conv-cache + 48-state threading + cross-chunk CTC
collapse, which IS re-implemented in C++, unlike plain inference). `golden/streaming/*.phonemes.txt`
pins the Python `StreamingRuntime` output over each frontend log-mel fixture (fp32 graphs exported
per-checkpoint into `assets/stream_{conv,encoder}.onnx`, gitignored); `test_streaming <conf>
assets/stream_conv.onnx assets/stream_encoder.onnx` reproduces it EXACTLY (ALL PASS).

`gen_highlight()` in `generate.py` is self-contained (needs only `data/lang/ambiguous_ayat.json`
+ the controller), so the highlight golden can be regenerated without the model/audio.

Fixtures derive from dataset audio + `best_mic.pt`; `.wav`/`.bin` are regenerated, not
committed (audio rule) — hand the generated package to the port team or regenerate where
data+model are present.
