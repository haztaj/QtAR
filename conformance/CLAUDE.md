# conformance/ — C++ SDK port acceptance test

Golden fixtures generated from the Python reference, so the cross-platform C++ engine
(see `docs/sdk-architecture.md`) can be validated bit-for-bit-ish. This is the **first
deliverable** of the SDK effort — de-risks the port before any C++ is written.

```bash
python conformance/generate.py                       # (re)build golden from the reference
python conformance/verify.py                          # self-check (reference vs golden) — must pass
python conformance/verify.py --candidate <out_dir>    # check the C++ port's outputs
```

- `spec.md` — the exact computation the port must reproduce (front-end DSP, matcher /
  sliding segmenter, formats, tolerances). **This is the contract.**
- `generate.py` — produces `assets/` (exact mel filterbank + Hann window + lexicon),
  `fixtures/` (inputs), `golden/` (expected outputs), `manifest.json`.
- `verify.py` — comparison logic: front-end log-mel within `1e-2`; matcher event sequences
  exact. Self-check recomputes from the reference (currently 0.0 diff, ALL PASS).

Two port-risk stages are covered independently: **front-end** (WAV→log-mel; the C++ reuses
the provided filterbank/window constants so only STFT+matmul+log can differ) and
**matcher/segmenter** (phonemes→ayah events, decoupled from the model). Model inference is
ONNX Runtime (shared engine, parity already established — not re-implemented).

Fixtures derive from dataset audio + `best_mic.pt`; `.wav`/`.bin` are regenerated, not
committed (audio rule) — hand the generated package to the port team or regenerate where
data+model are present.
