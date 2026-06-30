# Conformance spec — C++ SDK port acceptance test

The C++ engine must reproduce the Python reference. This spec defines the exact
computation, the fixture formats, and the tolerances. `generate.py` produces the golden
fixtures; `verify.py` checks a candidate against them. See `docs/sdk-architecture.md` §10.

```
python conformance/generate.py                       # (re)build golden from the reference
python conformance/verify.py                          # self-check (reference vs golden)
python conformance/verify.py --candidate <out_dir>    # check the C++ port's outputs
```

A candidate passes when **every** front-end log-mel is within tolerance and **every**
matcher event sequence matches exactly.

---

## Stage 1 — front-end DSP (WAV → log-mel)  [#1 port risk]

Input: mono float32 PCM at 16 kHz. Output: log-mel `[T, 80]`, row-major.

Steps (must match `training/data.py: logmel_16k` exactly):

1. **RMS-normalize** to `NORM_RMS = 0.1`:
   `rms = sqrt(mean(x^2)); if rms > 1e-6: x *= 0.1/rms; x = clamp(x, -1, 1)`.
2. **STFT** — `n_fft = 400`, `hop = 160`, `win_length = 400`, **Hann window (periodic)**,
   `center = True` (reflect-pad `n_fft//2 = 200` samples each side), `power = 2.0`
   (magnitude-squared). Frames: `T = 1 + len(x)//hop`. Bins: `n_fft/2+1 = 201`.
   - Use the provided **`assets/hann_window.bin`** `[400]` f32 verbatim (don't recompute).
3. **Mel filterbank** — HTK scale, 80 mels, `f_min = 20`, `f_max = 8000`, no norm:
   `mel[t] = power_spectrum[t] (1x201) · filterbank (201x80)` → `[T, 80]`.
   - Use the provided **`assets/mel_filterbank.bin`** `[201, 80]` f32 verbatim. This
     removes any mel-formula mismatch — the C++ only does the STFT + matmul + log.
4. **Log**: `log(max(mel, 1e-10))`.
5. Result is already `[T, 80]` (time-major).

Resampling to 16 kHz (if the device captures another rate) is a separate standard step,
not covered by these fixtures (inputs are already 16 kHz).

**Tolerance:** `max |candidate - golden| <= 1e-2` over the whole `[T,80]` tensor. The
reference self-check is exact (0.0); the tolerance allows for FFT/float ordering
differences. Tighten once the port is stable.

---

## Stage 2 — matcher / sliding-window segmenter (phonemes → ayah events)  [#2 port risk]

Decoupled from the model: fixtures provide the per-window phoneme lists directly, so this
tests the pure logic (`matcher/phoneme_matcher.py` + `demo/sliding.py`).

**Normalized edit distance** (`_edit_norm(a, b)`): Levenshtein(a, b) / max(len(a), len(b)).
Unit cases in `assets/edit_cases.json` must match exactly.

**Per-window classification** (`SlidingWindowSegmenter._window_best`): for window phonemes
`w` (length `n`), over every ayah `key` with phonemes `p` (length `L`):
- length-prune: skip unless `0.6*n <= L <= n/0.6`;
- score `= _edit_norm(w, p) - context.bonus_for(key)`;
- best = min score. A window is **confident** if `best <= max_cost (0.30)` and `n >= 3`.

**Sequential context bonus** (`SequentialContext.bonus_for(key)`), given `current`:
- `eff = bonus(0.22) + streak_bonus(0.05) * streak`;
- next `window`(2) ayat in canonical (surah,ayah) order: `eff * (1 - (j-1)/(window+1))`
  for the j-th next (j=1..2);
- same surah as `current`: `max(prev, surah_bonus=0.10)`;
- return the max applicable. `set_current` grows `streak` only when the new ayah is the
  expected next (else resets); canonical order spans surah boundaries (112:4 → 113:1).

**State machine** (`process`, per window in order): if not confident → no event. Else:
- `current is None` → set current, emit **detect**;
- `key == current` → no event;
- `key == expected_next` → set current, emit **advance**;
- else → needs `jump_votes`(2) consecutive same-key windows → emit **jump**.

Event: `{"event": "detect|advance|jump", "ayah": "S:A", "t": float, "cost": float}`.
**Comparison:** the ordered sequence of `(event, ayah)` must match exactly (`t`/`cost`
are informational).

---

## Model inference (ONNX Runtime — same engine as Python)

The acoustic model runs via **ONNX Runtime** on all platforms, so it isn't re-implemented.
Validated by `tests/test_inference.cpp`: feed each fixture's golden log-mel through the
C++ ORT session + CTC greedy decode (argmax → collapse repeats → drop blank id 0; tokens
in `assets/tokens.txt`) and compare the phoneme-id sequence to `golden/inference/<name>.phonemes.txt`.

```bash
g++ -std=c++17 -O2 -D_stdcall=__stdcall -I ../include -I ../src -I <ort>/include \
    test_inference.cpp ../src/inference.cpp ../src/decoder.cpp ../src/assets.cpp \
    <ort>/lib/onnxruntime.dll -o test_inference            # MinGW: -D_stdcall=__stdcall
./test_inference ../conformance ../export/onnx/model.onnx  # -> ALL PASS (same-model)
```

Notes:
- **Golden uses the fp32 export.** int8 vs fp32 differ on the occasional borderline frame,
  so the inference test must compare the C++ to the **same** model it runs (same-model =
  exact). On-device int8 is validated on the target ORT.
- **int8 deployment (resolved):** the shipped int8 is **weight-only dynamic quant restricted
  to MatMul** (`op_types_to_quantize=["MatMul"]`). Earlier full `quantize_dynamic` also
  quantized the Conv2dSubsampling → a `ConvInteger` op some ORT CPU builds (e.g. desktop 1.18)
  can't execute. `MatMulInteger` is supported everywhere and holds ~all the size win, so
  leaving the tiny conv in fp32 makes int8 run on every EP while staying argmax-lossless
  (100% vs fp32). **Static QDQ was rejected:** it quantizes activations, and this Emformer's
  attention/LayerNorm outliers blow up under static int8 — the phoneme argmax (and detection)
  collapsed. Don't re-introduce QDQ for this model. The int8 model is validated end-to-end by
  `test_detector` (114:1→2→3); the off-by-one phoneme `test_inference` shows on one fixture is
  the expected int8≠fp32 borderline-frame flip, absorbed by the matcher.
- **MinGW:** ORT's header defines the calling convention as `_stdcall` (single underscore),
  which g++ rejects; build with `-D_stdcall=__stdcall` (harmless on x64).

---

## File formats

- **`*.bin`** — float32 little-endian, row-major. `logmel` `[T,80]`, `mel_filterbank`
  `[201,80]`, `hann_window` `[400]`. Shapes are in `manifest.json`.
- **`golden/matcher/*.events.json`** — `{"events": [ ... ]}` as above.
- **`fixtures/matcher/*.json`** — `{"windows": [[ph,...], ...], "config": {...}}`.
- **`assets/`** — `mel_filterbank.bin`, `hann_window.bin`, `tokens.txt`,
  `ayah_phonemes.json` (the Stage-2 lexicon), `edit_cases.json`.
- **`manifest.json`** — front-end + matcher fixture index, all DSP params, tolerances.

## Candidate output layout (for `--candidate <dir>`)

Write, into `<dir>`, one file per fixture using the **basename** from golden:
- front-end: `<name>.logmel.bin` (float32, shape per manifest).
- matcher: `<name>.events.json` (`{"events": [...]}`), produced by running the candidate
  segmenter over that fixture's `windows`.

## Notes

- Fixtures are generated from dataset audio + `best_mic.pt`; the `.wav`/`.bin` are
  derived (audio isn't committed per project rule) — deliver the generated package to the
  port team out-of-band, or regenerate with `generate.py` where the data+model are present.
- Add more fixtures (more reciters, noisier/quiet-mic clips, jump/restart sessions) as the
  port matures — especially edge cases the C++ must get right.
