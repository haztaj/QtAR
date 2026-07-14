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

## Stage 3 — highlight controller (detections → render-ready state)  [SDK output contract]

Post-commit, model-independent: fixtures provide a sequence of *committed* detections
directly, so this tests the pure state machine (`matcher/highlight_controller.py`, ported to
`sdk/core/src/highlight.{h,cpp}`). It owns the deferral + ambiguity handling and emits the
snapshot every platform renders — the centralized highlight logic.

**Confusable map** — `assets/ambiguous_ayat.json` (produced by `matcher/find_ambiguous.py`).
Per ambiguous ayah: `confusable_with`, `predecessor`, `successor`. The controller's class for
a detected key = `{key} ∪ confusable_with`, deduped and sorted by `(surah, ayah)`.

**Per-detection logic** (`detect(key)`):
1. If a pending `await_successor` exists and `key` uniquely matches one option's `successor`
   → retroactively **confirm** that option (else the pending downgrades to `needs_choice`).
2. If `key` is **not** ambiguous → confirm it.
3. Else, with `last` = the last confirmed ayah:
   - predecessor pins it (exactly one option has `predecessor == last`) → confirm now;
   - else if every option has a distinct non-empty `successor` → hold **pending**
     `await_successor` (no guess: `active` stays unchanged — the deferral);
   - else → **pending** `needs_choice` (surface `options` for a manual `choose(key)`).

`choose(key)` confirms `key` iff it's in the pending options. `confirm` clears pending,
appends to `confirmed`, and sets `active = key`.

**Snapshot** — `HighlightState.to_dict()`:
```json
{ "confirmed": ["S:A", ...],
  "pending": null | {"ayah": null|"S:A", "options": ["S:A", ...], "reason": "await_successor|needs_choice"},
  "active": null | "S:A" }
```
**Comparison:** the full snapshot after each step must match **exactly** (`states` in
`golden/highlight/<name>.states.json`). Fixtures: `fixtures/highlight/<name>.json`
(`{"steps": [{"detect": "S:A"} | {"choose": "S:A"}, ...]}`).

**Public-snapshot extras (detector layer, NOT this controller):** the public
`HighlightSnapshot` the host renders carries two fields the controller doesn't own,
added by `Detector` on top of the controller state (so they're outside this golden):
`upNext` (predicted same-surah successor, revealed near completion) and, for `Mode::Chain`,
`activeSegment` / `activeSegmentCount` — waqf-segment progress within `active` ("part N of M";
count 0 = non-Chain/none, 1 = unsegmented ayah, N = split into N segments). Verified by
`tests/test_detector` (prints `segment S:A N/M`); the segment counts come from `UnitIndex`.

---

## Stage 2b — unit-chain decoder (phoneme stream → unit chain)  [research winning design]

Reference: `research/chain_sliding.py` (`decode_sliding`, `windowBest` internals,
`assemble`, `make_succ_full`) — the design validated on 747 continuous 4-ayah sequences
(aligned-hit 87.3%, ayah-chain SER 13.3%). C++ port: `sdk/core/src/chain.{h,cpp}`.

Units = waqf segments ("S:A#NN") + unsegmented ayat ("S:A") from
`assets/unit_phonemes.json`; the ayah is the derived (parent) label.

Pipeline the port must reproduce EXACTLY over a decoded phoneme stream (per-phoneme
times in seconds):

1. **Multi-scale windows** — scales (0.2, 0.7, 1.0, 1.5, 2.2) × window_s (10), hop 1.5 s;
   per scale, windows [t, t+W) for t = 0, 1.5, 3.0 … while t ≤ t_end + 1e-6. Windows with
   < 4 phonemes are skipped.
2. **windowBest** — 3-gram inverted index retrieval (posting lists in canonical
   (surah, ayah, segment) order); shortlist = raw-count top-60 UNION length-normalized
   (count/L) top-20 over the FULL counter, tie-breaks by first-insertion order (Python
   `Counter.most_common` / `heapq.nlargest` stability); tight length gate 0.5n ≤ L ≤ 1.3n;
   infix edit-norm (ref as substring of window, free edge gaps, / len(ref)); blended
   selection sel = cost − 0.15·min(L,n)/n among fires ≤ 0.30.
3. **Vote machine** — fires sorted by (w1, key, cost); consumed-time gate (MIN_ADVANCE 2.0,
   consumed = w1 − 2.0 on commit), same-unit / REPEAT_SUPPRESS (20 s, first occurrence)
   gates, exact-twin substitution toward the expected successor, votes_next=1 /
   votes_jump=2, strong fires (≤ 0.15) commit with one vote.
4. **Assembly** — 2-deep pending deferral: expected successor / same-parent forward
   confirms; an unexpected jump defers until a later emission supports it (junk
   tolerance 1); backward/repeat drops; end-of-stream flush of the chainable tail.

**Phase-2 posterior-aware scoring (`sub_min < 1`):** when the fixture stream carries per-
phoneme `alts` (top-k `[[token, prob], ...]`) and `params.sub_min < 1`, a substitution of
ref phoneme `ri` for a mismatching window phoneme costs `max(sub_min, 1 - p(ri)/p(greedy))`
if the model CONSIDERED `ri` (in that position's alts), else a full 1 — the soft branch of
`_infix_norm`. Off (`sub_min >= 1` / no alts) == the hard 0/1 distance. The
`soft_score_run` fixture pins it. (Note: retrieval counting must dedup per position via an
ORDERED map, not a set — a set's hash-randomized iteration breaks the Counter tie-break
determinism, the same failure mode as the sorted-tuple posting lists.)

**Comparison:** `emitted` (unit key sequence) and `assembled` (chain) must match
**exactly** (`golden/chain/<name>.chain.json`). Fixtures: `fixtures/chain/<name>.json`
(`{"stream": {"phonemes": [...], "times": [...], "alts"?: [...]}, "params": {...}}`). Beyond
the committed fixtures, the port was cross-validated EXACT over 200 real decoded test streams
(greedy 2026-07-07; Phase-2 soft path over 200 noisy streams 2026-07-10).

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

## Streaming model inference (incremental Emformer — `StreamingModel`)  [optional, Mode::Chain]

The true-streaming acoustic path (`sdk/core/src/streaming.{h,cpp}`) decodes only the NEW audio
each hop instead of re-decoding the whole window (see `export/streaming-export-plan.md`). It runs
**two** ONNX graphs — `stream_conv.onnx` (dynamic-T Conv2dSubsampling) + `stream_encoder.onnx`
(one fixed-shape STATEFUL Emformer step: chunk + 48 state tensors → log_probs + 48 states) — with
the C++ threading the conv boundary cache + the 48 states across `feed()` calls and collapsing CTC
greedy across chunk boundaries. This must reproduce the Python `StreamingRuntime` **exactly**.

Validated by `tests/test_streaming.cpp`: feed each fixture's golden log-mel in **20-frame chunks**
through the C++ `StreamingModel` and compare the phoneme-id sequence to
`golden/streaming/<name>.phonemes.txt` (produced by the Python runtime over the same graphs).

```bash
./test_streaming ../conformance assets/stream_conv.onnx assets/stream_encoder.onnx   # -> ALL PASS
```

Notes:
- **Golden + test use the fp32 encoder** (`assets/stream_encoder.onnx`), for the same reason as
  Model inference above — int8 argmax can flip on a **cross-ORT quantization tie** (Python pip ORT
  vs the C++-linked ORT round the MatMulInteger differently), so exactness requires the same fp32
  graph on both sides. On-device the C++ int8 stream is self-consistent; validated end-to-end by
  `test_detector --chain <conv> <enc>` (streaming detections == the windowed re-decode, exact).
- **Graphs are exported per-checkpoint into `assets/`** by `generate.py` (gitignored, like
  `silero_vad.onnx`) so the golden is self-contained and regenerates anywhere the checkpoint is
  present. The `feed` chunking (20 frames) is part of the contract — the golden is generated with it.

---

## File formats

- **`*.bin`** — float32 little-endian, row-major. `logmel` `[T,80]`, `mel_filterbank`
  `[201,80]`, `hann_window` `[400]`. Shapes are in `manifest.json`.
- **`golden/matcher/*.events.json`** — `{"events": [ ... ]}` as above.
- **`fixtures/matcher/*.json`** — `{"windows": [[ph,...], ...], "config": {...}}`.
- **`golden/highlight/*.states.json`** — `{"states": [ <snapshot>, ... ]}`, one per step.
- **`fixtures/highlight/*.json`** — `{"steps": [{"detect": "S:A"} | {"choose": "S:A"}, ...]}`.
- **`assets/`** — `mel_filterbank.bin`, `hann_window.bin`, `tokens.txt`,
  `ayah_phonemes.json` (the Stage-2 lexicon), `edit_cases.json`,
  `ambiguous_ayat.json` (the Stage-3 confusable map).
- **`manifest.json`** — front-end + matcher fixture index, all DSP params, tolerances.

## Candidate output layout (for `--candidate <dir>`)

Write, into `<dir>`, one file per fixture using the **basename** from golden:
- front-end: `<name>.logmel.bin` (float32, shape per manifest).
- matcher: `<name>.events.json` (`{"events": [...]}`), produced by running the candidate
  segmenter over that fixture's `windows`.
- highlight: `<name>.states.json` (`{"states": [...]}`), produced by running the candidate
  `HighlightController` over that fixture's `steps`.

## Notes

- Fixtures are generated from dataset audio + `best_mic.pt`; the `.wav`/`.bin` are
  derived (audio isn't committed per project rule) — deliver the generated package to the
  port team out-of-band, or regenerate with `generate.py` where the data+model are present.
- Add more fixtures (more reciters, noisier/quiet-mic clips, jump/restart sessions) as the
  port matures — especially edge cases the C++ must get right.
