# Streaming-Emformer export — scope

Status: **planning** (not built). This scopes the "true streaming" export flagged as a TODO
in `export/CLAUDE.md`, and clarifies what it does and does not buy us for the detection
modes.

## Why (the payoff)

Today the acoustic side is **windowed**: the SDK / demo re-decode a fixed or growing audio
buffer every hop and feed the whole thing through `EmformerCTC.forward`. That has two costs:

1. **Compute/latency** — the 4 s window recomputes the whole window each hop; latency is the
   hop (~1 s) not the model's true streaming latency (~1 frame ≈ 40 ms).
2. **The growing-buffer decode degradation that's been fighting the matcher.** `--mode stream`
   re-decodes an ever-growing buffer; long/continuous recitations decode worse as the buffer
   grows (global RMS-normalization over mixed-loudness audio, out-of-distribution multi-ayah
   length), which is exactly why short back-to-back ayat (Al-Buruj 85:12–16) fall out and we
   had to bolt on refocus/rank-persistence heuristics.

True streaming replaces window re-decode with **`Emformer.infer` chunk-by-chunk**: feed only
the *new* ~160 ms of audio each step, carry the encoder state, emit incremental phonemes.
That gives:

- **~4× less compute** than the 4 s window (only new audio; no re-processing) and true
  ~40 ms latency — the battery/wearable path.
- A **genuinely incremental phoneme stream** with *bounded* context (Emformer left-context
  32 frames ≈ 1.3 s + memory 4 segments), independent of how long the recitation runs. No
  growing buffer to re-decode, no global-normalization drift. This is the clean foundation a
  unified matcher needs (see "What this does / doesn't fix").

No retraining: `infer` uses the same weights and is the streaming form of the segmented
`forward` the model was trained with. Accuracy carries over (must be verified by parity).

## What it is (technical)

Per step the runtime feeds one **segment**: `segment_length` (4) encoder frames + 
`right_context_length` (1) lookahead = **5 encoder frames**, and gets back 4 output frames +
updated state. Confirmed API:

```
Emformer.infer(input[B, seg+rc, D], lengths[B], states) -> (output[B, seg, D], out_lengths, states)
states : List[List[Tensor]]   # per layer; carried step to step
```

4 encoder frames @ 25 fps = **160 ms** of audio = 16 log-mel frames @ 100 fps.

### The one genuinely new piece: streaming `Conv2dSubsampling`

`forward` runs the two stride-2, kernel-3 convs over the whole clip. Streaming must produce
exactly the 5 encoder frames each step needs from a 160 ms audio chunk **while respecting the
conv receptive field at chunk boundaries** — otherwise the boundary frames differ from the
batch result and parity breaks.

- 2-layer conv (k3s2 ×2): receptive field **7 input frames**, stride **4**. To emit N new
  subsampled frames, advance 4·N input frames but need the previous **RF − stride = 3** input
  frames of overlap.
- Two implementation options:
  - **(a) cache raw log-mel frames** — keep a rolling tail of input frames, recompute the conv
    over `[cache ++ new_chunk]`, output only the new frames. Simplest; ~3–6 frames of cache.
  - **(b) cache per-layer conv activations** — more efficient, more bookkeeping.
- Start with (a) for correctness; it's cheap (the conv is ~0.2 MB / tiny FLOPs).

## Work breakdown

| # | Component | What | New complexity | Risk |
|---|---|---|---|---|
| 1 | `StreamingEmformerCTC` (training/model.py) | wrapper: streaming conv-cache → `encoder.infer` → CTC head; takes/returns explicit state | conv cache; state plumbing | med |
| 2 | Streaming `Conv2dSubsampling` | option (a) cache; emit exactly the frames a segment needs | boundary arithmetic | **med-high** (off-by-frame → parity fail) |
| 3 | Python parity harness | streaming loop over a clip == `forward` on the same clip, within tol | — | low |
| 4 | ONNX export (`export_onnx.py --streaming`) | flatten `List[List[Tensor]]` state to named graph inputs/outputs; conv-cache tensors as inputs/outputs; fixed segment shape | state flattening; exporter quirks | **high** (see risks) |
| 5 | int8 | same weight-only dynamic MatMul-only path; re-verify argmax-lossless on the stateful graph | — | low-med |
| 6 | Runtime driver (Python ref) | feed 160 ms chunks, thread states + conv cache, CTC-collapse across chunk boundaries (dedup last-emitted vs first-of-next, drop blanks) | boundary token dedup | low-med |
| 7 | C++/SDK port | thread state tensors through the ORT session (inputs = outputs of last call); conformance golden | ORT I/O binding of many state tensors | med |
| 8 | Matcher integration | consume the incremental phoneme stream with a streaming matcher (replaces window re-decode) | new commit design | **high** (separate effort) |

Items 1–6 are the export itself; 7 is the on-device port; 8 is the matcher redesign that
actually cashes in the benefit.

## Parity & validation plan (the gate)

Streaming must equal the batch forward, or accuracy silently drifts. Gate at each layer:

1. **Conv parity** — streaming conv over chunks == `Conv2dSubsampling.forward` on the whole
   clip, max|Δ| ≈ 0 (it's deterministic; only boundary handling can differ).
2. **Encoder parity** — full streaming (`infer` loop) log-probs == `forward` log-probs on the
   held-out clips, within a small tol (Emformer `infer` and `forward` are designed to match;
   confirm empirically). **Argmax phoneme path must be identical** (that's what the matcher
   eats).
3. **ONNX parity** — ORT streaming session == PyTorch streaming, like the current export.
4. **End-to-end** — reproduce the regression fixtures (`demo/regression.py`) through the
   streaming path with equal or better sequences. Add streaming cases.

## What this does / doesn't fix (honest)

- **Does:** removes window re-decode → ~4× cheaper, ~40 ms latency; produces a clean
  incremental phoneme stream with bounded context, eliminating the growing-buffer
  degradation and global-normalization drift that forced the refocus/rank-persistence
  heuristics.
- **Does NOT, by itself:** give a unified any-length mode. The phoneme stream still feeds a
  Stage-2 matcher, and the sliding-vs-stream trade-off is ultimately a *matcher/commit*
  design question. Streaming acoustics is the **enabler**: with a reliable incremental
  phoneme stream we can go back to the clean incremental `PhonemeMatcher` beam (arbitrary
  length, restart/jump built in) + a commit policy, instead of re-decoding windows. Expect
  item #8 (matcher redesign) to be where the "one mode for any ayah length" actually lands —
  streaming export makes it *possible*, not automatic.
- **Accuracy:** unchanged model weights; parity-gated. No retraining.

## Risks / unknowns (to de-risk first)

1. **ONNX state export (highest).** `torch.onnx.export` must expose the nested
   `List[List[Tensor]]` state (and conv cache) as flat, named graph inputs/outputs with fixed
   shapes. The current export already fights Emformer's data-dependent masks (fixed-window
   workaround); the streaming graph is fixed-shape (good) but the **state I/O plumbing is
   new**. *De-risk: a 1-day spike exporting a 2-layer Emformer `infer` with explicit states to
   ONNX and round-tripping one chunk.* This decides feasibility.
2. **Conv boundary correctness** — off-by-one frame silently corrupts every segment. Gated by
   conv parity (#1 above).
3. **CTC collapse across chunk boundaries** — greedy dedup must persist "last emitted id"
   across chunks so a token split across a boundary isn't double-counted. Straightforward but
   easy to get wrong.
4. **int8 on a stateful graph** — the state tensors are activations (fp32, not quantized); only
   MatMul weights quantize. Should be fine; verify argmax-lossless.
5. **Marginal benefit vs cost** — the 4 s int8 window already runs at RTF 0.002 (comfortably
   real-time even on watch silicon), so the compute win is for battery/always-on, not
   feasibility. The *real* motivation is the clean phoneme stream for the matcher — worth being
   explicit that the payoff is coupled to item #8.

## Effort & phasing

- **Phase A — feasibility spike (≈1 day):** items 2–3 (streaming conv + Python parity) and the
  #1 risk ONNX state round-trip on a toy Emformer. Go/no-go on the export.
- **Phase B — export (≈2–3 days):** items 1, 4, 5, 6 — full `StreamingEmformerCTC`, ONNX with
  state I/O, int8, Python runtime + parity on real clips + regression fixtures.
- **Phase C — on-device (≈2–3 days):** item 7 — SDK/C++ port, conformance golden, ORT state
  binding.
- **Phase D — matcher (separate, larger):** item 8 — streaming matcher/commit that consumes the
  incremental stream; this is where the mode unification is designed and tuned.

Phases A–C are the "export"; D is the follow-on that realizes the benefit.

## Decision points for the user

1. **Motivation:** are we doing this primarily for (a) battery/latency/wearable, or (b) to fix
   the mode-split via a clean phoneme stream? If (b), phase D is the real work and should be
   planned alongside — the export alone won't change detection behavior.
2. **Scope now:** just the **feasibility spike (Phase A)** to retire the ONNX-state risk before
   committing, or the full A–C export?
3. **Target first:** desktop/Python streaming (fastest to validate) then port, or straight to
   the SDK path.

## Feasibility spike results (2026-07-04) — modeling ✅, direct ONNX export ✗

Ran the Phase-A spike on `best_mic.pt`:

1. **Streaming `Conv2dSubsampling` (cached) vs batch conv** — identical frame count, max|Δ|
   ~1e-6 across chunk sizes 7/16/32. The RF-7/stride-4 boundary cache is correct. ✅
2. **`Emformer.infer` loop (feed S+R=5 frames, advance S=4) vs `forward`** — max|Δ| ~1e-6,
   **100% argmax phoneme agreement** on 3 held-out clips. The encoder streams exactly. ✅
   State is fixed-shape and stable: 12 layers × 4 tensors = 48 (`[mem(4,1,256),
   lc_k(32,1,256), lc_v(32,1,256), counter(1,1,i32)]`).
3. **ONNX export of a single stateful step (48 state tensors as I/O)** — **BLOCKED**. The
   graph is fixed-shape, but `torchaudio.Emformer.infer` has **data-dependent control flow**
   (the int32 past-length counter is read via `.item()` to drive left-context slicing):
   - legacy tracer *exports* but ORT won't load it (`Reshape [ShapeInferenceError] Invalid
     position of 0` — a traced dynamic reshape);
   - dynamo (`torch.export`) fails on the unbacked symints from `.item()` (`Sym(u1) =
     aten.item(...)`), symbolic shapes like `9 - Min(4, 4 - Min(4, CeilToInt(u0/4)))`.
   Same root cause as the full-utterance data-dependent-mask problem, but now inside `infer`.

**Verdict:** the streaming *math* is proven and needs no retraining, but torchaudio's `infer`
is **not ONNX-exportable as-is**. Realizing streaming export requires a **static-shape
reimplementation of the Emformer streaming step** — replace the counter-driven left-context
slice with a fixed-size ring-buffer state (always 32 left-context frames), no `.item()`, no
data-dependent reshape. Same weights/math, ONNX-clean. This is the real Phase B and it's a
meaningful reimplementation (est. bumped: **~4–6 days**, medium risk), not a thin wrapper.

Options to weigh before committing Phase B:
- **(1) Static-shape reimpl of the streaming step** (recommended if we stay on Emformer):
  rewrite the per-layer streaming attention with ring-buffer caches; validate against the
  proven PyTorch `infer` parity above (which is now the golden).
- **(2) sherpa-onnx / icefall streaming encoder** — those ship export-clean streaming
  encoders (static caches) but mean swapping the encoder (retrain) — larger, re-opens a
  locked decision.
- **(3) Defer** — the 4 s int8 window already hits RTF 0.002; keep it, and pursue the
  mode-split fix at the *matcher* layer against the windowed decode for now.

Given the goal is the mode-split (not battery), option (3) + a matcher-side effort may reach
the goal sooner than the streaming reimpl; revisit streaming for the wearable/battery path.

## Static-shape streaming module — design sketch

Grounds the Phase-B reimplementation in the actual torchaudio internals
(`torchaudio/models/emformer.py`). The whole blocker is two small methods on
`_EmformerLayer`; everything else (attention math, FFN, layer norms, memory pooling) is
already static and reused unchanged.

### Where the data-dependence is

Per-layer state is 4 fixed-shape tensors (confirmed): `[memory(M,B,D), lc_key(L,B,D),
lc_val(L,B,D), past_length(1,B,int32)]` with `M=max_memory_size=4`, `L=left_context_length=32`,
`D=256`. Only two methods touch it:

- `_unpack_state` — reads `past_length.item()` and **trims leading zero-padding** off the
  left-context / memory buffers so warm-up chunks don't attend to it. This `.item()` + the
  data-dependent slice are the *entire* export blocker.
- `_pack_state` — appends the new K/V, keeps the last `L` (and last `M` memory), increments
  `past_length`. Its *output* shapes are already fixed (L, M); only the traced `shape[0]-L`
  arithmetic needs to be written ring-buffer style.

Steady state (`past_length >= L`) does no trimming — the buffers are simply full. So the trim
matters only for the first `L/S = 32/4 = 8` chunks (~1.3 s of each session).

### The fix: fixed buffers + a computed padding mask

Replace "trim the zeros" with "keep the full buffers, and **mask** the not-yet-filled slots"
— a mask built from `past_length` by tensor comparison (no `.item()`, ONNX-clean):

```python
# lengths as tensors, elementwise; arange(L)/arange(M) are constants folded into the graph
real_lc  = past_length.clamp(max=L)                       # [1,B] int
lc_pad   = torch.arange(L)  <  (L - real_lc)               # [L]  True = still-padding
mem_len  = torch.ceil(past_length / S).clamp(max=M)
mem_pad  = torch.arange(M)  <  (M - mem_len)               # [M]
# fold lc_pad / mem_pad (as additive -inf) into the attention key-padding mask that the
# layer already builds (_gen_padding_mask), so padded slots get zero attention weight.
```

`StaticEmformerLayer` then:
1. `_unpack_state` -> returns the **full** `state[0]`, `state[1]`, `state[2]` (no slicing) plus
   `lc_pad`/`mem_pad`.
2. attention runs on the full fixed-size K/V, with `lc_pad`/`mem_pad` added (−inf) to the
   existing key-padding mask — numerically identical to the trimmed version, but static shape.
3. `_pack_state` -> ring update: `new = torch.cat([buf, next])[-L:]` (fixed output L) or an
   `index_copy` into a rolling slot; `past_length += S`.

Everything downstream (`_EmformerAttention.infer`, FFN, norms, `memory_op` AvgPool) is unchanged
— same weights, same math.

### Two build strategies (correctness vs speed to first export)

- **(exact)** implement the mask above -> bit-parity with `forward`/`infer` including warm-up.
- **(warm-up-approx)** skip the mask, always attend to the full (zero-padded early) buffers.
  Only the first ~1.3 s of a session differs slightly; the matcher is error-tolerant, so this
  may be acceptable and is a faster first export. Decide by measuring the warm-up argmax delta
  against the parity golden. (Recommend building exact; keep approx as the fallback.)

### The module + fixed I/O

```
StreamingEmformerCTC.step(chunk[1, S+R, F],  *state) -> (log_probs[1, S, V], *new_state)
  conv_cache (RF-1 = 6 input frames)  ->  StreamingConv2dSubsampling  (spike-proven)
  -> 12 x StaticEmformerLayer(state[i])                                (this sketch)
  -> CTC head + log_softmax
state = conv_cache + 12*4 layer tensors = 49 fixed-shape graph inputs/outputs
```

Export: all shapes constant -> `torch.onnx.export` (legacy, opset 17) with the ~49 state
tensors as named I/O (dynamo not needed; there are no symbolic shapes once `.item()` is gone).
int8 as today (weight-only dynamic MatMul; state tensors stay fp32).

### Validation (the golden already exists)

The spike's PyTorch parity is the acceptance test: `StreamingEmformerCTC` looped over a clip
must reproduce `EmformerCTC.forward` — **argmax phonemes identical** (max|Δ| ~1e-6), then
ORT == PyTorch on the exported graph, then the `demo/regression.py` fixtures through the
streaming path. Test order: conv (done) -> one `StaticEmformerLayer` vs `_EmformerLayer.infer`
-> full stack -> ONNX.

### Phase-B progress

- **StaticEmformerLayer — DONE + validated** (`export/streaming_layer.py`, `python
  export/streaming_layer.py` -> PASS). The static-mask reimplementation of
  `_apply_attention_infer` is **bit-identical** to stock `_EmformerLayer.infer`: output diff
  ~1e-6 through warm-up (past_length 4→48) and **exactly 0.0** in steady state (past_length ≥ L).
  So the mask placement + ring-pack are correct and the `.item()` is gone. The single biggest
  correctness risk is retired.
- **StreamingEmformerCTC full-stack + ONNX export — DONE + validated** (`export/streaming_encoder.py`,
  `python export/streaming_encoder.py` -> both PASS). Streaming the whole pipeline (cached conv +
  12 `StaticEmformerLayer`s driven like `_EmformerImpl.infer` + CTC head) reproduces
  `EmformerCTC.forward` **100% argmax** (maxdiff ~1.5e-5). And the **stateful encoder step
  exports to ONNX and LOADS in onnxruntime** — the exact thing that was blocked — round-tripping
  vs PyTorch at log_probs 5.7e-6 / state 2.4e-6. Notes: use the **legacy exporter** (`dynamo=False`,
  opset 17); once `.item()` is gone the in-place mask assignments trace fine (no `cat`/`where`
  rewrite needed). On Windows set `PYTHONUTF8=1` — the dynamo path prints a ✅ that crashes cp1252.
  **The entire streaming-export feasibility risk is now retired.**
- **Packaged model + runtime driver — DONE + validated** (`export/streaming_runtime.py`,
  `python export/streaming_runtime.py`). Exports two graphs: `stream_conv.onnx` (Conv2dSubsampling,
  dynamic T, 1.4 MB — plain export, no Emformer) and `stream_encoder.onnx` (one fixed-shape
  stateful step, 38.3 MB) / `stream_encoder.int8.onnx` (10.1 MB, weight-only dynamic MatMul;
  state tensors stay fp32). `StreamingRuntime` feeds log-mel frames -> cached streaming conv ->
  encoder step per S-frame segment (threading the 48 state tensors) -> greedy CTC with
  blank/repeat collapse across chunk boundaries. Validated: the runtime produces **phoneme-
  identical** output to `EmformerCTC.forward` on held-out clips in **both fp32 AND int8**, RTF
  ~0.015 (incl. conv). Gotcha fixed: the runtime must start from the **zero** state
  (`layer._init_state`), not a dummy first step, or warm-up is corrupted.
- **SDK/C++ port of the driver — DONE + validated** (`sdk/core/src/streaming.{h,cpp}`,
  `sdk/core/tests/test_streaming.cpp`). `StreamingModel` = two ORT sessions (dynamic-T
  `stream_conv.onnx` + fixed stateful `stream_encoder.int8.onnx`) threading the 48 state
  tensors + the conv boundary cache + CTC collapse across chunks, a byte-faithful port of
  `StreamingRuntime`. `feed()` returns `{id, frame}` (frame = absolute 25 fps output frame ->
  time = frame·0.04) so the chain windows get phoneme times. **Parity: 5/5 EXACT phoneme match
  vs the Python runtime on the fp32 encoder** (incl. a 4486-frame / 204-phoneme clip — the
  logic gate: conv cache + state threading + collapse all correct). int8 differs on 1/5 by a
  single phoneme: a **cross-ORT quantization tie** at a borderline argmax (Python pip ORT vs
  the C++-linked ORT round the int8 MatMul differently), not a logic bug — and harmless for the
  error-tolerant matcher. So on-device the C++ int8 stream is self-consistent; validate the
  detector path against the C++ fp32/int8 decode, not the Python int8.
- **Remaining (detector integration + Android):** wire `StreamingModel` into `Detector::stepChain`
  (item 8) + bundle the two ONNX in the `.aar` + conformance golden. Design + the one subtlety below.

### Detector integration (item 8) — design + the front-end alignment invariant

`stepChain` today recomputes log-mel over the whole rolling buffer and RE-decodes it with the
fixed-window model every hop. Streaming replaces the *decode*: maintain a persistent, bounded
phoneme stream (`chainPh`/`chainTm`) that `StreamingModel::feed` extends with only the NEW
audio each hop; the scale-window + early-prefix slicing (already time-keyed) is unchanged.

**The one subtlety — the front-end is `center=True` / reflect-pad** (`frontend.logMel`), so
incremental feeding is exact only for **settled interior frames**:
- The **last ~2 frames** of the buffer are reflect-end-padded and CHANGE once more audio
  arrives -> feed only up to `T − guard` (guard≈2); the held-back frames settle next hop.
- The **buffer-start** reflect padding is wrong mid-stream, but those frames were already fed
  (as interior) in an earlier hop and are never re-fed (`fedFrames` monotonic).
- For buffer-frame index == absolute-frame index, keep the rolling buffer's start **hop-aligned**
  (erase the front cap in multiples of `hop`=160) when streaming is active. Then interior frames
  match the offline continuous log-mel EXACTLY, so the streamed `chainPh` == the whole-buffer
  greedy decode for settled frames — the integration acceptance test.

**Phase-2 posteriors:** the demo runs `chainSubMin=0.0` (soft substitution). Resolved by extending
`StreamingModel::Emit` with the per-emission top-k row (`feed(..., wantAlts=true)`, reusing
`decoder::topKAlts` for identical rounding) — the streaming path keeps full Phase-2 soft scoring, no
regression.

### Detector integration — DONE + validated (2026-07-10)

Wired into `Detector` (gated on `Config.streamConvPath`+`streamEncoderPath`; empty => windowed
re-decode, the default). `stepChain` now, when streaming is active: `streamFeed()` extends a
persistent `chainPh`/`chainTm`/`chainAlts` stream with only the newly-settled frames (feed up to
`T-guard`, guard=2; rolling-buffer front erased in whole hops to keep the start hop-aligned), then
the shared `chainMatch()` (early-prefix + all scale windows + vote + assemble) runs on that stream —
identical to the windowed path. Feeds continuously even on silence (keeps the acoustic state
gapless); only the *matching* is energy-gated. `reset()` clears the stream + model state.

**Acceptance test (the invariant above): PASS.** Same continuous audio through the C++ `Detector` in
`--chain` windowed vs `--chain <conv> <enc>` streaming (both `model_s123_mic_clean` weights,
`chainSubMin=0.0` Phase-2 soft), via `test_detector`:
- 54 s continuous 78:1–8 -> BOTH: `78:1 78:2 … 78:8`, **identical events AND timestamps**
  (12.0/18.0/25.5/27.0/39.0/43.5/48.0 s).
- 47 s continuous Al-Fātiḥa -> BOTH: `1:4 1:5 1:6` (identical).

So the streamed interior-frame decode is behavior-equivalent to the whole-buffer re-decode, and the
front-end alignment invariant holds. (Standalone `test_streaming` fp32 parity is 5/5; note ESET
NOD32 false-positive-deletes the freshly-linked `test_streaming.exe` as `Win64/Agent_AGen.MKX` —
add a `sdk\build\` AV exclusion to re-run it; `test_detector.exe` is unaffected and is the
end-to-end gate.)

### Android wiring — DONE (2026-07-10); on-device measurement pending

Kotlin -> JNI -> C++ plumbing complete: `Config.streaming` (default false, safe fallback to
windowed when the graphs are absent) -> `nativeCreate(..., streamConvPath, streamEncoderPath)` ->
`Detector` Config. `ModelManager` extracts `stream_conv.onnx` + `stream_encoder.int8.onnx` if
bundled. Demo `-PbundleStreaming` stages the two graphs into the APK (pair with `-PbundleModel`,
same checkpoint); unstaged otherwise so the **default download distribution stays windowed**. Demo
`Config(streaming = true)`. Verified: `:demo:assembleDebug -PbundleModel -PbundleStreaming` BUILD
SUCCESSFUL (streaming.cpp compiled arm64-v8a/armeabi-v7a/x86_64); the APK packages all three graphs.

### RTF win — MEASURED (2026-07-10): ~11x cheaper decode, identical detections

Instrumented decode-only wall-clock (`Detector::decodeStats`; `test_detector` prints RTF). On the
54 s continuous 78:1-8 stream (desktop, model_s123_mic_clean weights, Phase-2 soft):

| path | ms/hop | RTF (decode/audio) |
|---|---|---|
| windowed (22 s re-decode) | ~730 | 0.484 |
| **streaming** | **~65** | **0.043** |

**~11x cheaper per hop, byte-identical detections** (78:1-8 + Al-Fatiha unchanged). Key fix:
`streamFeed` first recomputed log-mel over the WHOLE 22 s buffer each hop, so the naive-DFT
front-end dominated both paths and streaming was only ~13% faster — computing log-mel over just the
NEW suffix (3-frame reflect margin; interior frames identical) unmasks the win (the encoder already
only sees new audio via the conv cache). The reduction is algorithmic (O(new audio) vs O(22 s
window); Emformer attention is O(U^2) in length), so it carries to ARM on-device. Verified live on
the phone (surah 111): `(stream)` hops, incremental phoneme stream, correct tracking; optimized
build re-deployed.

**Conformance golden — DONE (2026-07-10).** `golden/streaming/*.phonemes.txt` pins the Python
`StreamingRuntime` over each frontend log-mel fixture (fp32 graphs exported per-checkpoint into
`conformance/assets/`, gitignored); `test_streaming <conf> stream_conv.onnx stream_encoder.onnx`
reproduces it EXACTLY — **ALL PASS (6/6)**. Fifth port-risk stage, spec.md §Streaming model inference.

### Enabled by default + manifest delivery — DONE (2026-07-10)

`Config.streaming` now defaults **true**. The two graphs are delivered like the model: the manifest
gains optional `streamConv`/`streamEncoder` `{url, sha256}`, `ModelManager` downloads them into
`models/stream/` (sha256-verified, version-keyed, pruned on version change; a subdir so the model
prune/scan never touches them), and a download failure is non-fatal (windowed fallback). Manifest
fetched once and shared by `resolveModel` + `resolveStreaming`. `-PbundleStreaming` still wins for
offline dev. `:demo:modelManifest` emits the streaming keys (sha256 of `export/onnx/stream_conv.onnx`
+ `stream_encoder.int8.onnx`) when both exist. Builds: default `:demo:assembleDebug` compiles;
generated manifest JSON validated (streamConv/streamEncoder with correct sha256/URLs).

**Hosting DONE + download path validated on-device (2026-07-10).** `stream_conv.onnx` +
`stream_encoder.int8.onnx` + the regenerated `model_manifest.json` uploaded to the `model` release
(`gh release upload model ... --clobber`; `model.int8.onnx` unchanged, same sha, left in place). The
public-URL sha256 of all three matches the exported files. On-device: a fresh **download-build**
install (uninstall → `assembleDebug` → install, no bundle flags) fetched the model + both graphs from
the release into `…/files/quranrecite/models{,/stream}`, on-device sha256 **byte-exact** vs the hosted
files (conv `5d75caea…`, encoder `1a0c260c…`), detector init clean (no native error / windowed
fallback). One transient CDN read-timeout stalled the encoder mid-download → non-fatal windowed
fallback that session, and a relaunch re-downloaded it cleanly (the `.part` restart path works).
Default download builds now stream end-to-end.

### Phase D (matcher on the stream) — INVESTIGATED, not viable as a mode-split fix (2026-07-04)

Checked the crux before building: does the clean streaming stream contain each ayah? It does
**only for the first ayah**; continuations under-decode. Measured (normalized edit of the
stream vs each ayah): 78:40 solo CLEAN 0.31; 114:1 CLEAN 0.23 but 114:2/3 partial; 85:12 CLEAN
0.25 but **85:13–16 MISSING** (0.5–0.76); 98:1 partial, **98:2 MISSING** (0.69); 78:38 MISSING.
Stream lengths are ~half the true phoneme count (98: 122 vs 217).

**Root cause:** the model was trained on **single ayat**, so a *continuous* multi-ayah stream is
out-of-distribution and the acoustic decode under-produces the continuations — the streaming
decode == `forward` decode (validated), and both degrade after the first ayah. The windowed
`auto` mode works on continuous recitation *because* each 4 s window is a single-ayah fragment
(in-distribution) and decodes well; the whole-stream decode is *worse*. So a matcher on the
streaming stream (Phase D) would **regress** vs `auto`, not unify.

**Conclusion (2026-07-04):** streaming export's detection value is **single-ayah / push-to-talk**
(decode is in-distribution and clean) + **battery/latency/wearable** — not the continuous
mode-split. The real lever for robust *continuous* recitation is **training data**, not the
matcher and not the streaming export. `auto` remains the best continuous solution today.

### Phase D RE-CHECK (2026-07-10) — RESOLVED: continuous decode now works

Re-ran the crux with the CURRENT model (`best_s123_mic_clean`, trained on surahs 1-3 + Juz Amma
— i.e. WITH long/continuous-like ayat) + the segment-based **chain decoder** (not the old
whole-ayah matcher). On 30 test sequences of 4 consecutive ayat, decoding the CONCATENATED audio
in one forward pass (== streaming, parity re-confirmed 100% argmax on the current model):
- **continuous (forward == streaming) ayah recall 87.5%** vs **stitched per-ayah recall 85.0%**.

The continuous decode is now EQUAL-to-BETTER than in-distribution per-ayah decodes — the 2026-07-04
under-decode is GONE. Two causes: (1) the expanded corpus put long/continuous ayat in
distribution; (2) the chain decoder's waqf-segment units + context are robust to residual decode
gaps. **So streaming is now a valid drop-in for the continuous Mode::Chain**, not just
push-to-talk. And the compute case is stronger than in 2026-07: Mode::Chain re-decodes a **22 s**
window every hop (not the old 4 s), so streaming (decode only the new audio) is a bigger win and
directly cuts the on-device per-hop cost/latency the user felt. **Verdict: implement (C++ port).**

Streaming export re-produced + validated for `best_s123_mic_clean` (2026-07-10):
`stream_conv.onnx` 1.4 MB + `stream_encoder.int8.onnx` 10.1 MB; runtime == forward 100% phoneme
match (fp32 AND int8); streaming RTF 0.013.

### Design-specific risks

- **Mask placement** — `_gen_padding_mask` / `_EmformerAttention._forward` build the mask over
  a specific key order (`[mems, right_context, utterance]` + prepended left-context); the
  `lc_pad`/`mem_pad` must land on the right key positions. Main correctness risk; caught by the
  per-layer parity test.
- **Ring update in ONNX** — `cat([buf,next])[-L:]` must fold to a static shape; if the tracer
  balks, use `index_copy`/`slice-and-concat` with explicit constants.
- **`past_length` int32 in-graph** — only used now for the mask comparison (no `.item()`); keep
  it as a tensor I/O, incremented by the constant `S`.

## What it does NOT touch

Front-end (log-mel params), tokens, the matcher lexicon, and the model weights are all
unchanged. `export_onnx.py` full-utterance + 4 s window exports stay as-is (offline eval,
conformance golden, current SDK artifact); streaming is an additional artifact.
