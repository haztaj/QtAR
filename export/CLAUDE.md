# export/ — on-device model export

Turns `training/exp/best_phase2.pt` into an ONNX artifact that runs under
onnxruntime (Android/iOS/desktop), with int8 quantization.

## `export_onnx.py` — full-utterance export

```bash
python export/export_onnx.py --checkpoint training/exp/best_phase2.pt
# -> export/onnx/{model.onnx (fp32), model.int8.onnx, tokens.txt}
```

Pipeline: load checkpoint → ONNX export → PyTorch/onnxruntime parity → int8 dynamic
quantization (weight-only, MatMul-only) → CPU RTF. Verified results (best_mic):
- parity max|Δ| ≈ 1.5e-5 (PyTorch vs ORT, on a clip other than the traced one)
- size 43.8 MB fp32 → **15.2 MB int8**, **100% argmax agreement** (the matcher consumes
  greedy argmax phonemes, so int8 is lossless for our purposes)
- CPU RTF ~0.03 (≈30× real-time desktop; comfortably real-time on phone)

### int8: weight-only dynamic, MatMul-only (not QDQ)

- **Weight-only dynamic**, not static QDQ. QDQ quantizes *activations*, and this Emformer's
  attention/LayerNorm activations carry outliers that static int8 ranges can't hold — it
  destroyed the phoneme argmax and detection (verified). Dynamic quant keeps activations
  fp32 → argmax-lossless.
- **`op_types_to_quantize=["MatMul"]`**: full dynamic quant also emits `ConvInteger` (from
  the small Conv2dSubsampling), which some ORT CPU builds (desktop 1.18) can't execute.
  `MatMulInteger` is supported on every EP and the Emformer MatMuls are ~all the weight mass,
  so the small conv stays fp32 (~0.2 MB) and int8 runs everywhere. Validated end-to-end
  through the C++ core (`sdk/.../test_detector` → 114:1→2→3) on the desktop ORT that
  previously rejected `ConvInteger`.

## Why fixed-window, not dynamic-T

Emformer's attention masks are built from the `lengths` **values** (data-dependent
control flow). Both exporters fail on a dynamic time axis:
- legacy TorchScript tracer bakes in shape arithmetic (`T - right_context`) that goes
  negative at other lengths → `ConstantOfShape` runtime error;
- dynamo (`torch.export`) refuses the data-dependent ops outright.

Fix: **fix the time dimension** (`--fixed-frames 3000` = 30 s). Then all shape ops are
constant and only the mask *content* varies with `lengths` — exports cleanly and
parity holds across lengths. The app pads/crops each clip to the window. Cost: short
clips pay 30 s of compute, but RTF is so low it doesn't matter. Needs `onnxscript`
(torch 2.x) and `dynamo=False` (legacy exporter; opset 17).

## 4 s windowed export — the SDK artifact (`--fixed-frames 416 --tag _4s`)

```bash
python export/export_onnx.py --checkpoint training/exp/best_mic.pt --fixed-frames 416 --tag _4s
# -> export/onnx/{model_4s.onnx, model_4s.int8.onnx}
```

The SDK runs the encoder over sliding **4 s windows**, so the 30 s full-utterance model wastes
~7× compute padding each window to 3000 frames. Exporting at **416 frames (~4 s)** right-sizes
it: since Emformer masks padded frames via `lengths`, the output for the valid frames is
**identical** to the 30 s model — same detections — but far cheaper. 416 (not 400) gives a hair
of headroom so a full 4 s window (~401 frames) is covered with no crop.

Measured (best_mic, int8): **RTF 0.002** (vs 0.030 for 30 s — ~15× less compute; Emformer
attention is O(U²) in length) and **11.0 MB** (vs 15.2 MB — the T-sized constants shrink).
`test_detector` reproduces `114:1→2→3` with identical confidences (0.76/0.72/0.82). This is the
model the demo dev-bundles and the SDK ships; the 30 s model stays for offline full-utterance
eval / the conformance golden.

## TODO — true streaming export (`Emformer.infer`)

The 4 s window still **recomputes the whole window each hop**. True low-latency streaming needs
`Emformer.infer(chunk, states)` chunk-by-chunk (segment_length=4 encoder frames ≈ 160 ms),
processing only the *new* audio each step (~4× less than the 4 s window). It's fixed-shape and
thus ONNX-friendly, but requires:
- caching the `Conv2dSubsampling` boundary frames across chunks (stateful conv), and
- threading the Emformer state list in/out of the ONNX graph as inputs/outputs.
The encoder is unchanged, so accuracy carries over. Lower priority now that the 4 s window
already hits RTF 0.002 (comfortably real-time even on watch silicon); this is the battery/latency
optimization for always-on / wearable use.

## On-device notes

- int8 is the size win (3×); on x86 it isn't faster than fp32+MKL (RTF parity), but on
  ARM phones int8 matmul typically is. Keep both artifacts.
- `tokens.txt` (phoneme↔id) ships with the model; the on-device matcher
  (`matcher/phoneme_matcher.py` logic) + `data/lang/ayah_phonemes.json` complete Stage 2.
- The on-device feature front-end must match `training/data.py`: 16 kHz, 80-dim
  log-mel, n_fft 400 / hop 160, fmin 20 / fmax 8000.
