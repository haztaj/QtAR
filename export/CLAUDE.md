# export/ — on-device model export

Turns `training/exp/best_phase2.pt` into an ONNX artifact that runs under
onnxruntime (Android/iOS/desktop), with int8 quantization.

## `export_onnx.py` — full-utterance export

```bash
python export/export_onnx.py --checkpoint training/exp/best_phase2.pt
# -> export/onnx/{model.onnx (fp32), model.int8.onnx, tokens.txt}
```

Pipeline: load checkpoint → ONNX export → PyTorch/onnxruntime parity → int8 dynamic
quantization → CPU RTF. Verified results (best_phase2):
- parity max|Δ| ≈ 1.7e-5 (PyTorch vs ORT, on a clip other than the traced one)
- size 43.8 MB fp32 → **15.1 MB int8**, **100% argmax agreement** (the matcher consumes
  greedy argmax phonemes, so int8 is lossless for our purposes)
- CPU RTF ~0.03 (≈30× real-time desktop; comfortably real-time on phone)

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

## TODO — streaming export (the real-time path)

Full-utterance is the offline / push-to-talk artifact. True low-latency streaming
needs `Emformer.infer(chunk, states)` chunk-by-chunk, which is **fixed-shape and thus
more ONNX-friendly** than the dynamic full-utterance path — but also requires:
- caching the `Conv2dSubsampling` boundary frames across chunks (stateful conv), and
- threading the Emformer state list in/out of the ONNX graph as inputs/outputs.
This is the natural next deliverable; the encoder is unchanged, so accuracy carries over.
Alternative interim: re-run the fixed-window model on a growing buffer (simpler, wastes
compute, but RTF headroom allows it).

## On-device notes

- int8 is the size win (3×); on x86 it isn't faster than fp32+MKL (RTF parity), but on
  ARM phones int8 matmul typically is. Keep both artifacts.
- `tokens.txt` (phoneme↔id) ships with the model; the on-device matcher
  (`matcher/phoneme_matcher.py` logic) + `data/lang/ayah_phonemes.json` complete Stage 2.
- The on-device feature front-end must match `training/data.py`: 16 kHz, 80-dim
  log-mel, n_fft 400 / hop 160, fmin 20 / fmax 8000.
