#!/usr/bin/env python3
"""
Packaged streaming model + runtime driver (Phase-B step 3).

Exports the streaming acoustic model as two ONNX graphs and drives them incrementally:

  export/onnx/stream_conv.onnx           Conv2dSubsampling, dynamic T (no Emformer -> plain export)
  export/onnx/stream_encoder.onnx        one fixed-shape stateful Emformer step (chunk + 48 states
  export/onnx/stream_encoder.int8.onnx     -> log_probs + 48 states); int8 = weight-only MatMul

`StreamingRuntime` feeds 80-dim log-mel frames in, runs the cached streaming conv (via the conv
graph) to produce subsampled frames, and runs the encoder step per S-frame segment (S+R input,
threading the 48 state tensors), emitting greedy CTC phonemes incrementally (blank/repeat
collapse carried across chunk boundaries). No growing buffer, no re-decode.

  python export/streaming_runtime.py       # export + validate runtime == forward (argmax phonemes)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "export"))
from streaming_encoder import StreamingEncoder, EncoderStep   # noqa: E402

OUT = REPO / "export" / "onnx"
RF, STRIDE = 7, 4          # Conv2dSubsampling receptive field / time stride


class ConvOnly(torch.nn.Module):
    def __init__(self, sub):
        super().__init__(); self.sub = sub

    def forward(self, feats):                       # [1,T,80] -> [1,T',D]
        x = feats.unsqueeze(1); x = self.sub.conv(x); b, c, t, f = x.shape
        return self.sub.out(x.transpose(1, 2).contiguous().view(b, t, c * f))


def export_artifacts(model):
    OUT.mkdir(parents=True, exist_ok=True)
    enc = model.encoder
    S, R, D = enc.segment_length, enc.right_context_length, enc.emformer_layers[0].input_dim
    nL = len(enc.emformer_layers)

    # --- conv: dynamic T, plain export ---
    conv = ConvOnly(model.subsampling).eval()
    conv_path = OUT / "stream_conv.onnx"
    torch.onnx.export(conv, (torch.randn(1, 40, 80),), str(conv_path), input_names=["feats"],
                      output_names=["sub"], dynamic_axes={"feats": {1: "T"}, "sub": {1: "Tp"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)

    # --- encoder step: fixed shape, stateful ---
    # ZERO initial state (== states=None): what the runtime must start every session with.
    init = []
    for layer in enc.emformer_layers:
        init.extend(layer._init_state(1, torch.device("cpu")))
    step = EncoderStep(model).eval()
    enc_path = OUT / "stream_encoder.onnx"
    names_in = ["chunk"] + [f"s{i}" for i in range(4 * nL)]
    names_out = ["log_probs"] + [f"ns{i}" for i in range(4 * nL)]
    torch.onnx.export(step, (torch.randn(1, S + R, D), *init), str(enc_path), input_names=names_in,
                      output_names=names_out, opset_version=17, do_constant_folding=True, dynamo=False)

    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8_path = OUT / "stream_encoder.int8.onnx"
    quantize_dynamic(str(enc_path), str(int8_path), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"])
    return conv_path, enc_path, int8_path, dict(S=S, R=R, D=D, nL=nL, init=[t.numpy() for t in init])


class StreamingRuntime:
    """Drives stream_conv + stream_encoder over incoming log-mel frames -> phoneme ids."""

    def __init__(self, conv_onnx, encoder_onnx, meta):
        import onnxruntime as ort
        self.conv = ort.InferenceSession(str(conv_onnx), providers=["CPUExecutionProvider"])
        self.enc = ort.InferenceSession(str(encoder_onnx), providers=["CPUExecutionProvider"])
        self.S, self.R, self.D, self.nL = meta["S"], meta["R"], meta["D"], meta["nL"]
        self._init = meta["init"]
        self.reset()

    def reset(self):
        self._cache = np.zeros((1, 0, 80), np.float32)   # streaming-conv frame cache
        self._cache_start = 0
        self._emitted = 0                                # subsampled frames produced by conv
        self._sub = np.zeros((1, 0, self.D), np.float32) # subsampled buffer for the encoder
        self._seg = 0                                    # subsampled frames consumed by encoder
        self._states = [a.copy() for a in self._init]
        self._prev = -1                                  # last emitted id (CTC collapse across chunks)

    def _conv_stream(self, feats):                       # feats [T,80] -> new subsampled [n,D]
        buf = np.concatenate([self._cache, feats[None]], 1)
        if buf.shape[1] < RF:
            self._cache = buf; return np.zeros((1, 0, self.D), np.float32)
        o = self.conv.run(None, {"feats": buf})[0]       # [1, O, D]
        first = self._cache_start // STRIDE
        newo = o[:, self._emitted - first:]
        self._emitted += newo.shape[1]
        keep = max(0, STRIDE * self._emitted - self._cache_start)
        self._cache = buf[:, keep:]; self._cache_start += keep
        return newo

    def feed(self, feats):                               # feats [T,80] -> list[int] new phoneme ids
        self._sub = np.concatenate([self._sub, self._conv_stream(feats)], 1)
        out = []
        while self._seg + self.S + self.R <= self._sub.shape[1]:
            chunk = self._sub[:, self._seg:self._seg + self.S + self.R]
            feed = {"chunk": chunk}
            for i, a in enumerate(self._states):
                feed[f"s{i}"] = a
            res = self.enc.run(None, feed)
            lp, self._states = res[0], res[1:]
            out += self._collapse(lp[0])                 # [S, V]
            self._seg += self.S
        return out

    def _collapse(self, lp):
        ids = lp.argmax(-1).tolist()
        out = []
        for s in ids:
            if s != self._prev and s != 0:
                out.append(s)
            self._prev = s
        return out


def _test(checkpoint="training/exp/best_mic.pt"):
    from model import EmformerCTC
    from data import AyahDataset, logmel_16k
    import soundfile as sf
    ck = torch.load(REPO / checkpoint, map_location="cpu")
    model = EmformerCTC(num_tokens=ck["vocab"]); model.load_state_dict(ck["model"]); model.eval()

    conv_p, enc_p, int8_p, meta = export_artifacts(model)
    print(f"exported: stream_conv {conv_p.stat().st_size/1e6:.2f} MB | "
          f"stream_encoder fp32 {enc_p.stat().st_size/1e6:.1f} MB | int8 {int8_p.stat().st_size/1e6:.1f} MB")

    ds = AyahDataset("test")

    def greedy(lp, length):
        ids = lp[0, :length].argmax(-1).tolist(); out, prev = [], -1
        for s in ids:
            if s != prev and s != 0: out.append(s)
            prev = s
        return out

    for onnx_enc, tag in [(enc_p, "fp32"), (int8_p, "int8")]:
        rt = StreamingRuntime(conv_p, onnx_enc, meta)
        print(f"\nruntime vs forward (encoder {tag}):")
        for idx in [0, 3, 7]:
            feats = ds[idx]["features"].unsqueeze(0)
            with torch.no_grad():
                ref_lp, ref_len = model(feats, torch.tensor([feats.shape[1]]))
            ref = greedy(ref_lp.numpy(), int(ref_len[0]))
            rt.reset()
            stream = rt.feed(feats[0].numpy())            # whole clip in one feed (streams internally)
            # streaming emits up to the last full segment; compare on the overlap
            n = min(len(ref), len(stream))
            match = ref[:n] == stream[:n]
            print(f"  clip {idx}: forward {len(ref)} ph, stream {len(stream)} ph, "
                  f"first {n} {'MATCH' if match else 'DIFFER'}")
            if not match:
                print(f"    forward: {ref[:n]}\n    stream : {stream[:n]}")

    # RTF: encoder int8 step cost per 160 ms of audio
    import time
    rt = StreamingRuntime(conv_p, int8_p, meta)
    feats = ds[0]["features"].unsqueeze(0)[0].numpy()
    t0 = time.perf_counter(); rt.reset(); rt.feed(feats); dt = time.perf_counter() - t0
    audio_s = feats.shape[0] * 160 / 16000
    print(f"\nstreaming RTF (int8, incl. conv): {dt/audio_s:.4f}  ({audio_s:.1f}s audio in {dt*1e3:.0f} ms)")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if _test() else 1)
