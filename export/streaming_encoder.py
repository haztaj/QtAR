#!/usr/bin/env python3
"""
StreamingEmformerCTC — full streaming stack (Phase-B step 2) + ONNX export of the encoder step.

Assembles the pieces the spike/step-1 proved: a cached streaming Conv2dSubsampling, 12
`StaticEmformerLayer`s driven exactly like `_EmformerImpl.infer` (mems/right-context threaded
between layers), and the CTC head. Two checks:

  python export/streaming_encoder.py            # full-stack parity + ONNX encoder-step round-trip

1. **Full-stack parity** — streaming the whole pipeline chunk-by-chunk over a clip reproduces
   `EmformerCTC.forward` on that clip, argmax phonemes identical.
2. **ONNX encoder step** — the stateful per-chunk encoder graph (chunk + 48 state tensors ->
   log-probs + 48 states) exports and round-trips vs PyTorch. This is the graph that was blocked
   before StaticEmformerLayer removed the `.item()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "export"))
from streaming_layer import StaticEmformerLayer   # noqa: E402


def stream_conv(sub, feats, chunk=16):
    """Cached streaming Conv2dSubsampling; == batch conv (spike-proven). feats [1,T,F] -> [1,T',D]."""
    def conv_fwd(buf):
        x = buf.unsqueeze(1); x = sub.conv(x); b, c, t, f = x.shape
        return sub.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
    rf, stride = 7, 4
    T = feats.shape[1]
    cache = feats[:, :0]; emitted = 0; cache_start = 0; outs = []; pos = 0
    while pos < T:
        new = feats[:, pos:pos + chunk]; pos += new.shape[1]
        buf = torch.cat([cache, new], 1)
        if buf.shape[1] >= rf:
            o = conv_fwd(buf); first = cache_start // stride
            newo = o[:, emitted - first:]; outs.append(newo); emitted += newo.shape[1]
            keep = max(0, stride * emitted - cache_start); cache = buf[:, keep:]; cache_start += keep
        else:
            cache = buf
    return torch.cat(outs, 1) if outs else feats[:, :0]


class StreamingEncoder(nn.Module):
    """Static, ONNX-exportable form of `_EmformerImpl.infer` — 12 StaticEmformerLayers."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.layers = nn.ModuleList(StaticEmformerLayer(l) for l in enc.emformer_layers)
        self.S = enc.segment_length
        self.R = enc.right_context_length

    def infer(self, chunk, lengths, states):
        """chunk [B, S+R, D], states: list of 12 layer-states (or None). -> (out[B,S,D], lens, states)."""
        enc = self.enc
        x = chunk.permute(1, 0, 2)                          # [S+R, B, D]
        rc_start = x.size(0) - self.R
        right_context, utterance = x[rc_start:], x[:rc_start]
        out_lengths = torch.clamp(lengths - self.R, min=0)
        if enc.use_mem:
            mems = enc.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
        else:
            mems = torch.empty(0).to(dtype=x.dtype, device=x.device)
        output = utterance
        out_states = []
        for i, layer in enumerate(self.layers):
            output, right_context, st, mems = layer.infer(
                output, out_lengths, right_context, None if states is None else states[i], mems)
            out_states.append(st)
        return output.permute(1, 0, 2), out_lengths, out_states


def full_stream(model, features, seg_advance=None):
    """Stream the whole pipeline over a clip: conv -> encoder-infer loop -> CTC head. -> log_probs."""
    enc = StreamingEncoder(model.encoder)
    S, R = enc.S, enc.R
    with torch.no_grad():
        x = stream_conv(model.subsampling, features)        # [1, T', D]
        Tp = x.shape[1]
        outs, states, i = [], None, 0
        while i + S <= Tp:
            chunk = x[:, i:i + S + R]
            if chunk.shape[1] < S + R:                       # pad right-context at the tail
                chunk = torch.cat([chunk, torch.zeros(1, S + R - chunk.shape[1], x.shape[2])], 1)
            o, _, states = enc.infer(chunk, torch.tensor([chunk.shape[1]]), states)
            outs.append(o); i += S
        enc_out = torch.cat(outs, 1)
        return model.ctc_head(enc_out).log_softmax(-1)


class EncoderStep(nn.Module):
    """One fixed-shape encoder step for ONNX: (chunk, *flat_states) -> (log_probs, *flat_states)."""

    def __init__(self, model):
        super().__init__()
        self.enc = StreamingEncoder(model.encoder)
        self.head = model.ctc_head
        self.n = len(self.enc.layers)

    def forward(self, chunk, *flat):
        states = [list(flat[i * 4:(i + 1) * 4]) for i in range(self.n)]
        out, _, new_states = self.enc.infer(chunk, torch.tensor([chunk.shape[1]]), states)
        lp = self.head(out).log_softmax(-1)
        return (lp, *[t for st in new_states for t in st])


def _test(checkpoint="training/exp/best_mic.pt"):
    from model import EmformerCTC
    from data import AyahDataset
    ck = torch.load(REPO / checkpoint, map_location="cpu")
    model = EmformerCTC(num_tokens=ck["vocab"]); model.load_state_dict(ck["model"]); model.eval()
    ds = AyahDataset("test")

    # ---- 1) full-stack parity: streaming pipeline vs forward, argmax phonemes ----
    print("full-stack parity (streaming vs forward):")
    ok_all = True
    for idx in [0, 3, 7]:
        feats = ds[idx]["features"].unsqueeze(0)
        with torch.no_grad():
            ref, ref_len = model(feats, torch.tensor([feats.shape[1]]))
        strm = full_stream(model, feats)
        n = min(strm.shape[1], ref.shape[1])
        d = (strm[:, :n] - ref[:, :n]).abs().max().item()
        agree = (ref[:, :n].argmax(-1) == strm[:, :n].argmax(-1)).float().mean().item()
        ok_all &= agree == 1.0
        print(f"  clip {idx}: frames={n}  maxdiff={d:.2e}  argmax_agree={agree:.1%}")

    # ---- 2) ONNX encoder-step export + round-trip ----
    print("\nONNX encoder step:")
    enc = model.encoder
    L = enc.emformer_layers[0]
    S, R, D, nL = enc.segment_length, enc.right_context_length, L.input_dim, len(enc.emformer_layers)
    with torch.no_grad():
        _, _, st0 = StreamingEncoder(enc).infer(torch.randn(1, S + R, D), torch.tensor([S + R]), None)
    init = [t for layer in st0 for t in layer]
    step = EncoderStep(model).eval()
    chunk = torch.randn(1, S + R, D)
    with torch.no_grad():
        pt = step(chunk, *init)
    onnx_path = REPO / "export" / "onnx" / "_stream_step.onnx"
    names_in = ["chunk"] + [f"s{i}" for i in range(4 * nL)]
    names_out = ["log_probs"] + [f"ns{i}" for i in range(4 * nL)]
    try:
        torch.onnx.export(step, (chunk, *init), str(onnx_path), input_names=names_in,
                          output_names=names_out, opset_version=17, do_constant_folding=True,
                          dynamo=False)
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        feed = {"chunk": chunk.numpy()}
        for i, t in enumerate(init):
            feed[f"s{i}"] = t.numpy()
        outs = sess.run(None, feed)
        d_lp = float(np.abs(outs[0] - pt[0].detach().numpy()).max())
        d_st = max(float(np.abs(outs[1 + i] - pt[1 + i].detach().numpy()).max()) for i in range(4 * nL))
        print(f"  exported + loaded OK  ({onnx_path.stat().st_size/1e6:.1f} MB)")
        print(f"  round-trip: log_probs maxdiff={d_lp:.2e}  state maxdiff={d_st:.2e}")
        onnx_ok = d_lp < 1e-3 and d_st < 1e-3
    except Exception as e:
        print(f"  ONNX FAILED: {str(e)[:400]}")
        onnx_ok = False
    finally:
        onnx_path.unlink(missing_ok=True)

    print(f"\nRESULT: full-stack {'PASS' if ok_all else 'FAIL'}  |  ONNX {'PASS' if onnx_ok else 'FAIL'}")
    return ok_all and onnx_ok


if __name__ == "__main__":
    raise SystemExit(0 if _test() else 1)
