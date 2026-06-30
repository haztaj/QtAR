#!/usr/bin/env python3
"""
Export the trained EmformerCTC to ONNX (full-utterance), verify PyTorch parity,
quantize to int8, and measure CPU real-time factor.

This is the offline / push-to-talk export: feed a whole ayah clip -> phoneme
posteriors. Streaming export (chunk-by-chunk with Emformer.infer + conv-cache)
is a separate follow-up; see export/CLAUDE.md.

  python export/export_onnx.py --checkpoint training/exp/best_mic.pt

Outputs -> export/onnx/{model.onnx, model.int8.onnx, tokens.txt}
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
from data import AyahDataset, SAMPLE_RATE, load_tokens   # noqa: E402
from model import EmformerCTC                            # noqa: E402

OUT_DIR = REPO / "export" / "onnx"


class CTCExport(torch.nn.Module):
    """Wrap EmformerCTC for ONNX: (features, lengths) -> (log_probs, out_lengths)."""

    def __init__(self, model: EmformerCTC):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        return self.model(features, lengths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="training/exp/best_mic.pt")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamo", action="store_true", help="use torch.export-based exporter")
    ap.add_argument("--fixed-frames", type=int, default=3000, help="fixed input window (10ms frames; 3000=30s)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(REPO / args.checkpoint, map_location="cpu")
    model = EmformerCTC(num_tokens=ckpt["vocab"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    wrapper = CTCExport(model).eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch')}, vocab {ckpt['vocab']})")

    # Fixed input window: Emformer's mask arithmetic is data-dependent, so dynamic-T
    # export bakes in wrong shapes. A fixed T makes all shape ops constant; only the
    # length-mask CONTENT varies. The app pads/crops each clip to FIXED_T (30 s).
    FIXED_T = args.fixed_frames
    ds = AyahDataset("test")

    def padded(i: int):
        f = ds[i]["features"]
        t = f.shape[0]
        valid = min(t, FIXED_T)
        out = torch.zeros(1, FIXED_T, f.shape[1])
        out[0, :valid] = f[:valid]
        return out, torch.tensor([valid], dtype=torch.long)

    feats, lengths = padded(0)
    onnx_path = OUT_DIR / "model.onnx"
    print(f"exporting ONNX (opset {args.opset}, fixed T={FIXED_T}) ...")
    torch.onnx.export(
        wrapper, (feats, lengths), str(onnx_path),
        input_names=["features", "lengths"],
        output_names=["log_probs", "out_lengths"],
        dynamic_axes={"features": {0: "B"}, "lengths": {0: "B"},
                      "log_probs": {0: "B"}, "out_lengths": {0: "B"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=args.dynamo,
    )
    print(f"  wrote {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)")

    # --- Parity: PyTorch vs onnxruntime on a DIFFERENT-length clip ---
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    f2, l2 = padded(5)   # different clip / valid-length, same fixed window
    with torch.no_grad():
        pt_lp, pt_len = wrapper(f2, l2)
    ort_lp, ort_len = sess.run(None, {"features": f2.numpy(), "lengths": l2.numpy()})
    max_diff = np.abs(pt_lp.numpy() - ort_lp).max()
    print(f"parity (valid_len={int(l2[0])}): max|Δ|={max_diff:.2e}  "
          f"out_len pt={int(pt_len[0])} ort={int(ort_len[0])}  "
          f"{'OK' if max_diff < 1e-3 else 'MISMATCH'}")

    # --- int8 dynamic quantization ---
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8_path = OUT_DIR / "model.int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    print(f"  wrote {int8_path}  ({int8_path.stat().st_size/1e6:.1f} MB)")

    sess8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    o8_lp, _ = sess8.run(None, {"features": f2.numpy(), "lengths": l2.numpy()})
    # agreement on argmax phoneme path (what the matcher consumes)
    agree = (pt_lp.numpy().argmax(-1) == o8_lp.argmax(-1)).mean()
    print(f"int8 argmax agreement vs fp32: {agree:.1%}")

    # --- CPU real-time factor ---
    tokens = load_tokens()
    (OUT_DIR / "tokens.txt").write_text(
        (REPO / "data" / "lang" / "tokens.txt").read_text(encoding="utf-8"), encoding="utf-8")

    def rtf(session, n=30):
        durs, comps = [], []
        for i in range(n):
            f, l = padded(i)
            audio_s = int(l) * 160 / SAMPLE_RATE           # real (valid) audio seconds
            t0 = time.perf_counter()
            session.run(None, {"features": f.numpy(), "lengths": l.numpy()})
            comps.append(time.perf_counter() - t0)
            durs.append(audio_s)
        return sum(comps) / sum(durs)

    print(f"\nCPU RTF (lower=faster than real-time):")
    print(f"  fp32 : {rtf(sess):.3f}")
    print(f"  int8 : {rtf(sess8):.3f}")
    print("  (desktop CPU; a phone is slower — treat as a relative figure)")


if __name__ == "__main__":
    main()
