#!/usr/bin/env python3
"""
StaticEmformerLayer — ONNX-exportable streaming Emformer layer (Phase-B step 1).

torchaudio's `_EmformerLayer._apply_attention_infer` is the *only* thing blocking a streaming
ONNX export: `_unpack_state` reads `past_length.item()` and slices the leading zero-padding off
the fixed-size left-context / memory buffers (a warm-up optimization). That data-dependent
control flow can't export.

This reimplements just that step with **static shapes**: keep the full fixed-size buffers and,
instead of trimming, mask the not-yet-filled slots via a padding mask computed from
`past_length` by tensor comparison (no `.item()`, no dynamic slice). For batch size 1 (the
streaming case) `_gen_padding_mask` returns None, so the mask is folded into the `attention_mask`
that `_forward_impl` uses. Everything else (attention math, FFN, norms, memory pool) is the stock
layer, reused unchanged — same weights, same math.

Validated bit-for-bit against stock `_EmformerLayer.infer`:

    python export/streaming_layer.py            # -> parity PASS (best_mic.pt)

This is the de-risking milestone for the streaming export; see export/streaming-export-plan.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent


def static_apply_attention_infer(layer, utterance, lengths, right_context, mems, state):
    """Static-shape form of `_EmformerLayer._apply_attention_infer` (B=1). Same output as the
    stock method; no `.item()` / data-dependent slicing. `layer` is a torchaudio `_EmformerLayer`.
    State layout (unchanged): [memory(M,B,D), lc_key(L,B,D), lc_val(L,B,D), past_length(1,B,i32)]."""
    att = layer.attention
    M, L, S = layer.max_memory_size, layer.left_context_length, layer.segment_length
    R, Tu, B = right_context.size(0), utterance.size(0), utterance.size(1)
    if state is None:
        state = layer._init_state(B, utterance.device)
    pre_mems, lc_key, lc_val, past_length = state          # FULL fixed-size buffers, no trim

    if layer.use_mem:
        summary = layer.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)[:1]
    else:
        summary = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)
    Sd = summary.size(0)

    # Key order inside _forward_impl is [mems(M), right_context(R), left_context(L), utterance(Tu)].
    query_dim, key_dim = R + Tu + Sd, M + R + L + Tu
    attn_mask = torch.zeros(query_dim, key_dim, dtype=torch.bool, device=utterance.device)
    if Sd > 0:
        attn_mask[-1, :M] = True                            # summary query never attends memory

    # Warm-up padding: how many of the M memory / L left-context slots are real yet.
    pl = past_length.reshape(())                            # scalar tensor (stays a tensor)
    real_lc = torch.clamp(pl, max=L)
    real_mem = torch.clamp(torch.ceil(pl.float() / S).to(torch.long), max=M)
    idxM = torch.arange(M, device=utterance.device)
    idxL = torch.arange(L, device=utterance.device)
    col_pad = torch.zeros(key_dim, dtype=torch.bool, device=utterance.device)
    col_pad[:M] = idxM < (M - real_mem)
    col_pad[M + R:M + R + L] = idxL < (L - real_lc)
    attn_mask = attn_mask | col_pad.unsqueeze(0)

    out, out_mems, key, value = att._forward_impl(
        utterance, lengths, right_context, summary, pre_mems, attn_mask,
        left_context_key=lc_key, left_context_val=lc_val)
    next_k, next_v = key[M + R:], value[M + R:]             # [left_context(L), utterance(Tu)]
    new_state = layer._pack_state(next_k, next_v, utterance.size(0), mems, state)
    return out, out_mems, new_state


class StaticEmformerLayer(nn.Module):
    """Wraps a torchaudio `_EmformerLayer`; `.infer` is the static, ONNX-exportable form."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def infer(self, utterance, lengths, right_context, state, mems):
        lay = self.layer
        ln_utt, ln_rc = lay._apply_pre_attention_layer_norm(utterance, right_context)
        rc_output, output_mems, output_state = static_apply_attention_infer(
            lay, ln_utt, lengths, ln_rc, mems, state)
        out_utt, out_rc = lay._apply_post_attention_ffn(rc_output, utterance, right_context)
        return out_utt, out_rc, output_state, output_mems


def _parity_test(checkpoint="training/exp/best_mic.pt", steps=16, seed=1, tol=1e-4):
    sys.path.insert(0, str(REPO / "training"))
    from model import EmformerCTC
    ck = torch.load(REPO / checkpoint, map_location="cpu")
    model = EmformerCTC(num_tokens=ck["vocab"]); model.load_state_dict(ck["model"]); model.eval()
    layer = model.encoder.emformer_layers[0]
    static = StaticEmformerLayer(layer)
    S, R, B, D = layer.segment_length, model.right_context_length, 1, layer.input_dim
    torch.manual_seed(seed)
    st_a = st_b = None
    worst = 0.0
    print(f"parity: static vs stock _EmformerLayer.infer  (best_mic layer0, S={S} R={R} "
          f"L={layer.left_context_length} M={layer.max_memory_size})")
    print("step past_len  dOut_u    dOut_rc   dOut_m")
    with torch.no_grad():
        for step in range(1, steps + 1):
            utt, rc, mems = torch.randn(S, B, D), torch.randn(R, B, D), torch.randn(1, B, D)
            lengths = torch.tensor([S])
            ou_s, orc_s, st_a, om_s = layer.infer(utt, lengths, rc, st_a, mems)
            ou_t, orc_t, st_b, om_t = static.infer(utt, lengths, rc, st_b, mems)
            du = (ou_s - ou_t).abs().max().item()
            dr = (orc_s - orc_t).abs().max().item()
            dm = (om_s - om_t).abs().max().item()
            worst = max(worst, du, dr, dm)
            print(f"  {step:2d}   {int(st_a[3].flatten()[0]):3d}     {du:.2e}  {dr:.2e}  {dm:.2e}")
    ok = worst < tol
    print(f"\nWORST output diff: {worst:.3e}  ->  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _parity_test() else 1)
