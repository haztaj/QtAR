#!/usr/bin/env python3
"""
Streaming Emformer + CTC phoneme model (plain PyTorch).

  log-mel [B, T, 80]
    -> Conv2dSubsampling (4x time reduction)   [B, T//4, d_model]
    -> Emformer (streaming transformer)         [B, T//4, d_model]
    -> Linear CTC head + log_softmax            [B, T//4, vocab]

Train with torch.nn.CTCLoss (blank = id 0).  Streaming inference uses
Emformer.infer(...) with carried states (wired up at export time).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchaudio.models import Emformer


def _sub_len(lengths: torch.Tensor) -> torch.Tensor:
    """Output length after two stride-2, kernel-3, no-pad conv layers."""
    for _ in range(2):
        lengths = torch.div(lengths - 3, 2, rounding_mode="floor") + 1
    return lengths.clamp(min=1)


class Conv2dSubsampling(nn.Module):
    """4x time downsampling, ESPnet/icefall style. [B,T,F] -> [B,T',d_model]."""

    def __init__(self, in_freq: int, d_model: int, channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        # frequency dim after two stride-2 convs
        freq_out = ((in_freq - 1) // 2 - 1) // 2
        self.out = nn.Linear(channels * freq_out, d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = x.unsqueeze(1)                       # [B, 1, T, F]
        x = self.conv(x)                         # [B, C, T', F']
        b, c, t, f = x.shape
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        x = self.out(x)                          # [B, T', d_model]
        return x, _sub_len(lengths)


class EmformerCTC(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        in_freq: int = 80,
        d_model: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        num_layers: int = 12,
        segment_length: int = 4,
        left_context_length: int = 32,
        right_context_length: int = 1,
        max_memory_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.subsampling = Conv2dSubsampling(in_freq, d_model)
        self.encoder = Emformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            segment_length=segment_length,
            dropout=dropout,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=max_memory_size,
        )
        self.ctc_head = nn.Linear(d_model, num_tokens)
        self.segment_length = segment_length
        self.right_context_length = right_context_length

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        """features [B,T,F], lengths [B] -> (log_probs [B,T',V], out_lengths [B])."""
        x, lengths = self.subsampling(features, lengths)
        x, lengths = self.encoder(x, lengths)
        # Emformer consumes the last `right_context_length` frames as lookahead,
        # so its output tensor is shorter than the lengths it passes through.
        lengths = torch.clamp(lengths, max=x.size(1))
        log_probs = self.ctc_head(x).log_softmax(dim=-1)
        return log_probs, lengths

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from torch.utils.data import DataLoader
    from data import AyahDataset, collate, load_tokens

    vocab = len(load_tokens())
    model = EmformerCTC(num_tokens=vocab)
    print(f"vocab={vocab}  params={model.num_params()/1e6:.1f}M")

    ds = AyahDataset("val")
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate, num_workers=0)
    batch = next(iter(dl))

    model.eval()
    with torch.no_grad():
        log_probs, out_lens = model(batch["features"], batch["feature_lengths"])

    print("\nforward:")
    print("  in  features    ", tuple(batch["features"].shape))
    print("  in  feat_lengths", batch["feature_lengths"].tolist())
    print("  out log_probs   ", tuple(log_probs.shape))
    print("  out out_lengths ", out_lens.tolist())

    tgt_lens = batch["target_lengths"]
    feasible = bool((out_lens >= tgt_lens).all())
    print(f"  target_lengths  {tgt_lens.tolist()}")
    print(f"  CTC feasible after 4x subsample: {feasible}")

    # Compute a CTC loss to confirm the whole path is differentiable.
    model.train()
    log_probs, out_lens = model(batch["features"], batch["feature_lengths"])
    loss = nn.CTCLoss(blank=0, zero_infinity=True)(
        log_probs.transpose(0, 1), batch["targets"], out_lens, tgt_lens
    )
    loss.backward()
    print(f"\n  CTC loss (random init): {loss.item():.3f}")
    print("  backward OK")
