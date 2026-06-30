#!/usr/bin/env python3
"""
Phase-2 augmentation: simulate the mobile-phone channel so the clean-studio model
generalizes to learners on real devices.

Two stages:
  - WaveformAugment (audiomentations): applied to the 16 kHz waveform before
    feature extraction. Models a plausible capture chain — source variation -> room
    -> phone mic band -> air -> gain/noise -> clipping -> codec.
  - SpecAugment (torch): time/freq masking on the log-mel, applied after extraction.

External-corpus transforms (background noise, impulse responses) are optional hooks:
pass noise_dir / ir_dir to enable them. Without them we still get meaningful phone
simulation from synthetic noise + EQ + codec.

Speed/pitch perturbation is kept MILD on purpose — madd (vowel elongation) is
phonemically meaningful in recitation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import audiomentations as A


def build_waveform_augment(
    sample_rate: int = 16000,
    noise_dir: str | None = None,
    ir_dir: str | None = None,
) -> A.Compose:
    """Physically-ordered phone-channel chain. Each transform fires probabilistically."""
    transforms: list = [
        # --- source variation (mild!) ---
        A.PitchShift(min_semitones=-1.0, max_semitones=1.0, p=0.25),
        A.TimeStretch(min_rate=0.95, max_rate=1.06, leave_length_unchanged=False, p=0.25),
    ]

    # --- room impulse response (optional) ---
    if ir_dir and Path(ir_dir).exists():
        transforms.append(A.ApplyImpulseResponse(ir_path=ir_dir, p=0.4))

    transforms += [
        # --- cheap-mic band + coloration (poor mics roll off lows/highs) ---
        A.BandPassFilter(min_center_freq=350.0, max_center_freq=3400.0, p=0.4),
        A.SevenBandParametricEQ(min_gain_db=-8.0, max_gain_db=8.0, p=0.4),
        A.AirAbsorption(p=0.15),
        A.Gain(min_gain_db=-10.0, max_gain_db=6.0, p=0.4),   # level normalized out later;
    ]                                                        # mainly interacts with clipping

    # --- ambient noise (dominant poor-/quiet-mic degradation after RMS normalization) ---
    if noise_dir and Path(noise_dir).exists():
        transforms.append(A.AddBackgroundNoise(sounds_path=noise_dir,
                                               min_snr_db=0.0, max_snr_db=22.0, p=0.6))
    transforms += [
        A.AddGaussianSNR(min_snr_db=3.0, max_snr_db=30.0, p=0.6),
        A.AddColorNoise(min_snr_db=5.0, max_snr_db=30.0, p=0.3),   # pink/brown, more realistic
        A.ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=12, p=0.25),
        # --- codec round-trip (last, like a real upload); low bitrates for cheap encoders ---
        A.Mp3Compression(min_bitrate=8, max_bitrate=64, p=0.4),
    ]
    return A.Compose(transforms, shuffle=False)


class SpecAugment:
    """Time/frequency masking on a log-mel tensor [T, n_mels]."""

    def __init__(self, freq_mask: int = 15, time_mask_frac: float = 0.10,
                 n_freq: int = 2, n_time: int = 2):
        self.freq_mask = freq_mask
        self.time_mask_frac = time_mask_frac
        self.n_freq = n_freq
        self.n_time = n_time

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        T, M = feats.shape
        fill = feats.mean()
        for _ in range(self.n_freq):
            f = int(torch.randint(0, self.freq_mask + 1, (1,)))
            if f and M - f > 0:
                f0 = int(torch.randint(0, M - f, (1,)))
                feats[:, f0:f0 + f] = fill
        max_t = max(1, int(self.time_mask_frac * T))
        for _ in range(self.n_time):
            t = int(torch.randint(0, max_t + 1, (1,)))
            if t and T - t > 0:
                t0 = int(torch.randint(0, T - t, (1,)))
                feats[t0:t0 + t, :] = fill
        return feats


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import soundfile as sf
    from data import wav_to_logmel

    wav, sr = sf.read("data/raw/audio/minshawy_mujawwad/s078_a001.mp3", dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = wav.astype(np.float32)

    aug = build_waveform_augment(16000)
    spec = SpecAugment()

    # apply several times to confirm stability + variety
    import contextlib, os
    base_feats = wav_to_logmel(torch.from_numpy(wav), sr)
    print(f"clean: wav={len(wav)} feats={tuple(base_feats.shape)} "
          f"mean={base_feats.mean():.2f}")

    for k in range(3):
        with contextlib.redirect_stderr(open(os.devnull, "w")):
            a = aug(samples=wav, sample_rate=16000)
            feats = wav_to_logmel(torch.from_numpy(np.ascontiguousarray(a)), 16000)
            feats = spec(feats.clone())
        assert torch.isfinite(feats).all(), "non-finite features after augment"
        print(f"aug {k}: wav={len(a)} feats={tuple(feats.shape)} "
              f"mean={feats.mean():.2f}")
    print("augment + specaugment OK, features finite")
