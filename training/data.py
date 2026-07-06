#!/usr/bin/env python3
"""
Training data layer for the streaming Zipformer + CTC phoneme model.

Reads:
  data/raw/audio/manifest.csv        (path, reciter_id, surah_id, ayah_id, ...)
  data/lang/ayah_phonemes.json       "surah:ayah" -> "ph ph ph ..."
  data/lang/tokens.txt               phoneme -> id  (<blk> = 0)

Audio is decoded with soundfile (libsndfile handles MP3 here — torchaudio's
MP3 path needs torchcodec/FFmpeg which isn't wired up), resampled to 16 kHz,
and turned into 80-dim log-mel features with torchaudio.transforms (pure torch).

Reciter split is recomputed deterministically to match data/build_manifests.py
(sorted reciters: last 3 = test, previous 3 = val, rest = train).
"""

from __future__ import annotations

import contextlib
import json
import os
from functools import lru_cache
from pathlib import Path

import ffmpeg_fix  # noqa: F401  (registers choco FFmpeg DLLs before torchaudio import)
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parent.parent
MANIFEST_CSV = REPO / "data" / "raw" / "audio" / "manifest.csv"
AYAH_PHONEMES = REPO / "data" / "lang" / "ayah_phonemes.json"
TOKENS_TXT = REPO / "data" / "lang" / "tokens.txt"

# --- Feature config (must match export-time on-device front end) ---
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400          # 25 ms
HOP = 160            # 10 ms
WIN = 400
FMIN = 20.0
FMAX = 8000.0
LOG_FLOOR = 1e-10
NORM_RMS = 0.1       # gain-normalize every clip to this RMS (level-invariance for poor/quiet mics)

# Split sizes — keep in lockstep with data/build_manifests.py
VAL_RECITERS = 3
TEST_RECITERS = 3


# ---------------------------------------------------------------------------
# Vocab / labels
# ---------------------------------------------------------------------------

def load_tokens(path: Path = TOKENS_TXT) -> dict[str, int]:
    tok2id: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tok, idx = line.rsplit(" ", 1)
        tok2id[tok] = int(idx)
    assert tok2id.get("<blk>") == 0, "blank must be id 0 for CTC"
    return tok2id


def load_ayah_phonemes(path: Path = AYAH_PHONEMES) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v.split() for k, v in raw.items()}


def reciter_split(reciters: list[str]) -> dict[str, str]:
    """reciter_id -> 'train'|'val'|'test', matching build_manifests.py."""
    r = sorted(reciters)
    test = set(r[-TEST_RECITERS:])
    val = set(r[-(TEST_RECITERS + VAL_RECITERS):-TEST_RECITERS])
    return {x: ("test" if x in test else "val" if x in val else "train") for x in r}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _mel(sr: int) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        f_min=FMIN, f_max=FMAX, n_mels=N_MELS, power=2.0,
    )


@lru_cache(maxsize=None)
def _resampler(orig_sr: int) -> torchaudio.transforms.Resample:
    return torchaudio.transforms.Resample(orig_sr, SAMPLE_RATE)


def normalize_rms(wav: torch.Tensor, target: float = NORM_RMS) -> torch.Tensor:
    """Scale to a target RMS so level doesn't vary (studio clips span 0.01-0.2; a quiet
    mic is ~0.02). Applied consistently in train / eval / demo so the model is level-
    invariant. Done AFTER waveform augmentation so the relative noise level (SNR) — not
    absolute gain — is what the model must handle."""
    rms = wav.pow(2).mean().sqrt()
    if rms > 1e-6:
        wav = wav * (target / rms)
    return wav.clamp(-1.0, 1.0)


def logmel_16k(wav: torch.Tensor) -> torch.Tensor:
    """16 kHz mono waveform -> log-mel [T, N_MELS] (no resampling). Gain-normalized."""
    wav = normalize_rms(wav)
    mel = _mel(SAMPLE_RATE)(wav)                  # [N_MELS, T]
    logmel = torch.log(torch.clamp(mel, min=LOG_FLOOR))
    return logmel.transpose(0, 1).contiguous()    # [T, N_MELS]


def wav_to_logmel(wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
    """wav -> log-mel [T, N_MELS], resampling to 16 kHz if needed."""
    if orig_sr != SAMPLE_RATE:
        wav = _resampler(orig_sr)(wav)
    return logmel_16k(wav)


@contextlib.contextmanager
def _suppress_c_stderr():
    """Silence libmpg123/libsndfile C-level warnings on fd 2 (Python errors still raise)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def load_wav_16k(path: str) -> torch.Tensor:
    """Decode (soundfile) -> mono -> 16 kHz. [num_samples] float32. Silence on failure."""
    try:
        with _suppress_c_stderr():
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:                         # undecodable file — insurance
        print(f"[data] decode failed, using silence: {path} ({e})", file=__import__("sys").stderr)
        return torch.zeros(SAMPLE_RATE // 2)
    if audio.ndim > 1:                            # downmix to mono
        audio = audio.mean(axis=1)
    wav = torch.from_numpy(np.ascontiguousarray(audio))
    if sr != SAMPLE_RATE:
        wav = _resampler(sr)(wav)
    return wav


def load_features(path: str) -> tuple[torch.Tensor, int]:
    """Convenience: 16 kHz log-mel features for a file (no augmentation)."""
    return logmel_16k(load_wav_16k(path)), SAMPLE_RATE


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AyahDataset(Dataset):
    def __init__(self, split: str | None, max_seconds: float = 30.0,
                 augment: bool = False, noise_dir: str | None = None,
                 ir_dir: str | None = None, manifest_csv=MANIFEST_CSV,
                 row_filter=None):
        """split in {train,val,test} applies the reciter split to the main manifest.
        Pass split=None with a custom manifest_csv (e.g. RetaSy) to use all its rows.
        row_filter: optional callable(df)->bool mask for extra filtering.
        """
        assert split in ("train", "val", "test", None)
        manifest_csv = Path(manifest_csv)
        tag = split or manifest_csv.parent.name   # log label
        df = pd.read_csv(manifest_csv)
        if split is not None:
            split_map = reciter_split(df["reciter_id"].unique().tolist())
            df = df[df["reciter_id"].map(split_map) == split].reset_index(drop=True)
        if row_filter is not None:
            df = df[row_filter(df)].reset_index(drop=True)

        # Drop pathological long outliers — Emformer attention is O(U^2) and these
        # few clips (often multi-ayah / very slow) dominate memory. ~0.4% of data.
        if "duration" in df.columns and max_seconds:
            before = len(df)
            df = df[df["duration"] <= max_seconds].reset_index(drop=True)
            if len(df) < before:
                print(f"[{tag}] dropped {before - len(df)} clip(s) > {max_seconds}s")

        # Exclude files that hard-fail to decode (see data/raw/audio/bad_files.txt).
        bad_path = manifest_csv.parent / "bad_files.txt"
        if bad_path.exists():
            bad = set(bad_path.read_text(encoding="utf-8").split())
            before = len(df)
            df = df[~df["path"].isin(bad)].reset_index(drop=True)
            if len(df) < before:
                print(f"[{tag}] excluded {before - len(df)} undecodable file(s)")

        self.tok2id = load_tokens()
        self.ayah_ph = load_ayah_phonemes()

        # Drop rows whose ayah has no phoneme transcript (shouldn't happen).
        keys = df.apply(lambda r: f"{r['surah_id']}:{r['ayah_id']}", axis=1)
        mask = keys.isin(self.ayah_ph.keys())
        if (~mask).any():
            print(f"[{tag}] dropping {(~mask).sum()} rows without phonemes")
        self.df = df[mask.values].reset_index(drop=True)
        self.keys = [f"{r.surah_id}:{r.ayah_id}" for r in self.df.itertuples()]

        # Augmentation (train only). Built lazily to stay import-light.
        self.augment = augment
        self._wave_aug = None
        self._spec_aug = None
        if augment:
            from augment import build_waveform_augment, SpecAugment
            self._wave_aug = build_waveform_augment(SAMPLE_RATE, noise_dir, ir_dir)
            self._spec_aug = SpecAugment()
            print(f"[{tag}] augmentation ON"
                  + (f" noise={noise_dir}" if noise_dir else "")
                  + (f" ir={ir_dir}" if ir_dir else ""))

    def __len__(self) -> int:
        return len(self.df)

    def frame_lengths(self) -> list[int]:
        """Estimated log-mel frames per clip (100 fps) — for bucket batching."""
        if "duration" in self.df.columns:
            return [max(1, int(d * 100)) for d in self.df["duration"]]
        return [1] * len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        wav = load_wav_16k(row["path"])           # 16 kHz mono
        if self.augment:
            try:                                   # never let augmentation crash training
                with _suppress_c_stderr():        # codec round-trip is chatty
                    a = self._wave_aug(samples=wav.numpy(), sample_rate=SAMPLE_RATE)
                wav = torch.from_numpy(np.ascontiguousarray(a))
            except Exception:
                pass  # e.g. clip too short for MP3 codec — fall back to clean wav
        feats = logmel_16k(wav)
        if self.augment:
            feats = self._spec_aug(feats)
        phonemes = self.ayah_ph[self.keys[i]]
        target = torch.tensor([self.tok2id[p] for p in phonemes], dtype=torch.long)
        return {
            "features": feats,                    # [T, N_MELS]
            "target": target,                     # [U]
            "surah_id": int(row["surah_id"]),
            "ayah_id": int(row["ayah_id"]),
        }


class LengthBucketBatchSampler:
    """Variable-size batches bounded by (batch_size * max_frames) <= frame_budget
    AND (batch_size * max_frames^2) <= quad_budget.

    Clips are sorted by length so each batch is near-uniform (minimal padding) and
    long clips land in small batches. The linear budget bounds padding waste; the
    QUADRATIC budget bounds Emformer's O(T^2) attention memory. Without it, long-clip
    batches (e.g. 4 x 60 s at frame_budget 24000) push peak GPU memory past the point
    where the Windows/WDDM driver starts paging VRAM to system RAM and throughput
    collapses ~6-25x (measured on the RTX 5080 16 GB: 4x60 s = 833 ms/clip, peak 9.0 GB
    vs 3x60 s = 134 ms/clip, peak 6.8 GB — the cliff sits near ~8 GB allocated).
    quad_budget=1.0e8 (mel frames^2) keeps every batch shape comfortably below it.
    Batch order is shuffled each epoch; sizes vary, so set drop_last per need.
    """

    def __init__(self, lengths: list[int], frame_budget: int, shuffle: bool = True, seed: int = 0,
                 quad_budget: float = 1.0e8):
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0
        for i in order:
            new_max = max(cur_max, lengths[i])
            if cur and ((len(cur) + 1) * new_max > frame_budget
                        or (len(cur) + 1) * new_max * new_max > quad_budget):
                self.batches.append(cur)
                cur, cur_max = [i], lengths[i]
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            self.batches.append(cur)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            import random
            random.Random(self.seed + self.epoch).shuffle(order)
        for b in order:
            yield self.batches[b]

    def __len__(self) -> int:
        return len(self.batches)


def collate(batch: list[dict]) -> dict:
    batch.sort(key=lambda b: b["features"].shape[0], reverse=True)
    feat_lens = torch.tensor([b["features"].shape[0] for b in batch], dtype=torch.long)
    tgt_lens = torch.tensor([b["target"].shape[0] for b in batch], dtype=torch.long)

    T = int(feat_lens.max())
    feats = torch.zeros(len(batch), T, N_MELS)
    for i, b in enumerate(batch):
        feats[i, : b["features"].shape[0]] = b["features"]

    targets = torch.cat([b["target"] for b in batch])      # flat (for CTCLoss)
    return {
        "features": feats,            # [B, T, N_MELS]
        "feature_lengths": feat_lens, # [B]
        "targets": targets,           # [sum(U)]
        "target_lengths": tgt_lens,   # [B]
        "ayah_ids": [(b["surah_id"], b["ayah_id"]) for b in batch],
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    for split in ("train", "val", "test"):
        ds = AyahDataset(split)
        print(f"{split}: {len(ds)} clips")

    ds = AyahDataset("val")
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate, num_workers=0)
    batch = next(iter(dl))
    print("\nOne val batch:")
    print("  features        ", tuple(batch["features"].shape))
    print("  feature_lengths ", batch["feature_lengths"].tolist())
    print("  targets (flat)  ", tuple(batch["targets"].shape))
    print("  target_lengths  ", batch["target_lengths"].tolist())
    print("  ayah_ids        ", batch["ayah_ids"])

    n_tokens = len(load_tokens())
    assert batch["targets"].max() < n_tokens and batch["targets"].min() >= 1
    # CTC feasibility: input length must be >= target length for every item
    feasible = (batch["feature_lengths"] >= batch["target_lengths"]).all()
    print(f"\n  vocab size (incl blank): {n_tokens}")
    print(f"  CTC length feasibility : {bool(feasible)}")
