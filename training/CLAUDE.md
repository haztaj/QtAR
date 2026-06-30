# training/ — streaming Emformer + CTC

Plain-PyTorch trainer for the Stage-1 acoustic model. **Not icefall/k2** and **not
Zipformer** — both signed-off deviations (see root CLAUDE.md locked-decisions table).

## Files

- `data.py`     — `AyahDataset` + `collate`. soundfile decodes MP3 (libsndfile, no
                  FFmpeg), `torchaudio.transforms` does resample→16k + 80-dim log-mel.
                  Reciter split recomputed to match `data/build_manifests.py`.
- `model.py`    — `EmformerCTC`: Conv2dSubsampling (4x) → torchaudio `Emformer` →
                  Linear CTC head. ~9.8M params.
- `train.py`    — AMP + grad-accum loop, AdamW, warmup→cosine LR, greedy-decode PER,
                  best/last checkpoints to `training/exp/`.
- `ffmpeg_fix.py` — registers choco `ffmpeg-shared` DLLs; imported defensively by
                  `data.py`. No-op for the current soundfile path.

## Commands

```bash
python training/data.py                       # data-layer smoke test (shapes, CTC feasibility)
python training/model.py                      # model smoke test (params, forward, loss/backward)
python training/train.py --smoke --num-workers 0   # ~7 steps on GPU, verify AMP path
python training/train.py --epochs 60 --frame-budget 36000   # real run
```

## Memory: length-bucketed batching (important)

Emformer self-attention is **O(U²)** in sequence length. Clip durations are
heavy-tailed (median 5.4 s, max 79 s ≈ 1980 encoder frames), so fixed-size batches
OOM the moment a batch catches several long clips. Solution:

- `AyahDataset(max_seconds=30)` drops the ~0.4% pathological-length outliers
  (55 train / 5 val clips). Truncation is **not** an option — it breaks CTC
  audio↔transcript alignment.
- `LengthBucketBatchSampler(frame_budget)` sorts by length and caps
  `batch_size * max_frames ≤ frame_budget`, so long clips form small batches.
  `manifest.csv` carries a `duration` column for this (added post-hoc).
- Worst-case batch (8 × 30 s) peaks at **5.2 GB** with `frame_budget=24000`; the
  real runs use ~36000. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` is set but
  unsupported on Windows (harmless) — bucketing is what actually bounds memory.

## Key facts / decisions

- **4x time subsampling** (two stride-2 convs): 100 fps → 25 fps. Verified CTC
  length-feasible (output frames ≥ phoneme targets) on sampled batches; `CTCLoss`
  uses `zero_infinity=True` so any rare violation contributes 0 instead of erroring.
  `train.py` counts such items per epoch (`inf=` in the log) as a data-health signal.
- **Emformer right-context quirk:** Emformer treats the last `right_context_length`
  input frames as lookahead, so its output tensor is 1 frame shorter than the lengths
  it passes through. `model.forward` clamps `out_lengths` to the real time dim.
- **Default config:** d_model 256, 12 layers, 4 heads, ffn 1024, segment 4,
  left/right context 32/1, memory 4. ~9.8M params (small end of the 10–30M target,
  good for low-end phones). Streaming latency ≈ right_context (1 frame = 40 ms).
- **Metric:** greedy CTC phoneme error rate (PER) on val. This is a Stage-1 proxy;
  the real target is ayah-ID accuracy, measured end-to-end once the Stage-2 matcher
  exists (`matcher/`, `eval/`).
- **Split is by reciter** → val/test PER measures speaker generalization, the thing
  that actually matters for new users.

## Known noise / gotchas

- A handful of MP3s emit libmpg123 ID3/frame warnings on decode (`unrealistic small
  tag length`, `part2_3_length too large`). soundfile recovers and returns audio —
  non-fatal. Not yet suppressed (C-level stderr; fiddly on Windows).
- Augmentation (phone IRs, RIRs, noise, codec round-trips, SpecAugment) is **not yet
  wired in** — phase 1 trains on clean audio. Phase 2 (waveform aug + RetaSy learner
  adaptation) needs on-the-fly augmentation in `data.py.__getitem__`.
- `num_workers>0` on Windows uses spawn; `data.py` is import-safe so it works, but
  the first batch is slow to warm up.
