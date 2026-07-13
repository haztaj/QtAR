# RESUME — full-Quran retrain (handoff, 2026-07-13)

Cross-session/cross-OS handoff note. Read this first when picking up the full-Quran retrain
(e.g. a fresh Claude Code session on the Ubuntu box).

## Where things stand

- **Corpus expanded to the FULL QURAN** (all 114 surahs / 6,236 ayat), user-signed-off 2026-07-13.
  Data layer done + committed + pushed. See root `CLAUDE.md` status entry for full detail.
- **Phase-1 warm-start training PAUSED at epoch 15** (stopped intentionally to continue on Ubuntu).

## Checkpoints (⚠️ NOT in git — gitignored `.pt`; live on disk only)

- `training/exp/best_full.pt` — best model, **val PER 0.0618** (epoch 13) on the full 114-surah val.
- `training/exp/last_full.pt` — full resume state (epoch 15, step 60226, optimizer/scaler/step/best_per).
- Warm-started from `training/exp/best_s123_p31.pt`.

To carry to another machine: copy `training/exp/*.pt` (best_full ~40 MB, last_full ~118 MB). The
34 GB audio corpus (`data/raw/audio/`) + manifests are also gitignored — either mount the Windows
`C:\` (Fast Startup is already disabled, so NTFS mounts clean R/W) or copy `data/raw/` over.

## Resume phase-1 (if continuing it)

```bash
python training/train.py --epochs 30 --tag _full --resume training/exp/last_full.pt \
    --lr 2e-4 --frame-budget 20000 --quad-budget 6e7 --num-workers 8
# continues at epoch 16 with the cosine schedule intact.
```

On Linux you can raise `--frame-budget` well above 20000 (see "Windows lessons" below) — the 20000
was a Windows-only safety cap. Benchmark one epoch and push it up until GPU memory is comfortably used.

## Phase-2 (NEXT — the higher-impact stage; split already built)

Mic/RetaSy augmentation adaptation on the full corpus. Splits already generated:
- `data/raw/phase2/combined_train.csv` (150,812 clips = full clean train + cleaned RetaSy)
- `data/raw/phase2/retasy_test.csv` (530 clips / 57 held-out learner reciters)

```bash
python training/train_supervisor.py --epochs 12 --tag _full_p2 \
    --train-manifest data/raw/phase2/combined_train.csv --epochs-per-run 1 \
    --init-from training/exp/best_full.pt -- \
    --lr 1.4e-4 --augment --frame-budget 24000 --num-workers 8
# (frame-budget can go higher on Linux)
```
Note: the audiomentations RAM-leak that motivates `train_supervisor.py` per-epoch restarts is much
milder on Linux (fork workers); the supervisor is still fine to use.

Optional follow-on: rescue ~2,900 Al-Fatiha (surah-1) RetaSy learner clips the old filter dropped
(the filter is already widened) — needs re-running `extract_retasy.py` + `retasy_flag.py` (GPU) +
a by-ear review pass, then regenerate the phase-2 split.

## After training

- Re-export ONNX (windowed + streaming + suffix), then regenerate the chain/segment assets:
  waqf `segment_phonemes.json`, `ambiguous_units.json`, forced-alignment `segment_spans.csv`
  (these were deferred — coupled to the retrained model), and re-validate on `research/audio_bench.py`.
- **The ~6× index (6,236 vs 1,057 ayat) means the detection design (early-prefix, chain decoder,
  export window sizes) needs re-validation on audio_bench before shipping.**

## Windows lessons (why we're moving to Ubuntu)

- **VRAM contention:** the Windows desktop/background apps hold ~7–8 GB of the 16 GB card, so training
  had to stay under the shared headroom (OOM'd at epoch 3 at frame_budget 36000; dropped to 20000,
  ~4.4 GB peak). Linux (esp. minimal/headless) doesn't share the GPU with a heavy desktop → higher budget.
- **WDDM paging cliff:** Windows pages VRAM to system RAM near ~8 GB allocated → throughput collapse.
  The whole quad-budget batching guard exists for this. No WDDM on Linux.
- **EcoQoS throttling:** detached/background processes are power-throttled while the machine is idle
  (epochs 72–96 min idle vs ~17 min active). Not a thing on Linux (use the `performance` cpu governor).
- **Filesystem:** training only READS the corpus, so a read-only NTFS mount of `C:\` is fine (checkpoints
  write elsewhere). For best throughput copy the corpus to native ext4, but only if the target drive is
  fast — benchmark first (reading off a fast internal NVMe NTFS can beat a slow external ext4).
