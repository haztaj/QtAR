# RESUME — full-Quran retrain (handoff, 2026-07-13)

Cross-session/cross-OS handoff note. Read this first when picking up the full-Quran retrain
(e.g. a fresh Claude Code session on the Ubuntu box).

## Where things stand

- **Corpus expanded to the FULL QURAN** (all 114 surahs / 6,236 ayat), user-signed-off 2026-07-13.
  Data layer done + committed + pushed. See root `CLAUDE.md` status entry for full detail.
- **Phase-1 warm-start training PAUSED at epoch 15** (stopped intentionally to continue on Ubuntu).

## Running on Ubuntu (booted on the same hardware; work from the same `C:\` folder)

The simplest path is to use the **same repo folder off the Windows drive** — the checkpoints AND the
34 GB audio corpus are already there, so no copying. `C:\` mounts clean R/W because Windows Fast
Startup + hibernation are already disabled (`HiberbootEnabled=0`, no `hiberfil.sys`).

```bash
# 1. Mount the Windows drive (find the partition with: lsblk -f). ntfs3 kernel driver = R/W.
sudo mkdir -p /mnt/c && sudo mount -t ntfs3 /dev/nvme0n1p3 /mnt/c   # adjust device
cd /mnt/c/Users/hazem/projects/QtAR

# 2. Git works from the NTFS mount after 3 one-time config fixes (cosmetic/ownership guards —
#    none break git; commit/pull/push all work):
git config --global --add safe.directory /mnt/c/Users/hazem/projects/QtAR   # "dubious ownership"
git config core.filemode false      # NTFS has no Unix perm bits -> avoids fake "mode changed" diffs
git config core.autocrlf input      # only if CRLF/LF noise shows files as modified (harmless otherwise)
git pull                            # get RESUME.md + the committed corpus/code changes

# 3. Python env (native Linux — no ffmpeg_fix hack needed; ffmpeg/libsndfile are trivial here):
#    install the CUDA PyTorch wheel + deps (torch/torchaudio, soundfile, pandas, onnxruntime,
#    rapidfuzz, audiomentations, mutagen, lhotse). Then smoke-test:
python training/train.py --smoke --num-workers 0

# 4. Full performance (Linux advantages over Windows): set the CPU governor to performance,
#    and raise --frame-budget well above the Windows 20000 cap (no WDDM cliff, no desktop VRAM
#    contention). Benchmark one epoch and push the budget up until GPU memory is comfortably used.
sudo cpupower frequency-set -g performance   # optional; needs linux-tools
```

**Training-I/O caveat:** reading 123k MP3s/epoch off the NTFS mount is slower than native ext4.
Benchmark one epoch first. If I/O-bound, copy ONLY `data/raw/audio/` to the ext4 drive and repoint
the manifest there; keep everything else (git, checkpoints, code) in the same `C:\` folder.

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
