# RESUME — full-Quran retrain (handoff, 2026-07-13)

Cross-session/cross-OS handoff note. Read this first when picking up the full-Quran retrain
(e.g. a fresh Claude Code session on the Ubuntu box).

## ⏩ LIVE SESSION STATE (2026-07-14) — read this first

**★ SHIPPED (2026-07-16): `best_full_tu.pt` — real-learner-adapted, DEPLOYED.** Fine-tune of
`best_full_p3` on 17,837 real learner recitations (`data/raw/phase2/tusers_manifest.csv`, all 114
surahs, source **https://archive.org/download/quran-speech-dataset**, licensing cleared) mixed at 8.5% into the phase-3 manifest
(`combined_train_tusers.csv`), 8 ep moderate aug. **Best = epoch 7, val_PER 0.0443 (best clean ever).**
Validation (all improved vs `best_full_p3`): audio_bench **141/151 (93%)** @ normRms 0.15 (best;
was 137), clean test **95.8%**, suppression 0.976. **All three of the user's failing phone takes
fixed:** Al-Hijr now tracks 15:1–15:5 with the wrong-`2:1` fast-commit glitch GONE; An-Nahl complete
16:1–16:7; the quiet An-Nahl take now gets 16:1–16:3. This vindicates the diagnosis that the crowding/
continuous-tracking wall is decode-quality-limited and REAL learner data (not synthetic aug — that
retrain failed) is the lever. **Deployed model-only** (host + manifest bump to `best_full_tu-22s-v5`;
the full-Quran index + native chain fixes + normRms 0.15 were already on-device from earlier this
session) — the app auto-downloads on next launch, no APK rebuild. ONNX set: `model_full_tu_{22s,5s}.int8.onnx`
+ re-exported streaming graphs; all hosted on the `model` release. RetaSy short-surah metric dipped
to 74.9% (narrow 14-surah slice; broad real-audio improved — do not chase it).
audio_bench arm: `--arms tu`.

**Android/deploy toolchain (Ubuntu):** `JAVA_HOME=/home/hazem/jdk17`,
`ANDROID_HOME=/home/hazem/android-sdk` (NDK 27.2.12479018), adb at
`$ANDROID_HOME/platform-tools/adb`. Gradle fails with "JAVA_HOME is not set" without the export.
Deploying a debug build: ALWAYS `adb shell am force-stop com.quranrecite` before `adb install -r`
(a plain reinstall does not restart a running app -> you test stale code). Release/Play process:
see `sdk/android/README.md` ("Release to Play" — the store page already exists; just bump
`versionCode` + `bundleRelease` + upload).

**Env:** venv at `/home/hazem/qtar-venv` (Python 3.12, torch 2.13+cu130). Run training/eval with
`/home/hazem/qtar-venv/bin/python`. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
The Windows-path fix (`training/data.py::_localize_path`) re-anchors manifest paths — already applied.

**Phase-1 DONE** — stopped at epoch ~19: cosine schedule hit `lr=0` (frozen, val_PER plateaued 0.064).
Best base = `best_full.pt` (epoch 13, PER 0.0618). Do NOT try to squeeze more phase-1.

**Phase-2 finding (IMPORTANT):** the first full-corpus phase-2 (`best_full_p2.pt`) REGRESSED learners.
End-to-end ayah-ID on the full 6,236 index (eval/evaluate.py):
| model | trained on | clean top-1 | learner top-1 |
|---|---|---|---|
| `best_s123_mic_clean` (DEPLOYED) | 1,057 ayat | 93.6% | **77.0%** |
| `best_full` (phase-1) | full | 95.0% | 35.8% |
| `best_full_p2` (phase-2, FAILED) | full | 95.0% | **15.8%** |
Root cause: RetaSy learner data diluted to 0.76% of the full pool (was ~4% in the deployed recipe);
heavy augmentation on 99% professional audio specialized the model away from real learner voices.
`best_full_p2` is NOT deployable. Learner audio only covers 14 short surahs / 76 ayat (78,97,103–114) —
the standing data hole; the learner metric can't speak to the other 100 surahs.

**WINNER: `best_full_p2r2.pt`** — full-corpus phase-2, 2 rebalance rounds done (2026-07-14). Best ckpt =
epoch 15, val_PER 0.050. Round 2 = RetaSy oversampled 13× → **9.07%** (`combined_train_rebal2.csv`),
16-epoch cosine from `best_full`, `--augment` + noise. Log: `training/exp/train_full_p2r2.log`.

**FINAL end-to-end ayah-ID (full 6,236 index, eval/evaluate.py, top-1):**
| model | trained on | clean | learner | note |
|---|---|---|---|---|
| DEPLOYED `best_s123_mic_clean` | 1,057 ayat | 93.6% | 77.0% | old base, short-surah home turf |
| failed `best_full_p2` (0.76% RetaSy) | full | 95.0% | 15.8% | collapsed — dead |
| `best_full_p2r` round-1 (4.4%) | full | 95.2% | 59.8% | superseded |
| **`best_full_p2r2` round-2 (9.1%)** | full | **95.2%** | **74.3%** (top-3 79.8%) | **DEPLOY CANDIDATE** |

RetaSy weight is the dominant learner lever: 0.76%→15.8, 4.4%→59.8, 9.1%→74.3. Round-2 climb:
ep5 69.2 → ep10 71.5 → ep15 74.3 (tapering as LR annealed). **Verdict: `best_full_p2r2` is a deployable
full-corpus replacement** — beats deployed on clean (+1.6, best PER 0.050), matches on learner within noise
(74.3 vs 77 on a 530-clip / 14-short-surah test where the Juz-30 model has a built-in edge; top-3 already
beats it), and is the ONLY model with real full-114 coverage. The last ~3 learner pts need real learner
audio for more surahs (the data hole), NOT more epochs — training chase STOPPED here (user agreed on Telegram).

**REDEPLOY (Phase C) — STARTED 2026-07-14, then BLOCKED by an audio_bench regression. Details:**
- ✅ Ubuntu toolchain installed (user ran `sudo apt install build-essential cmake ninja-build`).
- ✅ ONNX export from `best_full_p2r2` — `model_full_p2r2_{22s,5s}.int8.onnx` + streaming graphs
  (`stream_conv.onnx`+`stream_encoder.int8.onnx`, OVERWRITTEN from p2r2). All parity + int8-lossless.
- ✅ C++ SDK built natively on Linux: downloaded ORT `sdk/build/onnxruntime-linux-x64-1.22.0`, then
  `cmake -S sdk/core -B sdk/build/cmake-linux -G Ninja -DORT_HOME=<that> -DCMAKE_EXE_LINKER_FLAGS="-L<ort>/lib -Wl,-rpath,<ort>/lib"`.
  Conformance ALL PASS. Binary: `sdk/build/cmake-linux/test_detector` (audio_bench BIN auto-detects it).
- ✅ Full-Quran chain assets rebuilt: `segment_phonemes.json` 1029→**6877** (text-only regen; `.bak_1057` kept),
  `ambiguous_units.json` (full Quran), `conformance/generate.py` regenerated `conformance/assets/`
  (`ayah_phonemes.json` 6236, `unit_phonemes.json` **10510**) — NOTE generate.py then errored on Windows-path
  audio decode for the golden *fixtures* (runtime assets are fine; conformance goldens NOT regenerated → the
  conformance TEST may now mismatch until generate.py's manifest path is localized like `data.py::_localize_path`).
- ❌ **audio_bench GATE FAILED for `best_full_p2r2`: 118/151 (78%) vs anchor `p31suf` 134/151 (89%).** Gap
  concentrated on CONTINUOUS phone audio (pulled sessions + continuous streams). Single-clip evaluate.py
  (74.3% learner) did NOT catch it — audio_bench did (exactly why it's the ship gate).
- 🔬 **Root cause CONFIRMED — `best_full_p2r2` lacks PHASE-3 continuous training.** `probe_suppression.py`:
  p2r2 ratio **0.685** (suppressing) vs p31 **0.876** (healthy). The full-corpus chain was phase-1→phase-2 only;
  it never got the phase-3 continuous-concatenation fine-tune that cured the Emformer repetition-suppression
  pathology. On continuous recitation it deletes repeated phrases → chain decode drops units.
- **FIX APPLIED — `best_full_p3.pt` (phase-3, epoch 5) — SUPPRESSION CURED + GATE RECOVERED.** Built a
  full-corpus phase-3 mixed manifest `data/raw/phase3/combined_train_p3full.csv` (base = `combined_train_rebal2.csv`
  9%-RetaSy all-114 + continuous `windows_train.csv` oversampled ×5 → 14% of mix; keeps coverage + learner while
  adding continuous windows). p3 = fine-tune `best_full_p2r2`, lr 1e-4, NO augment, tag `_full_p3`; STOPPED at
  epoch 5 (evidence-driven). Results: **probe 0.685→0.889 (cured), learner 74.3→83.4% (>deployed 77), clean
  val PER 0.0485.** Exported `model_full_p3_{22s,5s}.int8.onnx` (parity + int8-lossless).
  **audio_bench GATE: `p3fullsuf` 132/151 (87%) vs p2r2suf 118 (broken) vs anchor p31suf 134 (89%)** — the
  continuous-audio regression is essentially gone (real_112_114_cont 13→15 EXACT, fix_78_38_40 1→3, sess_983417
  2→5, sess_836602 1→4; and p3full BEATS anchor on cold-start fix_98_1_3 3/3 vs 1/3). NO restore stage was even
  needed (my rebalanced base kept learner during p3, unlike p31). `best_full_p3` = full-114 coverage + anchor-parity
  gate + best learner ⇒ **the ship candidate.** audio_bench arms added: `p3full`, `p3fullsuf`.
- **OPTIONAL polish (p3.1 restore):** 5 ep on `combined_train_rebal2.csv`, lr 5e-5, --augment, from best_full_p3 —
  MIGHT close the 2-unit gate gap / firm up robustness. Low value here (learner already up); do only if chasing >134.
- **SHIP = best_full_p3 (user chose accept-and-ship, no restore). STAGED this session:**
  - Final onnx set re-exported from `best_full_p3` (22s + 5s + streaming re-exported; streaming PASS, one clip
    +1 int8 phoneme — tolerated). Hosting bundle at `export/onnx/host_full_p3/` with hosted names
    (`model.int8.onnx`=22s, `model_suffix.int8.onnx`=5s, `stream_conv.onnx`, `stream_encoder.int8.onnx`) +
    **`model_manifest.json`** (version `best_full_p3-22s-v4`, real sha256s, GitHub release URLs).
  - Bundled Android assets UPDATED to full-Quran (`sdk/android/quranrecite/src/main/assets/quranrecite/`:
    `ayah_phonemes.json` 6236, `unit_phonemes.json` 10510, `ambiguous_ayat.json` 466). ⚠️ These are BUNDLED in
    the app (not downloaded), so the full-Quran index REQUIRES an app rebuild — the model download alone won't update it.
  - ✅ **HOSTED (2026-07-14):** uploaded all 4 onnx + `model_manifest.json` to the `model` release
    (`gh release upload model export/onnx/host_full_p3/* --clobber`). Verified end-to-end: hosted manifest =
    `best_full_p3-22s-v4`, hosted `model.int8.onnx` sha256 matches the manifest (the app's download check). The
    model download path is LIVE — apps doing a version check pull best_full_p3.
  - ✅ **ANDROID BUILD DONE (2026-07-15).** Ubuntu Android toolchain set up userspace (no Android Studio):
    JDK 17 `~/jdk17`, SDK `~/android-sdk` (cmdline-tools 12.0) with `platform-tools`, `platforms;android-35`,
    `build-tools;35.0.0`, `ndk;27.2.12479018`, `cmake;3.22.1`; `sdk/android/local.properties` → `sdk.dir`.
    `JAVA_HOME=~/jdk17 ANDROID_HOME=~/android-sdk ./gradlew :demo:assembleDebug` → **BUILD SUCCESSFUL** (2m16s).
    `demo/build/outputs/apk/debug/demo-debug.apk` (75.6 MB) VERIFIED to bundle the full-Quran index
    (ayah_phonemes 6236, unit_phonemes 10510, ambiguous_ayat 466) + native `.so` (onnxruntime + core),
    ABIs arm64-v8a + x86_64. Download-build → bundles full-Quran index, pulls hosted best_full_p3.
  - ✅ **ON-DEVICE SMOKE TEST PASSED (2026-07-15, Samsung SM-F966B / Android 16 / arm64).** `adb install`
    (uninstalled the old Windows-key build first) + launch: app runs, no crash, `libquranrecite_jni.so` loads.
    VERIFIED on-device: extracted bundled index = full Quran (ayah_phonemes 6236, unit_phonemes 10510); the app
    DOWNLOADED the hosted set `best_full_p3-22s-v4.onnx` + `.stream_conv`/`.stream_encoder`/`.suffix` from the
    GitHub release. **BUG FOUND + FIXED:** `ModelManager.ASSETS_VERSION` was still `"s123-v1"` — the corpus
    changed so existing users updating would NOT re-extract the new index (only fresh installs would). Bumped to
    `"full-p3-v1"`; rebuild+reinstall confirmed re-extraction to `assets-full-p3-v1/` (6236). **Live-recitation
    detection test still pending (needs a human reciting into the phone; app debug logging is off by default).**
  - **REMAINING (user, for a PUBLIC release):** `:demo:assembleRelease`/`bundleRelease` needs the upload keystore
    (`sdk/android/keystore.properties`, gitignored, user's passwords) → signed `.aab` → Play Console. Launcher
    icon still default. OPTIONAL: regenerate conformance goldens — `conformance/generate.py` errored on a
    Windows-path audio decode; localize its `manifest.csv` paths (like `data.py::_localize_path`) first, else the
    conformance TEST mismatches the new full-Quran `unit_phonemes.json` (runtime unaffected).
- audio_bench.py edits this session: `BIN` auto-detects `cmake-linux/test_detector`; `compose()` localizes
  manifest paths; added arms `p2r2 / p2r2suf / p2r2stream / p2r2sufvad`.
- Nothing committed — `.pt`, exports, rebuilt assets all on disk only (assets/onnx are gitignored or working-tree).

**Telegram channel:** working for OUTBOUND, but inbound requires launching the session with
`claude --channels plugin:telegram:telegram` (channel delivery is a startup opt-in; reload can't add it).
`.mcp.json` was patched to absolute bun path (`/home/hazem/.bun/bin/bun`) — a plugin update reverts it.

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

### What actually worked on Ubuntu (2026-07-14, real setup notes)

The box came up as a near-bare Ubuntu (kernel 7.0): **system Python is 3.14** (no PyTorch wheels),
**no pip/venv/conda**, and **`sudo` needs a password** (can't `apt install` non-interactively). The
no-sudo path that worked end-to-end:

```bash
# 1. uv (userspace installer, no sudo). NOTE: a snap-confined shell (e.g. VS Code / Claude Code)
#    redirects $HOME/.local -> $HOME/snap/code/<n>/.local, so uv lands there:
curl -LsSf https://astral.sh/uv/install.sh | sh          # -> ~/snap/code/<n>/.local/bin/uv
UV=~/snap/code/*/.local/bin/uv                           # or wherever `find ~ -name uv` shows

# 2. managed CPython 3.12 + a venv on ext4 (NOT on the NTFS mount — avoids symlink quirks):
$UV python install 3.12
$UV venv /home/hazem/qtar-venv --python 3.12
PY=/home/hazem/qtar-venv/bin/python

# 3. torch (Blackwell/sm_120 needs cu13x). torch 2.13.0+cu130 works on the 5080:
UV_HTTP_TIMEOUT=300 $UV pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
#   (pypi.nvidia.com is slow — a wheel may time out once; just re-run, uv caches the rest.)
$UV pip install numpy pandas scipy soundfile mutagen rapidfuzz     # phase-1 core deps
#   phase-2 --augment also needs audiomentations, BUT numpy 2.5 is too new for numba/librosa
#   -> pin numpy (e.g. numpy<2.3) before installing audiomentations/torch-audiomentations.
```

**Manifest path fix (REQUIRED after the move):** every manifest (`data/raw/audio/manifest.csv`,
`data/raw/phase2/*.csv`) stores absolute **Windows** paths (`C:\...\QtAR\data\raw\audio\...`), so on
Linux all audio silently loaded as *silence*. Fixed in code: `training/data.py::_localize_path()`
re-anchors any path on its `/data/` segment onto the current repo root (location-independent,
idempotent) — applied to `df["path"]` and the `bad_files.txt` set at load. No CSV regeneration needed.

**GPU / budget (measured on the 5080, desktop holding only ~1.3 GB — the Windows VRAM-contention
problem is gone):** `--frame-budget 48000 --num-workers 8` → ~8.1 GB peak (9.8 GB total),
3724 batches/epoch, ~8.3 min/epoch. Throughput is GPU-compute-bound past ~48k (32k→48k was only
+13% for +50% budget), so pushing budget toward full VRAM buys little and risks OOM on long clips.
A couple of `hani_rifai` MP3s hit "Unspecified internal error" → silence fallback (bad_files
candidates; harmless, ~2 per 1000 clips). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set.

**Phase-1 resume launched 2026-07-14** (epoch 16→30, `--resume last_full.pt`, tag `_full`,
lr 2e-4, budget 48000); log at `training/exp/train_full_resume.log`. Then → Phase-2 below.

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
