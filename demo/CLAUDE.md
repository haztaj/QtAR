# demo/ — live microphone ayah detection

`live_detect.py` runs the full pipeline on live mic input:

  mic → Silero VAD gate → 16 kHz log-mel → Emformer+CTC → greedy phonemes
      → fuzzy matcher → ranked surah:ayah → persistence-based commit

Uses the **PyTorch checkpoint** directly (variable length — avoids the fixed ONNX
window). Shows the running top-3 + a progress % through the committed ayah, commits
when the top-1 holds the margin for K phonemes (`✓ DETECTED`, can `REVISED →`), prints
the Arabic text, and on ayah completion prints `● COMPLETE … → up next: X+1` and
pre-advances the sequential context so the UI can show the next ayah.

Sequential context carries across ayat (after X, expect X+1; surah boundaries handled),
and completion (`ayah_progress`) drives auto-advance — the live realization of
"detect ayah 11 done → present ayah 12, confirm as we go."

**Continuous recitation (no pauses):** completion advances *immediately* — when an ayah
completes the buffer is reset to a short tail (`--reset-tail`, 0.3 s) so the next ayah
is detected fresh without waiting for a VAD pause. Completion keys off the **top**
candidate (`detected = committed or top_key`), **not** a formal commit — a shared-prefix
ayah (113:1 vs 114:1) can block a commit yet still complete and advance. (Earlier the
demo only advanced on a VAD pause AND required a commit, so continuous reciters got
stuck on the first ayah — see the 2026-06-30 session investigation.)

**Quiet mics:** `--norm-rms` (0.1) gain-normalizes each window into the (loud) training
distribution. Studio training audio is ~RMS 0.1; a quiet mic (~0.02) otherwise decodes
to mostly blanks. The matcher's completion also requires the input to have produced
≥70% of the ayah's phonemes (`min_input_frac`), so a sparse/quiet decode can't falsely
"complete" a long ayah via deletions.

**Sticky context:** the prior is strong + builds a streak (shown as `*N` in the live
line) so a sequence resists jumping to another surah. Knobs: `--context-bonus` (0.22),
`--surah-bonus` (0.10), `--streak-bonus` (0.05). If it ever sticks too hard when you
deliberately jump, the revisable commit still corrects it (and resets the streak).

## Run

```bash
python demo/live_detect.py                       # sliding mode (default), default mic
python demo/live_detect.py --mode stream         # prefix-anchored: early detection, any length
python demo/live_detect.py --mode buffer         # legacy growing-buffer approach
python demo/live_detect.py --list-devices        # pick a mic index
python demo/live_detect.py --device 3            # e.g. Logitech BRIO
```

## Three modes (`--mode`)

- **`sliding` (default)** — fixed-window segmentation for **continuous (no-pause)
  recitation**. Slides a window (`--window` 4 s, `--hop` 1 s) across the stream; each
  window is classified by whole-window edit distance to each ayah (`demo/sliding.py`
  `SlidingWindowSegmenter`); a state machine assembles confident windows (cost <
  `--window-cost` 0.30) into the ayah sequence, biased by the sticky context. Boundaries
  are found by CONTENT, not pauses. Bounded per-window cost (~0.02 RTF, lighter than
  buffer for long sessions). Validated on the quiet-mic continuous session
  (114:1→2→3). The single-ayah model decodes each window well; the long growing-buffer
  under-decodes past ayah 1, which is why this mode exists.
- **`stream`** — **prefix-anchored** early detection (`demo/streaming.py` `StreamDetector`).
  Scores each ayah by PREFIX ALIGNMENT (`prefix_align`: min cost to turn the input into a
  *prefix* of the ayah, input fully consumed, ayah free to end anywhere). So an ayah of **any
  length** surfaces as soon as its prefix is discriminative — *before it finishes* — and a
  short ayah drops out once the input outgrows it (its tail becomes insertion cost). This is
  the mode for **long ayat the sliding window can't see**: sliding matches each 4 s window as
  a *whole* against *whole* ayat, so a long ayah (e.g. 78:40, 105 phonemes) is length-pruned
  from every window and never detected — `stream` detects it at ~20 % recited.
  Commit policy is **rank persistence**, not absolute cost: on a quiet mic the cost is high
  (~0.4–0.6) with small margins to confusables, yet the correct ayah holds **#1 for many
  hops**, so an ayah commits when it leads for K hops (`persistence` 3; a non-continuation
  **jump** needs `jump_persistence` 5; a **backward** step to an earlier ayah of the same
  surah is suppressed as decode noise). The buffer is **not reset per ayah** (resetting to a
  tail decodes garbage) — it grows (capped 30 s) and the top-1 naturally hands off
  A→A+1→A+2, so a continuous recitation is committed ayah by ayah; sequential context biases
  the expected next and resists backward flickers. A ≥ 2 s silence resets buffer + context (a
  new passage). `--commit-cost` (0.55) is only a loose garbage gate. Validated offline:
  **78:38→78:39→78:40** continuous → `detect 78:38 / advance 78:39 / advance 78:40` (clean,
  no noise); 78:40 solo → `detect 78:40 @21 %`; 114 continuous → `114:1` (no false commits).
  Why not the matcher's `partial_candidates`: its min-over-nodes scoring doesn't penalize a
  short ayah when the input runs past it, so short ayat match a tiny early decode at cost ~0
  and cause false early commits — prefix alignment consumes the whole input and avoids that.
  Known limit: a *very* quiet continuous recording may still miss weaker continuations (they
  never lead); the acoustic decode is the ceiling there, not the matcher. (C++ core still uses
  the sliding segmenter; porting `stream` is a follow-up.)
- **`buffer` (legacy)** — growing buffer + completion-decoupled advance + revisable
  commit + ayah-end detection. Works when reciters pause between ayat (VAD segments
  each); gets stuck on the first ayah in true no-pause continuous recitation. Kept as a
  fallback. All the commit/context/completion knobs below apply to this mode.

Interactive — must be run in a real terminal (mic access). Ctrl-C to quit.

## Session recording + investigation

Each run records to `demo/sessions/` (reset/overwritten each session — one copy):
- `session.wav` — full 16 kHz mono recording (includes pauses, so timestamps are real)
- `events.jsonl` — one line per finalized ayah: `index`, `start_s`/`end_s` +
  `start_sample`/`end_sample`, `detected`, `committed`, `completed`, `expected` (context),
  `streak`, `top3` (key/cost/progress), `phonemes`
- `meta.json` — model, args, start time

Disable with `--no-record`; change location with `--session-dir`.

**Investigate a detection** with `demo/analyze_session.py`:
```bash
python demo/analyze_session.py            # list every ayah (index, time, detected, top-3)
python demo/analyze_session.py 4          # deep-dive ayah #4
python demo/analyze_session.py 4 --play-out demo/sessions/aya4.wav   # also export the clip
```
The deep-dive re-extracts the exact audio segment, re-runs the model+matcher, and prints
the decoded phonemes and candidate ranking **with and without** the context that was
active — i.e. it answers "why did ayah N detect as X?". This is the workflow for
"investigate the wrong detection for the 4th aya".

## Notes / knobs

- `--threshold` / `--persistence`: commit policy (defaults T=0.15, K=5, matching the
  tuned values in matcher/CLAUDE.md). Very short ayat (Al-Ikhlas, An-Nas) may not reach
  K=5 — lower K or rely on the pause "best guess".
- `--infer-every` (0.4 s): how often the running guess updates while reciting.
- `--vad-threshold` (0.5): Silero speech sensitivity; raise in noisy rooms.
- Validated offline on known clips (112:1, 114:1, 108:1 all correct). Live capture is
  the only part that needs a real mic to exercise.
- Feature front-end is imported from `training/data.py`, so it always matches training.
