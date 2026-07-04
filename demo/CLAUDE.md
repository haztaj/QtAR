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

**Pauses between ayat (`auto`/`stream`):** the loop resets on a **Silero VAD** speech-end
(min-silence 500 ms, noise-robust) so each pause isolates the next ayah as a clean single-ayah
decode (in-distribution). This replaced a crude energy + 2 s counter that missed sub-2 s pauses
(a real recitation with 0.5–1.5 s gaps got stuck on the first ayah — the gaps never crossed 2 s,
so the buffer never reset). Continuous recitation yields no VAD 'end' until the finish, so the
content matchers (sliding+stream) run as before.

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
python demo/live_detect.py                       # auto mode (default) — any ayah length
python demo/live_detect.py --mode sliding        # short continuous ayat only
python demo/live_detect.py --mode stream         # prefix-anchored: long/individual ayat
python demo/live_detect.py --mode buffer         # legacy growing-buffer approach
python demo/live_detect.py --list-devices        # pick a mic index
python demo/live_detect.py --device 3            # e.g. Logitech BRIO
```

## Four modes (`--mode`)

- **`auto` (default)** — runs `sliding` **and** `stream` per hop and merges their commits
  (`demo/auto.py` `AutoDetector`), so **one mode handles any ayah length**. It works because
  the two are complementary and never conflict: `sliding` is *silent* on long ayat (a 4 s
  window of a 100+-phoneme ayah is length-pruned — it fires nothing, not garbage) and gets all
  short ayat; `stream` gets long ayat and an agreeing subset on short ones. The merge is a
  union with dedup (a `recent` window blocks repeats/belated dupes) + a tiebreak for the rare
  hop where both fire *different* new ayat (prefer the continuation of the last commit, else
  lower cost). Each sub-matcher keeps its own window + sequential context; the driver decodes
  both the fixed 4 s window and the stream's anchored buffer each hop (2 decodes; negligible at
  RTF 0.002). Validated on every fixture in one mode (78:40, 78:38→40 ×2, 85:12→16, 114:1→3 —
  see `regression.py`). Why not a single merged *scorer*: the difference is buffer discipline
  (stateless fixed window vs anchored buffer), not the cost metric — one buffer can't be both
  (prototyped; stayed stuck). A principled single detector still wants the streaming-Emformer
  decode (see `export/streaming-export-plan.md`); `auto` reaches the same behaviour now.
  **Known limit — mixed long+short *continuous*:** a passage that mixes a long ayah then
  shorter ones with no pauses (e.g. Al-Bayyinah 98:1→4) is partial. After a long ayah, the
  stream buffer stays *locked* on it (the buffer starts with it and its prefix keeps matching
  as the degrading decode lags), so a **medium** continuation in the gap (too long for the 4 s
  window, not reachable before the buffer refocuses late) is missed. A completion-refocus
  (release the buffer once a committed ayah's progress hits ~100%) recovers the *next* long
  ayah — 98:1→4 gets 3/4 (98:1, 98:3, 98:4; misses the medium 98:2). This is the growing-buffer
  decode lag, the case streaming-export fixes. **Workaround today:** a brief pause between ayat
  triggers the 2 s silence-reset, so each ayah is a clean single-ayah detection (any length).

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
  new passage). The buffer is bounded on a **refocus** signal (when a new forward leader holds
  2 hops, the driver clips the buffer to its recent ~11 s tail) — an unbounded buffer decodes
  worse and worse on multi-ayah audio and buries later ayat (a louder 2nd take of 78:38→40
  committed nothing until this was added). `--commit-cost` (0.75) is only a loose garbage gate.
  Validated offline:
  **78:38→78:39→78:40** continuous → `detect 78:38 / advance 78:39 / advance 78:40` (clean,
  no noise); 78:40 solo → `detect 78:40 @21 %`; 114 continuous → `114:1` (no false commits).
  Why not the matcher's `partial_candidates`: its min-over-nodes scoring doesn't penalize a
  short ayah when the input runs past it, so short ayat match a tiny early decode at cost ~0
  and cause false early commits — prefix alignment consumes the whole input and avoids that.
  **Known limit — short back-to-back ayat:** a run of short ayat (~4 s each, e.g. Al-Buruj
  85:12–16) degrades the growing buffer faster than refocus recovers, so `stream` typically
  gets only the first couple. **Use `sliding` for short-ayah passages** (its 4 s whole-window
  matching gets all of 85:12→16); `stream` is for long/individual ayat that sliding can't see.
  A *very* quiet continuous recording can also miss weaker continuations (they never lead) —
  the acoustic decode is the ceiling there, not the matcher. **Mode choice: sliding = short
  continuous; stream = long/individual.** A single mode for both needs the streaming-Emformer
  + CTC-alignment path (the flagged next step). (C++ core still uses sliding; porting `stream`
  is a follow-up.)
- **`buffer` (legacy)** — growing buffer + completion-decoupled advance + revisable
  commit + ayah-end detection. Works when reciters pause between ayat (VAD segments
  each); gets stuck on the first ayah in true no-pause continuous recitation. Kept as a
  fallback. All the commit/context/completion knobs below apply to this mode.

Interactive — must be run in a real terminal (mic access). Ctrl-C to quit.

## Regression check (`regression.py`)

Guards the tricky detection cases against silent matcher/model regressions:

```bash
python demo/regression.py            # -> ALL PASS / FAILURES (exit 0/1)
```

Runs each preserved `test_fixtures/*.wav` through the relevant mode and asserts the committed
ayah sequence matches a golden:

| fixture | mode | expected |
|---|---|---|
| `user_78_40_naba_long` | stream | `78:40` (a long ayah sliding can't see) |
| `user_78_38to40_naba_continuous` | stream | `78:38 → 78:39 → 78:40` (continuous long ayat) |
| `user_114_quietmic` | stream | `114:1` |
| `user_114_quietmic` | sliding | `114:1 → 114:2 → 114:3` |

The `.wav` files are gitignored (audio rule), so a missing fixture is **skipped, not failed**
— it runs wherever the audio is present (the committed `*_events.jsonl` document each case).
Add a case in `CASES` when a new fixture is preserved.

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
