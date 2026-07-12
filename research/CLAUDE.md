# research/ — detection-methodology experiments

Hypothesis-driven experiments at the Python matcher layer (research-first: methodology
findings > shipping UI — see root CLAUDE.md). GPU work is cached once (decoded phoneme
streams with per-phoneme frame times); the experiment arms run CPU-only and iterate fast.

## segment_ablation.py — what is the right detection unit?

Motivated by two findings (2026-07-06):
- **Negative result:** no per-clip commit-margin policy fixes long-ayah false commits
  (4 policy families swept) — whole-ayah units are information-limited: long shared
  prefixes diverge late.
- Waqf segments (median 8.5 s) put the corpus back in the regime the matcher already
  handles at 97% (`data/build_segments.py`).

Arms, evaluated on **segment-cut test-reciter clips** (spans from
`data/raw/segments/segment_spans.csv`):
- **A — whole-ayah index** (status quo): quantifies mid-ayah failure (a clip from
  segment k>=2 can't match ayah-start references).
- **B — segment index**: trie over 1,029 waqf segments + all unsegmented ayat as
  single units (the realistic replacement index).

Metrics per arm: top-1 unit / parent-ayah accuracy, time-to-detection in **seconds**
(per-phoneme frame times x 0.04), split by segment position (seg 1 = ayah start vs
seg >= 2 = **mid-ayah cold start** — the pause-resume case the live VAD reset needs).

```bash
python research/segment_ablation.py            # decode-cache once, then CPU arms
```

Cache: `data/raw/segments/test_streams.pkl` (gitignored).

## dissect_misses.py — why the segment index misses

Of arm B's 286 parent-misses (2026-07-06): **74.5% are exact twins** (phoneme-identical
references — Baqarah's recurring refrains like «أفلا تعقلون»; indistinguishable by
construction, resolved by sequential context + ambiguity deferral, the machinery already
proven at ayah level), 2.4% near-twins, 0.7% margin-close. True residual (truth ranked
low): **2.07% of all streams**, skewing short (<4 s) — decode/alignment quality, the
posterior-aware matcher bridge is the known lever.

**Effective context-resolved ceiling of the segment index: 97.9%** — equal to the
ayah-level Juz-Amma baseline (97.4%), while adding mid-ayah entry, pause-resume, and
5-7 s TTD. Conclusion so far: **the waqf segment is the right detection unit**; the
ayah is a derived (parent) label. Next: chained decoder over continuous clips (segment
n -> n+1 context) to validate the ceiling empirically; segment-level ambiguity map
(extend matcher/find_ambiguous.py) to formalize the twin classes for deferral.

## chain_sliding.py — chained segment decoding (v9 + filter bank, working)

Sliding-window chaining over full continuous clips. The design emerged from a measured
iteration chain (each step isolated by a targeted diagnostic — see git history):

| v | design | positional | lesson |
|---|---|---|---|
| v1 (chain_decoder.py) | commit-and-reset anchor | 33% (pos2 10%) | resets lose the next unit's prefix -> cascade |
| v2 | sliding + trie-terminal scoring | 0.2% | terminal scoring needs boundary-aligned input |
| v3 | + whole-window edit-norm | 8% | edge junk swamps unaligned windows |
| v4 | + restart shortlist + infix | 0.8% | trie shortlist can't retrieve mid-window entries |
| v5 | + 3-GRAM shortlist, loose len gate | 31% | retrieval fixed (oracle: shortlist 95%, truth cost median 0.08) |
| v6 | + maximal munch, confidence votes | 59% | short formulaic refs embed at ~0 cost; SER 112% (insertions) |
| v7 | + temporal emission gating | 59%, SER 35%, exact 46% | soft time anchor: gate emissions, never matching |
| v8 | + TWIN SUBSTITUTION via context | 63% | twins tie on cost AND length; only context can pick |
| v9 | + blended selection (cost - 0.15*coverage) | oracle D 78% | pure-cost fires snippets, pure-longest swallows short truths |
| v9+ | + MATCHED FILTER BANK: scales (0.7,1,1.5,2.2), TIGHT gate 0.5n-1.3n | oracle D 85% | see below — the gate/scale pairing is the point |
| v10 | + full-counter shortlist normalization + 0.2 tiny scale | oracle D 89% | short refs were crowded out + gate-orphaned, NOT badly decoded |

**Short-ref recovery (2026-07-07) — a hypothesis overturned by its diagnostic.** The
plan was posterior-aware retrieval for the <12-ph bucket (oracle A=0%). The diagnostic
(scratch diag_short.py) showed the decodes are nearly perfect (mean 3-gram recall 0.90,
`kallā` decoded exactly, infix cost 0.00, 100% appear in the counter) — posteriors are
NOT the problem. Two mechanical faults instead: (1) the length-normalized shortlist
pass ran over the raw-count top-180 only, and a 5-ph ref ranks ~230th by raw count even
when decoded perfectly -> normalize over the FULL counter (heapq.nlargest); (2) no
window scale served the L 4-12 band (smallest window 7 s ~ 25 ph -> gate needs L >= 12.5)
-> add scale 0.2 (2 s ~ 7 ph; band ~L 4-9; noise bounded — only 16 refs live there;
0.2 beat 0.25 on the oracle: <12 D 36.4% vs 18.2%). Oracle: <12 A 0 -> 82%, D 0 -> 36%;
12-25 D 71 -> 82%; ALL D 84.6 -> 88.7%. Remaining <12 D-losses are mostly exact twins
(kallā = 89:17#01/83:14#01/104:4#01) — resolved downstream by twin substitution.

**Matched filter bank (2026-07-07).** Loss dumps showed v9's remaining D-losses were
same-parent neighbor swallows: a 5-phoneme segment covers 18% of a 10 s window while its
23-phoneme successor covers 82% — no static selection rule can save it. Adding a small
scale (0.7) with the LOOSE gate regressed end-to-end (aligned-hit 82.2 -> 72.8 on the
150-seq quick pass): small windows added recall but also noisy fires that flooded the
vote machine (raw unit SER 32.6 -> 41.6%). The fix is the pairing: each window scale
only fires refs of its own length class (gate 0.5n <= L <= 1.3n), with scale 2.2
covering the long band the tightening orphans. Recall AND precision improved together.
Oracle funnel (oracle_funnel.py, 200 clips): D 84.6% overall; 45+ phoneme refs 99-100%,
12-25 at 71.4%; only dead zone is refs <12 phonemes (11 units, 2.6% — too few 3-grams
to retrieve at all).

**Context ablation (v8, 250 clips): twins 26% -> 46% (+20), pos2 +10, positional +5.**
Position-1 twins are unresolvable cold by definition (no prior context in per-clip eval);
continuous recitation supplies context at every position — cross-clip evaluation is the
next condition to build. Oracle funnel (scratch): shortlist 95% / gate 92% / cost 91% /
wins-once 74% / 2-consec 52% — assembly now extracts nearly all available window wins.

Open gaps: pos1 69% vs oracle 74 (small assembly gap); long segments + multi-scale window
coverage; cross-ayah context evaluation; then the C++ port of the winning design.

## continuous_eval.py — multi-ayah continuous streams (the deployment condition)

Runs of 4 consecutive ayat per test reciter (concatenated cached streams, mean ~132 s,
segmented + unsegmented ayat mixed), v8 decoder with FULL context: within-ayah segment
succession + cross-ayah handoff. Metrics are alignment-based (edit traceback), not
prefix-positional — one early insertion must not mark every later unit wrong.

**Full-run result (747 seqs / 5,007 units, v12 config: v10 + early-prefix +
streak gating, 2026-07-08): aligned-hit 90.5% raw / 89.3% after assembly; twins
63.6% blind -> 79.3% with context+assembly; unit SER 11.3%; ayah-chain SER 12.4%;
exact 4-ayah sequences 58.8%.** (v10 baseline for comparison: SER 13.2 / hit 87.3 /
ayah 13.3 / exact 52.6 — early-prefix pays for the streak gating and then some.)
The methodology's central claims are all measured: segments are the right unit,
twins dominate the miss mass, sequential context resolves them, and window scale
must be matched to ref length. Cumulative across the whole arc: aligned-hit
78.5 -> 89.3, exact seqs 32.3 -> 58.8, ayah-chain SER 22.2 -> 12.4.

**Determinism note (2026-07-07):** `build_ngram_index` used to store sets; set
iteration order is hash-randomized per process, so Counter tie-breaks — exactly the
twin cases — varied run to run (~±0.5 SER, ±5 pts twins). The index now stores
sorted tuples; two independent processes produce byte-identical results. All numbers
above are from the deterministic version.

## Assembly layer (deferral confirmation) — in continuous_eval.py

Ports the HighlightController's core rule to unit chains: **expected successors confirm
immediately; unexpected jumps defer until the next emission supports them** (successor
or same-parent forward), else they're dropped as interlopers — the twin-error signature.
Backward/repeat emissions within a parent are dropped as window re-fires.

Full-run effect (747 seqs, ablation arm on identical decodes, v10 config,
2026-07-07): unit SER 26.5% -> **13.2%**, ayah-chain SER 33.1% -> **13.3%**,
exact sequences 29.7% -> **52.6%**; retention cost -1.2 aligned-hit (88.5 -> 87.3).
Insertions are gone — the SER sits at the hit-miss floor.

**Retention fix — 2-deep pending buffer (junk tolerance 1).** The original 1-deep
deferral cost 2.6 pts aligned-hit (128 lost hits). The drop-fate diagnostic
attributed ~110 of them to ONE mechanism: support was only checked against the
immediately-next emission, so a single junk emission between a true unit and its
supporter killed the true unit — dominated by cold starts (91 lost: true -> junk ->
true cascades where the chain never seeds) plus terminal/interloper variants of the
same sandwich. Fix: hold up to TWO pending emissions; an arrival that supports the
OLDER pending retro-confirms it and discards the junk between. Junk still needs
support to enter the chain, so insertion control is preserved by construction:
SER went DOWN (14.5 -> 13.2) while recovering hits (85.9 -> 87.3), and the ayah
chain improved to 13.3%.

**Pipeline (each layer measured + ablated):** multi-scale matched-filter windows
(each scale gated to its own ref-length band, 0.2-2.2 x 10 s) + 3-gram retrieval
(raw-count + full-counter length-normalized shortlist) + infix scoring + blended
selection -> successor votes + twin substitution -> deferral assembly (2-deep
pending buffer). Remaining window-D losses (11%): munch overshoot in the 12-25 band
and exact twins (downstream-resolvable); residual retention cost 1.2 pts.

**C++ port — done (2026-07-07).** `sdk/core/src/chain.{h,cpp}` (`Mode::Chain` in the
Detector: one rolling-buffer decode per hop, scale windows sliced by time). Pinned by
conformance (`golden/chain/`, spec.md §Stage 2b, exact match) AND cross-validated
EXACT (emitted + assembled) over 200 real decoded test streams. Audio-level smoke
(119 s, 2:30-2:33 held-out reciter): C++ Detector parent chain == offline Python
reference on the same audio. `assemble`/`make_succ_full` now live in chain_sliding.py
(module-level = the citable reference).

## On-device findings (2026-07-08, live phone sessions — Samsung foldable)

Debugged from per-hop engine logs + pulled session WAVs, each reproduced offline:

- **The fire threshold must track decode quality.** Consumer phone-mic decodes run
  ~30% PER (vs ~10% for dataset audio; verified on a pulled 53 s session of 2:6-2:9 —
  both best_s123 AND best_mic decode it equally poorly, so this is the acoustic gap,
  not an s123 fine-tune regression). True units then cost 0.35-0.45 against the 0.30
  reference threshold -> almost no fires ("no detection at all"). Sweep on the real
  session: **0.45 recovers the exact true chain** (0.50 tips over — junk disrupts the
  cold start); on CLEAN audio 0.45 regresses (ayah SER 13 -> 22) — so 0.30 stays the
  clean reference and 0.45 ships as the phone config (`window_best(fire_cost=)` /
  `windowBest(fireCost)` / Kotlin `Config.chainCost`). The vote + 2-deep assembly
  layers are what make the loose phone threshold safe (v6-era junk explosion absorbed).
- **v11 — context-gated EARLY-PREFIX firing** (`decode_sliding(early_prefix=0.5)`,
  `_prefix_norm`): whole-unit matching cannot fire until a unit is ~complete (the old
  Auto stack's partial-prefix scoring is what made it feel fast). Now: each
  largest-scale window checks whether the decode TAIL matches >= 50% of the EXPECTED
  unit's prefix -> fires it early. Only ever fires the unit context predicts (low
  risk). **Improves accuracy too, not just latency** (clean 150-seq quick pass,
  assembly arm: SER 12.9 -> 10.7, aligned-hit 87.5 -> 91.4, exact seqs 45.3 -> 56.7 —
  prefix matches survive decode errors that sink whole-unit matches). Conformance
  fixture `early_prefix_run` pins the C++ path.
- **Cold-start provisional highlight** (detector-level, not research): the assembler
  defers the first detection until a supporter arrives — a 10-20 s dead window on
  short surahs (measured 16.6 s on a live surah-111 session). The pending unit's ayah
  now shows immediately as the provisional ACTIVE highlight (public-snapshot layer;
  the conformance-pinned controller and assembly are untouched); the first real
  confirmation overwrites it. Cold start only — mid-stream pendings stay invisible.
- **Remaining phone-latency wall: decode quality — RESOLVED by the mic retrain.**
  Mid-surah units on the quiet mic fired at 0.41-0.44 (threshold 0.45); best_s123_mic
  (val PER 0.130 -> 0.079, learners 48 -> 66%) puts them comfortably under. User
  verdict on-device: "tracking is fast now."
- **v12 — streak protection + trusted-expectation gate (2026-07-08, from a live wrong
  jump).** Observed: 2:255->2:257 streak, junk fire 2:275#05, then EARLY 2:275#06 at
  0.29 -> COMMIT 2:275 (jump). Root cause is an early-prefix amplification loop: after
  ANY junk emission, `expected` is the junk's successor and the prefix probe hunts for
  it EVERY hop at the loose phone threshold — on noisy decode it eventually
  pseudo-matches, manufacturing the assembler's supporter. Two changes: (1) early
  prefix requires a TRUSTED expectation (voter streak >= 1 — the last commit extended
  the chain); (2) once streak >= 3, non-NEAR jumps (near = same surah, 0..2 ayat
  ahead, so post-miss recovery stays cheap) need votes_jump+1 and strong fires no
  longer commit alone. Session replay: the exact audio now yields the clean
  2:255 -> 2:256 -> 2:257 -> 2:258 chain. Conformance golden unchanged (synthetic
  fixtures have no streak-triggering junk); C++ exact. Full 747 with v11+v12: unit
  SER 13.2 -> 11.3, aligned-hit 87.3 -> 89.3, ayah-chain SER 13.3 -> 12.4, exact
  52.6 -> 58.8 — the gating costs nothing on clean audio and kills the live
  wrong-jump class.
- **Posterior-aware matching Phase 0+1 (2026-07-09) — retrieval is no longer the
  bottleneck (near-neutral result).** Phase 0: the decode caches now carry per-phoneme
  top-k posterior alternatives (`greedy_with_alts`; both full + unseg caches rebuilt) —
  the enabler that routes the model's uncertainty across the greedy decode->match
  boundary (`lp.argmax` used to throw it away). Phase 1: posterior-aware RETRIEVAL
  (`window_counts` / `decode_sliding(retr_conf=)`) expands to the top-2 alternatives at
  low-confidence window positions (greedy prob < retr_conf), dedup per position so it
  adds recall not multiplicity. **Finding: the plan targeted the <12-ph retrieval floor,
  but the v10 length-normalized shortlist union had already saturated it** — a diagnostic
  on the 15 short truth units shows greedy already surfaces 100% in the shortlist.
  Result: clean 747 a WASH (greedy -> posterior: SER 11.3 -> 11.2, aligned-hit 89.3
  unchanged, exact 58.8 -> 58.1, twins 76 -> 81 — a small twin gain offset by a small
  exact dip; the 250-seq subset was byte-identical); phone-mic (~30% PER, 9 pulled
  sessions) net +1 correct unit (2:258 recovered in one Ayat-al-Kursi session), no
  regressions. So retrieval has little headroom; the remaining decode-quality losses are
  in SCORING/selection (the ref is retrieved but doesn't win).
- **Posterior-aware matching Phase 2 (2026-07-10) — soft SCORING helps in the deployment
  regime, free on the benchmark.** `_infix_norm(win_alts, sub_min)`: a substitution of ref
  phoneme `ri` for a mismatching window phoneme costs `max(sub_min, 1 - p(ri)/p(greedy))`
  when the model CONSIDERED `ri` (in top-k), else a full 1 — cheap where the model nearly
  picked `ri`, full where it was confidently different. Measured on a NOISE-AUGMENTED eval
  (phone-channel augmentation -> ~30% PER via `build_noisy_cache.py`, `continuous_eval
  --noisy`), the deployment regime the clean caches (~10% PER) don't represent. Crux
  diagnostic: at noisy substitution sites, the correct phoneme is in the top-k 60.9% of the
  time (top-2 47.1%) — recoverable signal exists — but 42% of errors are DELETIONS (a
  different mode, untouchable by substitution softening). Results:
  - **Clean 747: byte-neutral** at any sub_min (SER 10.5, hit 90.1, exact 58.0 on the
    150-subset, both greedy and sub_min 0.0) — no regression.
  - **Noisy 747: sub_min 0.0 (aggressive floor) SER 16.6 -> 14.8, aligned-hit 84.0 -> 85.7,
    exact 48.9 -> 52.6, twins 77.4 -> 78.3.** sub_min 0.3 was neutral (too conservative).
  Mechanism (micro-test): soft scoring lowers the truth AND its competitors' costs, so it
  only FLIPS a selection when the truth has more recoverable subs than the wrong
  competitors — which happens on noisy audio (its errors are uncertain near-misses) but not
  on clean (competitors already far). **Net posterior-aware-matching verdict: retrieval
  (Phase 1) is saturated; SCORING (Phase 2, sub_min~0) is a real ~+1.7 aligned-hit / -1.8
  SER win in the ~30% PER phone regime with zero clean cost.** Phase 0 is the enabler;
  Phase 1/2 stay off by default (greedy) and switch on for the phone regime
  (`retr_conf`, `sub_min`).
- **Phase 2 C++ port — done (2026-07-10).** Routed posteriors inference->chain:
  `decoder::topKAlts` (per-emitted-phoneme top-k), `PhonAlts` through `windowBest` /
  `decodeStream` / `Detector::stepChain`, `infixNormSoft`, `Config/ChainParams::subMin`
  wired C++ -> JNI -> Kotlin `Config.chainSubMin` (demo sets 0.0). Conformance ALL PASS
  (new `soft_score_run` fixture pins the soft path; greedy goldens byte-identical) +
  cross-validated EXACT over 200 real NOISY decoded streams. Caught a latent determinism
  bug from the Phase-1 refactor: `window_counts` routed retrieval hits through a set
  (hash-randomized) — replaced with an ordered dedup so greedy stays byte-deterministic.

## Rolling-window CROWDING of short units — user-reported, two wrong diagnoses, then measured (2026-07-11)

**User report:** reciting a run of short surahs (al-Ikhlas 112 -> al-Falaq 113 -> an-Nas 114)
on-device, the first ayat track but tracing STALLS partway and drops the tail (114:5-6 never
detected). Root-caused from two pulled session WAVs (one continuous, one ayah-by-ayah with
pauses; `best_s123_mic_clean` int8, on-device `chainCost=0.45`).

**Two diagnoses were WRONG before the right one — logged so the mistake isn't repeated:**
1. **"v11/v12 logic regression"** (from a stale June-30 `test_detector.exe` getting 15/15 vs
   current 11) — FALSE. The June-30 binary conflated a different decode (int8-vs-fp32, and its
   own older rolling logic); toggling early-prefix (v11) and soft-scoring (Phase-2) on the SAME
   stream changed nothing, and disabling early-prefix was *worse*. Not the voter/gating.
2. **"decode-quality-limited / 114:6 undecodable"** (the full-utterance decode never produced
   114:6's phonemes; window_best proposed 114:4 only at cost 0.56 > 0.45, and never proposed
   114:5-6) — ALSO FALSE as a ceiling. It was an artifact of matching over the WIDE window: a
   focused decode of the tail region alone (`[50-66s]`) retrieves 114:5, and the fix below
   retrieves 114:6 too.

**Actual root cause: the 22 s rolling window (largest filter-bank scale 2.2 x 10 s) CROWDS OUT
short units.** After several commits the window is dominated by earlier/again-decoded content;
short tail units are a small fraction of it, so `window_best` never surfaces them (retrieval,
not scoring). Reproduces in the batch reference `decode_sliding` on the user's stream too — so
it is the shared matching regime, not the incremental `chainMatch` path.

**Why the benchmarks missed it (measurement gap — the important lesson):** `continuous_eval.py`
runs over CACHED professional-audio phoneme streams (~10% PER) and never re-windows audio; the
conformance `golden/chain` fixtures are short SYNTHETIC streams. Neither models the rolling
audio buffer, so neither can exhibit window-crowding. Confirmed: the harness aligned-hit does
NOT fall with length (run-len 4/8/12 -> 87.5 / 92.4 / 91.7%). Crowding needs BOTH poor phone
decode AND the wide window — only reproducible on real phone WAVs through the full Detector.

**Fix prototyped + measured — focused-window / VAD reset for Chain mode (`Config.chainVadReset`,
off by default).** On a Silero speech-END the Detector drops the buffered ayah's audio+phonemes
(so the next ayah decodes in a focused window) while KEEPING the voter/assembler chain context
(expected/streak/emitted survive the pause). `test_detector` env hooks `QR_COST` / `QR_VAD`.
Measured (n=2 real phone WAVs, cost 0.45):

| WAV | current (no reset) | chainVadReset | truth |
|---|---|---|---|
| paused (ayah-by-ayah) | 11/15 (miss 113:1-2, 114:5-6) | **15/15 exact** | 15 |
| continuous | 11/15 (stalls at 114:2) | **15/15 exact** | 15 |

Both recover the exact truth chain, no junk insertions. **Caveats / not yet shipped:** (a) n=2,
same reciter, short surahs only; (b) the harness cannot measure this (no audio/VAD), so broad
validation needs more real recitation on-device — ESPECIALLY LONG ayat (Baqarah), where a false
mid-ayah VAD boundary would reset mid-ayah and cost the ayah's prefix (the v1 commit-and-reset
cascade / seg>=2 mid-ayah cold-start risk that made Chain go pause-tolerant in the first place);
(c) off by default, not wired to JNI/demo — the on-device long-ayah A/B is the gate before it ships.

### audio_bench.py — audio-level regression harness; VAD-reset verdict = TRADE, not a fix (2026-07-11)

The n=2 result above was **too optimistic** — corrected by an offline harness built to replace
"validate live through the user." `continuous_eval.py` runs on cached PHONEME streams so it cannot
see rolling-buffer/VAD/decode-quality effects; **`research/audio_bench.py` drives real audio through
the full C++ Detector (`test_detector.exe`) and scores emitted vs truth.** Two corpora, both offline:
(A) COMPOSED held-out test-reciter streams (concatenated ayah clips, ± inter-ayah gap ± phone-channel
augmentation) — the always-available REGRESSION net; (B) REAL pulled phone sessions with hand-labeled
truth (`data/raw/audio_bench/real/`, gitignored) — the FAILURE-REGIME net (add each new pulled WAV).

Full run (windowed, cost 0.45), baseline vs `chainVadReset`:

| case | truth | baseline | +chainVadReset |
|---|---|---|---|
| short_112_114_cont (pro-aug) | 15 | 15/15 exact | 13/15 |
| short_112_114_paused (pro-aug) | 15 | 14/15 | 12/15 |
| short_105_108_paused (pro-aug) | 19 | 18/19 | 15/19 |
| long_baqarah_1_5 (pro-aug) | 5 | 3/5 | **1/5** |
| long_baqarah_253_257 (pro-aug) | 5 | 5/5 | 4/5 |
| real_112_114_paused (phone) | 15 | 11/15 (tail 0/2) | **15/15 exact** |
| real_112_114_cont (phone) | 15 | 11/15 (tail 0/2) | **15/15 exact** |

**Verdict: `chainVadReset` is a TRADE, not a fix.** It rescues the real-phone short-surah crowding
(+4 units, tail recovered on both) but REGRESSES every clean case and DESTROYS long ayat (Baqarah
1:5 3/5 -> 1/5 — the mid-ayah VAD-boundary risk, now measured not just feared). Do NOT ship as a
blanket setting. Two further facts the harness surfaced: (1) **professional reciters don't reproduce
the crowding at all** (baseline already 14-15/15) — it needs real phone-mic decode quality, which is
why corpus (B) is essential; (2) **the de-crowding is WINDOWED-ONLY** — streaming decodes
incrementally, so a boundary reset can't recover crowded tail phonemes: it clears the accumulated
phoneme stream leaving too few to match the tail (measured no gain + slight clean harm; the earlier
9/15 divergence was a broken outBase time axis, since fixed). `chainVadReset` is now a safe NO-OP in
streaming (VAD gated to windowed chain); the harness runs windowed.

**Targeted gate — the fix (2026-07-11).** The blunt reset's damage is mid-LONG-ayah resets (a
breath pause reset loses the ayah prefix). Discriminator: a short ayah COMMITS then pauses (small
gap between last commit and the pause); a long ayah's mid-breath comes many seconds after the last
commit. So `Config.chainResetMaxGap` — reset ONLY if `(pause_time - last_commit_time) <= gap`.
Also suppress the reset before the FIRST commit (initial `lastConfirmSec=-1e9`) — an early bad
reset alone took Baqarah 1:5 from 3->1. `test_detector` env `QR_RESET_GAP`. Full harness sweep:

| case | baseline | blunt (1e9) | gap 3.0 | **gap 4.0** |
|---|---|---|---|---|
| short_112_114_cont (pro) | 15/15 | 13 | 14 | 14 |
| short_112_114_paused (pro) | 14/15 | 12 | 14 | 14 |
| short_105_108_paused (pro) | 18/19 | 15 | 18 | 18 |
| long_baqarah_1_5 (pro) | 3/5 | **1** | 3 | 3 |
| long_baqarah_253_257 (pro) | 5/5 | 4 | 5 | 5 |
| real_112_114_paused (phone) | 11/15 | 15 | 13 | 14 (tail 1/2) |
| real_112_114_cont (phone) | 11/15 | 15 | 13 | 15 exact (tail 2/2) |

**gap=4.0 is the operating point** (the shipped default when the reset is on): recovers the
real-phone crowding (+3 / +4, continuous fully to 15/15 exact) with ZERO long-ayah regression and
a 1-unit clean cost — vs blunt's Baqarah 1/5 collapse. A REAL fix, not a trade. WINDOWED-ONLY
(streaming can't benefit — see above; `chainVadReset` is a safe no-op there). Still defaults OFF
(needs on-device validation); the download build gets the fix only when run windowed (trades the
~11x streaming RTF for accuracy). Run: `python research/audio_bench.py [--only <substr>]` (windowed,
~35 s/case); set `QR_RESET_GAP` to sweep the gate.

## Reassessment / taint audit (2026-07-11) — what rests on cached-stream evidence

The crowding episode proved `continuous_eval.py`'s cached phoneme streams are blind to
rolling-buffer / decode-quality / VAD effects. Auditing every tuned knob for that taint:

| finding | evidence base | status |
|---|---|---|
| waqf unit, twins, context, assembly | clean caches, held up on-device | sound |
| streaming parity + ~11x RTF | byte-validated, conformance-pinned | sound |
| gap=4.0 gated VAD reset | audio_bench end-to-end | sound method, thin corpus (was n=2) |
| `chainCost=0.45` | ONE pulled session, Python layer, pre-harness | **TAINTED — resweep end-to-end** |
| `chainSubMin=0.0` (+1.7 claim) | noise-AUGMENTED caches — augmentation shown NOT to reproduce the real regime; always-on in the harness, never ablated on real audio | **TAINTED — ablate end-to-end** |
| v12 headline (89.3 hit / 58.8 exact) | clean caches; real-WAV baseline is ~73% | **overstates deployment** — audio_bench real corpus is the honest number |
| `build_waveform_augment` realism | pro-aug composed streams do NOT crowd; real phone WAVs do | **training-side taint** — the mic retrain may also be under-served |

**Ranked directions (added to roadmap):**
- **A. Real-corpus revalidation (in progress)** — corpus B grown from 2 → 34 real WAVs
  (7 labeled `demo/test_fixtures` user recordings + 25 rescued pulled sessions, semi-auto
  labeled via `research/label_sessions.py` → `real/labels_proposed.csv` → user-confirmed
  `labels.csv`); `audio_bench.py` now takes named ARMS (`--arms base,vad,stream,streamvad,
  cost035..cost050,hardsub,noearly,vad_g3`; env hooks `QR_SUBMIN`/`QR_EARLY` added to
  test_detector). Re-validate every tainted knob end-to-end on real audio.
- **B. Augmentation-realism gap** — diagnose WHY real phone audio decodes at ~30% PER when
  augmented professional audio doesn't reproduce the failure (phone DSP: AGC / noise
  suppression / spectral tilt?). Payoff is double: a faithful synthetic failure-regime corpus
  (unlimited harness data) + a retrain lever (42% of noisy errors are deletions only a better
  decode fixes).
- **C. Streaming-side de-crowding** — the deployment default streams and gets no crowding fix;
  design work gated on A's data (how big is the streaming↔windowed gap across the corpus?).

**A first results (2026-07-11, corpus = 5 composed + 2 real + 7 fixtures, 109 truth units):**

| arm | total | notes |
|---|---|---|
| base (windowed) | 88/109 (81%) | |
| **+chainVadReset gap=4.0** | **94/109 (86%)** | best arm; all gains real-phone, 1-unit composed cost |
| stream (deployment default) | 89/109 (82%) | streamvad == stream everywhere (no-op gate verified) |

- **Streaming ≠ windowed on hard audio.** The old "identical detections" held only on the two
  clean validation clips. Streaming WINS long quiet-mic cases (fix_78_38_40_cont 2/3 vs 0/3,
  baqarah_1_5 4/5 vs 3/5) and LOSES slightly on short runs (105-108 17/19 vs 18/19,
  253-257 4/5 vs 5/5). Different decode regimes, different failure modes — neither dominates.
- **fix_98_1_3_paused 0/3 in EVERY arm — cold-start crowding, a class the gated reset cannot
  touch.** Focused-window truth costs (current model): 98:1 0.56 / 98:2 0.33 / 98:3 0.16 — two
  of three are comfortably under the 0.45 threshold IF the window were focused. But the long
  quiet 98:1 never fires -> nothing commits -> the commit-gated VAD reset never activates ->
  the window never focuses. A matcher-state-aware reset (allow a pre-first-commit reset when
  no pending unit is mid-match) is the candidate mechanism — direction C2.
- **Model-generation regression found (quiet-mic regime):** `best_s123_mic_clean` (deployed) is
  consistently WORSE than `best_s123_mic` on the quiet-mic fixture's focused decodes
  (98:1/2/3 truth costs 0.56/0.33/0.16 vs 0.50/0.26/0.11). The clean retrain was selected on
  the cleaned learner test (+1.9) with no quiet-phone check — the `mic` bench arm now measures
  this end-to-end.
- **Harness scoring now includes the trailing PROVISIONAL** (the cold-start active highlight the
  user sees; test_detector prints `provisional:`) — confirmed-only scoring under-reported what
  the on-device experience shows on short clips.
- **Corpus B grown 2 -> 9 labeled real WAVs + 7 confident session labels** (labels.csv; 8 more
  sessions await user ear — labels_proposed.csv carries decode-screen identifications: 105:1-2,
  113:1-2, 113:3-5, 98:1, 2:255 x2, 2:285 all MISSED entirely by the detector at cost 0.50,
  mostly short single-ayah clips = cold-start deferral, visible only as provisional).

**A final verdicts (2026-07-11, full sweep + combos, 21 cases / 138 truth units, provisional-
inclusive scoring; anchor = deployed config `mic_clean` + cost 0.45 + subMin 0 + early-prefix,
windowed = 118/138 86%):**

| knob | verdict | evidence |
|---|---|---|
| `chainCost 0.45` | **SURVIVES** | 0.50 alone +4 (122) but ANTI-stacks with vadReset (mic050vad 124 < micvad 129); 0.40 neutral; 0.35 collapses (106) |
| `chainSubMin 0.0` (Phase-2 soft) | **NEUTRAL end-to-end** | hardsub 119 ≈ base 118 — the noisy-cache +1.7 did NOT transfer to real audio; keep off-by-default posture, no end-to-end case for it |
| early-prefix (v11) | **SURVIVES** | noearly 115 < base 118 |
| model `best_s123_mic_clean` | **OVERTURNED** | `best_s123_mic` arm 125 (91%) vs 118 (86%), better or equal on every hard case (fix_98_1_4 2/4->4/4, real_cont 11->13, Kursi session 2/3->3/3) — the clean retrain was selected on the cleaned learner test (+1.9) and never checked end-to-end |
| `chainVadReset` gap 4.0 | **CONFIRMED, stronger than first measured** | vad 126 (91%) over base 118; windowed-only as before |
| **winning combo** | **`best_s123_mic` + 0.45 + vadReset = 129/138 (93%)** | +11 units over deployed; cost 0.50 must NOT ride along |

Caveats: corpus B is one user's voice/devices (that IS the beta deployment target, but the
diverse-learner slice is the cleaned RetaSy test where mic_clean is +1.9 — model choice is a
regime trade the user must arbitrate); streaming graphs are exported from mic_clean, so a model
revert also means re-exporting `stream_{conv,encoder}` for the streaming path.

## Repetition suppression — the decode-collapse ROOT CAUSE found (2026-07-11 pm)

Live report post-deployment ("tracking is very bad"): two fresh takes (112 paused-ish, 114
continuous; same phone/minutes as a third 113 take that tracked PERFECTLY) confirm NOTHING in
ANY config — old/new model, windowed/streaming, reset on/off all identical. Not a regression;
a failure class. Chased to the bottom with three probes (all reproducible offline):

1. **Focused 5 s slices decode every ayah cleanly under the 0.45 threshold** (112: 0.00/0.21/
   0.11/0.36; 114: 0.14/0.09/0.27/0.08) while the wide-window decode emits only 36-46 phonemes
   for the whole take (whole short ayat deleted). Not retrieval — the DECODE collapses.
2. **No length effect**: the [0-5 s] region decodes IDENTICALLY (19 ph) whether the input is
   5/10/15/20 s. The collapse is not attention diffusion over long input.
3. **Context-replacement is decisive**: region [5-10 s] («maliki n-naas») decodes to 16 ph
   standalone, 11-13 ph with zeroed/noise context, but **5 ph («...n-naas» only) with the TRUE
   preceding audio** — which ends in the SAME phrase («rabbi n-naas») — in the Emformer memory.

**The model's left-context memory suppresses repeated phrases.** Surahs 112/113/114 are
maximally repetitive; a flat/quiet continuous recitation makes consecutive ayat acoustically
near-identical -> the encoder deletes the repeats. The model was **trained on single-ayah clips
only** — continuous multi-ayah audio (let alone cross-ayah phrase repeats) is entirely out of
its training distribution. This RETRO-EXPLAINS the original crowding episode: focused windows
"fixed retrieval" because focusing removes the repeated phrase from the model memory — the
suppression was upstream of retrieval all along. It also explains why streaming can't be fixed
by boundaries (its incremental decode always carries the full left context).

Mitigations, in order of depth:
- **Confident-emission-armed VAD gate (SHIPPED as interim; 3 design iterations, each measured):**
  the reset gate's anchor is now `max(lastConfirmSec, lastEmitSec)` where `lastEmitSec` is set
  only by CONFIDENT emissions (cost <= 0.5 x chainCost). Iteration lessons: (1) blunt lastEmitSec
  (any emission, pre-commit only) let quiet-take junk fires (0.35-0.45) trigger early resets that
  clipped first ayat — bench 129 -> 124; (2) confidence bar fixed the sessions but pre-commit-only
  starved SURAH TRANSITIONS (113:1 emits at 0.00, sits pending, and the unfocused window deletes
  its supporter 113:2 by repetition suppression — gap-from-last-commit was stale); (3) final form
  arms on consumed content ALWAYS. **Bench 133/143 (93%): +4 net vs anchor** (cold-start 112 take
  0 -> exact 112:1-4, short_112_114_cont 13 -> 15 EXACT; real_112_114_cont 15 -> 13 redistribution
  — the one regression, accepted). fix_98_1_3 unchanged (98:1 never emits confidently — needs the
  fresh-suffix window). Conformance ALL PASS (the gate lives in Detector::feed, not chain.cpp).
- **`chainEmitTrimKeep` (experimental, off):** trim the rolling buffer on every emission. Fixed
  112 alone but NOT the continuous 114 take — the suppressor phrase survives any trim that keeps
  the successor's prefix. Kept for ablation.
- **Fresh-context suffix window — v13, MEASURED + WINNING (2026-07-11 pm).** Per hop, ALSO
  decode the rolling buffer's last 5 s STANDALONE through a right-sized graph
  (`model_s123_mic_5s.int8.onnx`, --fixed-frames 516, RTF 0.002) and match over it with a
  restricted two-window bank (full suffix + last 2 s). Fresh Emformer memory sidesteps the
  suppression unconditionally — no pause, no commit, no VAD needed. `Config.chainSuffixSec` +
  `chainSuffixModelPath` (0/empty = off); `QR_SUFFIX`/`QR_SUFFIX_SEC` in test_detector.
  **Bench 145/151 (96%) vs the shipped micvad config 138 (91%):** continuous-114 take 1/4 ->
  4/4 EXACT, both real_112_114 takes EXACT, fix_98 1/3 -> 2/3 at 5 s... iteration lessons:
  (1) **the suffix pass SUBSUMES chainVadReset** (suffix-without-reset 144 > with-reset 143 —
  the reset is net-negative alongside it; v13 ships with vadReset OFF); (2) mid-LONG-ayah
  suffix junk (0.42-0.43 fires) floods the assembler's 2-deep pending buffer and evicts true
  pendings (Baqarah -2) -> **skip the pass while the voter EXPECTS a unit too long to fit the
  suffix's length gate** (len > 6.5 x suffixSec ~ 32 ph; no streak requirement — a lone jump
  emission counts); (3) **7 s suffix REGRESSES overall (140)** — wider window re-admits
  mid-band junk on quiet takes + carries more suppressing context; it did lift fix_98 to its
  2/3 decode ceiling, but 5 s is the operating point (fix_98's medium-unit ceiling = phase-3's
  job). Cost: ~+15% hop decode desktop (405 ms/hop incl. suffix). Streaming variant (parallel
  reset-every-5s stream, possibly staggered x2) is designed but unbuilt — windowed first.
- **Concatenation training, phase 3 (the root fix):** synthesize continuous multi-ayah training
  clips from the existing corpus (consecutive ayat, same reciter, short gaps). Teaches the
  encoder to emit repeats; no new data collection. Validate with the context-replacement probe +
  audio_bench gate.

## Phase-3 concatenation training — the repetition-suppression ROOT FIX (2026-07-11 pm)

Real continuous per-surah recitations (user-sourced, licensing cleared) replaced the synthetic
concatenation plan. **Corpus:** 45.2 h / 360 files / 9 sources / 7 voices
(`data/raw/continuous/`, spec-driven `data/download_continuous.py`); `yasser_ad_dussary` is a
held-out TEST reciter -> quarantined EVAL-ONLY (`_meta.eval_only`). **Labeling:** hierarchical
alignment (`data/align_continuous.py`): chunked whole-file log-probs (28 s windows, settled-
interior stitching — contiguous-by-construction after a seam-hole bug) -> banded edit DP with
free leading/trailing skip (absorbs isti'adha/basmala) -> per-ayah forced-align refinement.
Full sweep: **~9,700 ayah alignments, 8 flagged (99.9%)**, incl. three 2-4.4 h Baqarah files at
100% span. **Windows:** `make_phase3_windows.py` -> 5,344 train (34.6 h) + 697 eval windows
(<=28 s multi-ayah, midpoint-reconciled bounds, flag-excluded); PCM memmap cache
(`extract_continuous_pcm.py`) so the Dataset slices instantly; `AyahDataset` handles mixed
per-ayah + window rows (`make_phase3_manifest.py`).

**Gates + iteration (selection by gates, NEVER val PER — the taint-audit rule):**
- `research/probe_suppression.py` — the mechanistic gate: in-context/alone phoneme ratio on
  regions following repeated phrases (user takes + held-out yasser continuous).
  **baseline best_s123_mic 0.434; p3 0.94 — suppression GONE, including on the held-out voice.**
- **p3** (15 ep fine-tune of best_s123_mic on the mixed manifest, lr 1e-4): probe PASS, learner
  84.0 (best), clean 96.0, val PER 0.072 — but **bench p3suf 141 < anchor 145: GATE FAIL.**
  Losses concentrated in the quiet-mic long-ayah family — 35 h of clean pro windows diluted
  poor-mic robustness. Also: **suffix still +7 on p3 (v13 NOT retired)** and **p3stream 125 —
  streaming's deficit is more than suppression; the battery config did not come back.**
- **p3.1 RESTORE** (5 ep on the pure phase-2 manifest from best_s123_p3, lr 5e-5): suppression
  mostly STICKY (0.876), quiet-mic edge restored. **Bench p31suf 145/151 = anchor tie; learner
  85.3 (best ever); clean 96.2 (best ever); val PER 0.069.** Strictly better than the deployed
  model on every axis except a bench tie -> the ship candidate (`best_s123_p31.pt`).

Lessons: (1) the root fix works exactly as diagnosed — the model deletes repeats only because
single-ayah training never showed it continuous audio; (2) capability mixing needs a restore
pass — new-regime data dilutes old-regime robustness, and a short low-LR polish on the old mix
recovers it while the new capability sticks; (3) v13 remains necessary (decoder-level windows
still pay even with a healthy decode); (4) streaming's hard-audio deficit persists post-fix.

**Post-ship live finding — early-prefix runaway on near-twin successors (fixed same evening).**
On-device v3 test: during surah 113 the highlight ran ~13 s AHEAD of the reciter — ayat 3/4/5
EARLY-fired at 1.5 s intervals on the shared «wa min sharri...» opening (trace: costs
0.40/0.40/0.14). The healthy phase-3 decode UNMASKED this: suppression used to mangle the
repeated prefix, accidentally starving the v11 early-prefix probe. Two guard designs measured:
(a) LCP-raised minimum length — NO effect (the probe is edit-normalized; a 0.45 threshold
tolerates the post-LCP mismatches); (b) full-ref twin-distance margin — NO effect (113's twins
differ in their TAILS; the probe only sees the prefix). The fix: **discrimination margin over
the REGION THE PROBE SEES** — early fire requires `prefixCost <= normEditDist(first-minI
phonemes of expected vs of last-emitted) - 0.15` (`normEditDist` in chain.cpp). Session replay:
113 commits 19.5/21/22.5/24 s -> 19.5/22.5/28.5/33 s (tracking the recitation, which ran to
~37 s), sequence still exact 15/15; bench 145/151 held; conformance ALL PASS; user-verified
live ("113 tracking is fixed"). Distinct successors (prefix distance ~0.7+) early-fire as
before — the guard only defers fires whose evidence cannot tell the next ayah from the current.

## Segment-level ambiguity map (matcher/find_ambiguous.py --units)

Formalizes the twin classes for the production deferral layer:
`data/lang/ambiguous_units.json` over the chain decoder's unit index (1,029 waqf
segments + 712 unsegmented ayat = 1,741 units, tau 0.15, matcher-consistent metric).

- **206 ambiguous units / 84 classes** (11.8% of units vs 2.5% at ayah level —
  segmentation multiplies ambiguity, as the miss dissection predicted); 141 units in
  exact-duplicate classes == EXACTLY the decoder's twin sets (cross-checked equal).
- **Context resolves 198/206 (96%)**: predecessor 22 / successor 23 / both 153 —
  neighbours now follow the unit chain (segment n±1, crossing ayah boundaries within
  the surah). The **8 context-insensitive** cases are structural: 2:134↔2:141 unit
  pairs (near-identical ayat whose successors are also confusable — needs deeper
  N-back), 3:1↔2:1 (both alif-lam-mim, surah openers), 99:8↔99:7 (the known ayah-level
  case). These are the `needs_choice` fallback set.
- **All 206 are cross-parent** (`cross_parent` flag): every unit confusion changes the
  highlighted ayah — no harmless within-ayah twins exist in this corpus.
- **Negative result — near-twin substitution:** extending the decoder's twin
  substitution from exact-ref equality to map confusables (`decode_sliding(confusable=)`)
  measured NEUTRAL on the full 747 run (SER 14.5% = 14.5%; quick-pass +0.4 was noise).
  Exact twins already capture the resolvable mass; the 65 near-twin units are too rare.
  The map's value is the deferral/highlight contract, not decoder accuracy.

## salat_eval.py / salat_probe.py — encoder reuse for prayer-state detection

Feasibility probes for a NEW standalone feature (2026-07-12): hands-free **salah (prayer)
state detection**, solo worshipper on a stand — started as a "wake word" idea, pivoted to
full prayer tracking as the main use case. Reintegrates into the recite app later. See the
project memory `project-salat-state-detection`.

Domain fact that shapes everything: only **three prayer phrases are audible** — takbir
("Allahu akbar"), sami'allah ("sami'allahu liman hamidah"), salam. The ruku/sujud tasbih and
tashahhud are said silently, so postures during them are INFERRED from the marker sequence +
timing, never detected. **sami'allah is a unique per-rakah anchor** (fires once, rising from
ruku) → self-correcting sync that breaks the identical-takbir counting ambiguity.

Central question: does the recitation-trained encoder transfer to these phrases with NO
retraining? Probes reuse the shared front-end + encoder (the whole point — reuse the
foundation, not the ayah chain matcher):
- **`salat_probe.py`** — raw per-phrase decode inspector: energy-segment → greedy phonemes →
  infix-norm edit distance to each phrase's G2P reference. Shows decode quality by ear-proxy.
- **`salat_eval.py`** — the scored eval (path a, the no-retrain ceiling): **Silero VAD** →
  greedy phonemes → **fuzzy-onset + duration 3-marker discriminator** → edit-distance
  alignment vs a known ground-truth marker sequence. `--audio`/`--ckpt`/`--truth` overridable.

**Result** (`data/salat_phrases.m4a`, 14 utterances, one voice, order
takbir×5·sami'allah×4·takbir×1·salam×4): **11/14, 0 confusions, 0 false markers** — every
error a safe "miss," never a wrong marker. sami'allah **4/4**, salam **4/4** (after principled
`q≈k` / `S≈s` cue tuning — systematic model-confusion classes, not per-clip overfit), takbir
**3/6**: half the takbirs decode with no recognizable "allahu" because **"akbar" never
decodes**, so takbir is detected by ABSENCE of the other cues, not a positive signature. Same
on best_s123_p31 and best_s123_mic → a real transfer property, not a model quirk. Silero
segmented the clip 14/14 = one segment per utterance (energy VAD over-split the long salam).

**Verdict: moderate engineering project, not research.** MVP runs on the CURRENT model, no
retraining — the sami'allah anchor + salam end are 100% cold, and missed takbirs are
recoverable via the deterministic takbir count between anchors (zero false transitions to
corrupt the state machine). **Path (b)** — a light fine-tune giving takbir a positive "akbar"
signature (~50–100 self-recorded takbirs+salams) — is the clear next accuracy lever but
deferrable. **Caveats:** N=14, one voice, one clip — a signal, not a validated accuracy; cue
tuning unproven on new audio. Next step: record a few full real prayers, re-run
`salat_eval.py --audio … --truth …` to confirm 11/14 holds, THEN build the `Mode::Salah`
module (Silero VAD → 3-marker discriminator → anchored state machine) as its own module
consuming the shared encoder.
