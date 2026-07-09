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
  regressions. So retrieval has little headroom; the
  remaining decode-quality losses are in SCORING/selection (the ref is retrieved but
  doesn't win) — **posterior-aware SCORING (Phase 2, soft substitution cost in
  `_infix_norm`) is where the headroom is.** Phase 0 stays (it's Phase 2's enabler and
  costs nothing); Phase 1 stays off by default (`retr_conf=None`), available for the
  phone regime.

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
