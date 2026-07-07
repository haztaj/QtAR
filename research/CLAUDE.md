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

**Full-run result (747 seqs / 5,007 units, v10 config + deterministic index,
2026-07-07): aligned-hit 88.5% raw / 85.9% after assembly; twins 40.0% blind ->
76.0% with context+assembly (blind is a coin flip among identical-ref twins); unit
SER 14.5%; ayah-chain SER 14.8%; exact 4-ayah sequences 50.3%.** The methodology's
central claims are all measured: segments are the right unit, twins dominate the
miss mass, sequential context resolves them, and window scale must be matched to
ref length (filter bank + short-ref recovery: aligned-hit 78.5 -> 85.9, exact seqs
32.3 -> 50.3, ayah-chain SER 22.2 -> 14.8 across the two changes).

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

Full-run effect (747 seqs, third ablation arm on identical decodes, v10 config,
2026-07-07): unit SER 27.7% -> **14.3%**, ayah-chain SER 35.3% -> **14.3%**,
exact sequences 29.7% -> **50.9%**; retention cost -2.5 aligned-hit (88.6 -> 86.1).
Insertions are gone — the SER sits at the hit-miss floor.

**Pipeline (each layer measured + ablated):** multi-scale matched-filter windows
(each scale gated to its own ref-length band, 0.2-2.2 x 10 s) + 3-gram retrieval
(raw-count + full-counter length-normalized shortlist) + infix scoring + blended
selection -> successor votes + twin substitution -> deferral assembly. Remaining
window-D losses (11%): munch overshoot in the 12-25 band and exact twins (downstream-
resolvable); next levers: assembly retention (2.5-pt aligned-hit cost), then the
C++ port of the winning design.

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
