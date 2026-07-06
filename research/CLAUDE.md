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

## chain_sliding.py — chained segment decoding (v8, working)

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

**Full-run result (747 seqs / 5,007 units, 2026-07-07): aligned-hit 81.9%; twins
33.1% blind -> 68.1% with context (+35.0 — blind is a coin flip among identical-ref
twins; context resolves >2/3); unit SER 28.7%; exact sequences 20.1%; smoothed
ayah-chain SER 36.5%.** The methodology's central claims are now all measured:
segments are the right unit, twins dominate the miss mass, sequential context
resolves them.

Remaining (assembly-layer, not detection): insertion control (unit SER ~32% vs ~15%
hit-misses) and the parent/ayah chain derivation (naive smoothing gives ~50% ayah SER
— needs the production HighlightController deferral logic, not island-dropping).
