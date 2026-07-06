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
