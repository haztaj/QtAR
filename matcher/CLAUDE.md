# matcher/ — Stage-2 incremental fuzzy matcher

Disambiguates which ayah is being recited from a stream of phonemes. Pure-CPU,
no model dependency — it consumes phoneme tokens (ground-truth G2P for testing,
or Stage-1's greedy/posterior output at runtime).

## `phoneme_matcher.py`

- `PhonemeTrie` — array-backed trie over all 564 ayat in phoneme space. Shared
  prefixes merge, so prefix-overlapping ayat surface as one candidate set and
  separate only as recitation diverges. 564 ayat → ~12.2k nodes.
- `PhonemeMatcher` — streaming approximate string match. Frontier maps trie-node →
  best edit cost aligning observed phonemes so far to the path reaching that node.
  Per input phoneme: insertion (stay), substitution (advance, consume), deletion
  (advance, no consume; bounded epsilon-closure of `max_del`). Beam-pruned to
  `beam_width`. A global entry beam (root re-seeded to cost 0 each step) enables
  restart/jump detection (`allow_restart`).
- `candidates(k)` — ranked (key, cost, norm_cost) from terminal nodes in the frontier,
  sorted by `cost / ref_len`. `commit_margin()` returns top candidate + the norm-cost
  gap to the runner-up (commit when the gap clears a threshold).

## Validated (self-test on ground-truth phonemes)

```
python matcher/phoneme_matcher.py
```

- Exact-match top-1: **99.1%** (564 ayat). The ~5 misses are ayat with *identical*
  phoneme strings — indistinguishable by phonemes alone (repeated refrains), a real
  property, not a bug. These are the exact-duplicate classes in `ambiguous_ayat.json`
  (`82:13↔83:22`, `83:9↔83:20`, `83:23↔83:35`, `84:2↔84:5`, `109:3↔109:5`) and are all
  resolved at the highlight layer by sequential context / deferral (see
  `highlight_controller.py`) — the matcher isn't expected to break these ties alone.
- Learner-error robustness: 10% err → top1 98.7% / top3 100%; 20% → 96/97; 30% → 92/93.
- Early detection: true ayah in top-3 after **69%** of phonemes on average.

## `find_ambiguous.py` — confusable-ayah map (corpus-agnostic)

Precomputes which ayat are too phoneme-similar to tell apart on their own, and whether
sequential context can break each tie. Runs on the Juz Amma lexicon now and the full-Quran
lexicon later (length-pruned pairwise; `rapidfuzz` if installed, DP fallback otherwise).

```bash
python matcher/find_ambiguous.py            # -> data/lang/ambiguous_ayat.json
python matcher/find_ambiguous.py --tau 0.12 --lexicon <full-quran.json> --out <path>
python matcher/find_ambiguous.py --units    # segment-level unit index (waqf segments +
                                            # unsegmented ayat) -> data/lang/ambiguous_units.json
```

Metric = the matcher's own normalized Levenshtein (1/1/1), so "ambiguous" == what the runtime
actually confuses. Per ambiguous ayah it emits `confusable_with` (the candidate set) and
`resolvable_by`: **predecessor** (prior ayah pins it), **successor** (WAIT for the next ayah,
then it pins it — retroactive), **both**, or **none** (context can't break the tie → needs an
option fallback). This is the input to the deferral + centralized-highlight logic.

Juz Amma @ tau 0.15: **26 ambiguous ayat / 13 pairs** (10 exact-duplicate). All resolvable by
context except **99:8↔99:7** — consecutive *and* 99:8 ends the surah, so no neighbour helps.
(Still the ONLY ayah-level `none` case on the expanded 1,057-ayah corpus.)

**Unit mode (`--units`, 2026-07-07):** runs over the chain decoder's unit index (1,029 waqf
segments + unsegmented ayat = 1,741 units); neighbours follow the unit chain (segment n±1,
crossing ayah boundaries within the surah) and each ambiguous unit gets a `cross_parent`
flag (does the confusion change the highlighted ayah?). Result @ tau 0.15: **206 ambiguous
units / 84 classes** (141 in exact-duplicate classes — verified equal to the decoder's twin
sets), 96% context-resolvable (pred 22 / succ 23 / both 153), **8 `none`** (2:134↔2:141 unit
pairs — successors also confusable, needs deeper N-back; 3:1↔2:1 alif-lām-mīm; 99:8↔99:7);
all 206 cross-parent. Near-twin (non-exact) substitution in the research decoder measured
neutral end-to-end — the map's consumer is the deferral/highlight layer, not the matcher.

## `highlight_controller.py` — centralized highlight state machine (Stage 3)

Sits ON TOP of the commit layer: consumes *committed* detections and emits render-ready
`HighlightState` snapshots so **every platform/SDK version just draws the snapshot** — the
deferral + ambiguity handling live here once, not re-coded per UI. This is the SDK's public
output contract (state snapshots, signed off 2026-07-03). Ported to `sdk/core/src/highlight.*`
and conformance-pinned (`golden/highlight/`, C++ byte-identical to this reference).

Snapshot: `confirmed[]` (settled, highlighted) · `pending{ayah, options[], reason}` (deferred)
· `active` (emphasize now). On an **ambiguous** detection (from `ambiguous_ayat.json`) it does
NOT guess: predecessor pins it → confirm now; else successors distinct → hold
`await_successor` (**`active` stays put — no highlight**) and the next ayah retro-confirms it;
else `needs_choice` → surface `options` for a manual `choose(key)`. See the finder's
`resolvable_by`. Self-test: `python matcher/highlight_controller.py` (drives all paths).

## Commit policy (`CommitTracker`) — tuned

Committing on a single margin crossing is unreliable: early in a recitation a
transient wrong candidate can briefly lead with a large margin (gives ~27% commit
accuracy). `CommitTracker` requires the SAME top-1 to hold `margin >= threshold`
for `persistence` consecutive phonemes. **Persistence (K) is the dominant lever,
not the threshold.** Sweep via `eval/tune_commit.py`:

| | clean test | held-out learners |
| K=1 (naive) | 27% acc | 28% acc |
| K=3 | 64% | 59% |
| **K=5 (default)** | **86% acc, 14% false, ~84% latency** | 70% acc, 30% false |
| K=8 | 95% acc, 5% false | 90% acc but commits <6% of clips |

Tradeoff is accuracy/coverage/latency. Defaults T=0.15, K=5. Raise K for
safer/rarer commits, lower for earlier/looser. **The learner false-commit (30% at
K=5) is limited by acoustic-model quality on learners, not the policy** — more
learner data (long surahs) is the real fix, not a higher K (which just stops
committing). `eval/evaluate.py --commit-margin --persistence` reports it.

## Sequential context + revisable commit (`SequentialContext`, revisable `CommitTracker`)

Real use case: a reciter starts at some ayah and continues, so after committing ayah
X the next is almost certainly X+1. Two pieces:

- **`SequentialContext`** — a **sticky** continuation prior. After `set_current(X)`:
  the next `window` ayat (canonical order, surah boundaries work: 112:4 → 113:1) get a
  strong cost **bonus** (default 0.22, decaying with distance); all ayat of the **current
  surah** get a smaller `surah_bonus` (0.10, resists jumping out of the surah); and a
  **streak** boost (`streak_bonus` 0.05 per confirmed continuation) grows the prior as a
  sequence builds, so after a few ayat it's hard to dislodge. `set_current` grows the
  streak only when the new ayah is the expected continuation, else resets it. `rerank`
  applies `bonus_for(key)` to **partial** candidates. Still soft + the revisable commit
  corrects a genuine jump (streak then resets). Tune via the demo's `--context-bonus`,
  `--surah-bonus`, `--streak-bonus`.
- **Revisable `CommitTracker`** — commits can change. First commit needs `persistence`
  (K=5); changing an established commit needs `revise_persistence` (default K+3,
  hysteresis). So it's flexible early (no context, leader can shift) but stable once a
  sequence is going. `update(top, margin)` returns the key on (re)commit; `.committed`
  holds the latest.

The demo (`demo/live_detect.py`) carries `SequentialContext` across utterances
(each ayah's commit sets the expectation for the next) and shows `REVISED →` when the
detection changes its mind.

### Ayah-end detection (`PhonemeMatcher.ayah_progress`)

`ayah_progress(key, complete_cost=0.40)` walks the ayah's trie path (`key_to_path`),
finds the deepest frontier node (≈ recitation position in that ayah), and returns
`(progress 0..1, terminal_norm_cost, complete)`. `complete` flips True when the ayah's
TERMINAL is reached with norm-cost ≤ `complete_cost` — a **content-based ayah-end
signal, independent of pauses**. Verified: progress climbs 0→100% monotonically and
completion fires for all tested ayat, first at ~72% of phonemes on average (deletion-
advance reaches the terminal just before the literal end — good for pre-advancing UI).

The demo uses this to announce `● COMPLETE` and pre-advance: when ayah X completes it
sets `SequentialContext.current = X`, so the UI shows X+1 and detection expects the
continuation. Drives "show ayah 12 automatically once 11 is done." Tune via
`--complete-cost` (lower = fires later/stricter).

### Early detection — partial-path scoring (`partial_candidates`, `partial_for`)

`candidates()` returns only **completed** ayat (terminal nodes), so a long ayah isn't a
candidate until its end. `partial_candidates(k, min_progress)` fixes this: per ayah it
walks the path and ranks by the **minimum normalized cost** over active frontier nodes
(the best-matching prefix), while reporting `progress` from the **deepest** active node.
Ranking by best-prefix (not deepest) matters: a mispronounced *ending* would otherwise
force the match deep at high cost and sink the correct ayah (this was the 114:6↔88:10
confusion — a garbled "wan-naas" ending dropped 114:6 out of top-3; min-cost keeps it
#1). So a long ayah ranks high from its first words AND a fully-recited ayah survives a
bad ending. The frontier self-filters — a shallow node after many input
phonemes carries high insertion cost and is beam-pruned — so "deepest active node" ≈
input position with low cost, for plausible ayat only.

Measured (clean prefixes, min-cost scoring): from **30%** of an ayah, partial top-1 75%
/ top-3 94%; from 50%, 92% / 98%; from 70%, 97% / 100%. Beam widened **300→600** so the
correct ayah isn't pruned when similar-sounding ayat crowd it (per-pass ~29 ms over all
564 ayat — still negligible). Also lifted error robustness (20% err 96→98%, 30% 92→96%).
`SequentialContext.rerank` runs on partial candidates, so early detection + continuation
prior combine; `min_progress` (default 0.2) filters tiny ambiguous prefixes. `candidates()`
retained unchanged for eval/commit back-compat.

## Tuning knobs

`sub/ins/del_penalty` (default 1/1/1 = plain Levenshtein), `beam_width` (300),
`max_del` (2), `allow_restart`. Asymmetric penalties can encode learner-error
priors (e.g. cheaper deletions if learners skip letters). Defer tuning until the
end-to-end eval (`eval/`) shows where real errors come from.

## Integration

Runtime bridge (posteriors → phoneme stream) lives in `eval/evaluate.py`
(`greedy_phonemes`) and will move to the on-device runtime. The matcher API
(`reset` / `step` / `candidates` / `commit_margin`) is the streaming contract.
