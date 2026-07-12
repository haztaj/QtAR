# Salat marker — data-collection spec

Recording spec for the marker-detector adaptation set (see `research/CLAUDE.md`
"salat_eval.py / salat_probe.py" and the project memory `project-salat-state-detection`).

## Why this shape (the two problems it solves)

1. **Class balance.** In natural prayers the anchor is the rarest but most important marker
   (~1 sami'allah per rakah vs ~5 takbirs). Training on full prayers starves the signal the
   whole state machine hinges on. → Record **dedicated, balanced per-marker files** (each file =
   one marker → self-labeling).
2. **The real difficulty is off-axis + in-motion articulation**, not distance (the shipped ayah
   detector works at this same phone-on-stand placement). → Record each rep **performing the
   actual posture**, so the hard acoustic condition is captured. Recording markers standing still
   facing the phone would reproduce the clean/close condition that already works but does NOT
   transfer.

## The set = balanced POSITIVES + real NEGATIVES

Positives alone make a detector that cannot REJECT recitation — that is exactly why Fatiha's
«ar-rahmani r-rahim» fired false salams on the audible prayers. So the set needs both:

- **Positives** — the dedicated per-marker files below.
- **Negatives** — real recitation + silence + movement/rustle noise, supplied by the **full real
  prayers** (`data/{Isha_4_raka_audible,Sunna_2_raka_silent,Witr_3_raka_audible}.mp4`, plus more)
  and the app's recitation corpus. Full prayers are ALSO the validation exam (see below).

## Recording protocol (applies to every positive file)

- **Phone on the stand, at real prayer distance/position** — identical to deployment.
- **Perform the posture each rep** — this is the whole point; do not shortcut it standing still.
- **Natural in-prayer cadence + variation** — say each as you would in prayer (not a rushed
  machine-gun); let speed / pitch / vowel length vary naturally across reps.
- **One marker (and one position) per file** — keeps labeling trivial.
- **Spread over 2–3 sessions / days** so room acoustics and voice vary.
- Native phone format is fine (`.mp4` / `.m4a`); the pipeline ffmpeg-converts to 16 kHz mono.

## What to record (positives)

Takbir is NOT one condition — it is said at several postures with very different acoustics, so
capture each. sami'allah has one natural position; salam has two (opposite head turns).

| File (suggested name) | Marker / position | Acoustic condition | Target reps |
|---|---|---|---|
| `salat_pos_takbir_stand_sN` | takbir, standing (ihram) | on-axis (easiest) | ~40 |
| `salat_pos_takbir_ruku_sN`  | takbir, bowing into ruku | forward/down, off-axis | ~40 |
| `salat_pos_takbir_sujud_sN` | takbir, going into sujud | **face-to-floor — most muffled/off-axis, likely hardest** | ~40 |
| `salat_pos_samiallah_sN`    | sami'allah, rising from ruku | head coming up | ~50 |
| `salat_pos_salam_right_sN`  | salam, turning head right | ~90° off-axis right | ~30 |
| `salat_pos_salam_left_sN`   | salam, turning head left | ~90° off-axis left | ~30 |

(`sN` = session number, e.g. `_s1`, `_s2`. Reps are approximate — more is better, balance matters
more than exact counts. The face-to-floor sujud takbir is the priority: it's the hardest condition
and takbir currently has the "akbar" content hole that fails even close-mic.)

## Negatives (already partly in hand)

- The 3 full prayers already recorded; add a few more (varied prayers/pace) over time.
- Optionally a dedicated `salat_neg_recitation_sN` (plain recitation on the stand) if more
  recitation negatives are wanted beyond the app corpus.

## Validation discipline (non-negotiable)

Train on the isolated positives + negatives, but **ALWAYS measure success on the real full
prayers — never on the isolated files.** The isolated files are training fuel; the real prayers
are the exam. This is the taint-audit lesson: the first isolated clip looked great (11/14) and did
not transfer to real prayers. Metric = per-marker detection on the full prayers, anchor =
sami'allah-count-per-rakah (Isha 4, Witr 3, Sunna 2) + salam-at-end, with recitation false-positive
rate tracked.

## Delivery

Place all files in `data/` (audio is gitignored — never committed). Ping me once a session is in
and I'll extract/segment the positives, assemble positives+negatives, adapt the marker detector,
and gate on the full prayers.
