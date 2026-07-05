# Mobile audio capture — recommendations (for the app phase)

Captured 2026-06-30. **Android implemented** in `sdk/android/quranrecite/.../AudioCapture.kt`
(`VOICE_RECOGNITION` + `AutomaticGainControl`, NS/AEC off); iOS reference for later.

> **Capture must not share a thread with inference (learned 2026-07-05).** The first Android
> build ran the model inline on the `AudioRecord` read loop, so any inference stall (ORT warm-up,
> GC, a long stream-buffer decode) stopped `read()` and AudioRecord silently overran its buffer —
> a real session lost **~30% of samples** (20 s captured over a 30 s recitation), punching holes in
> the audio and wrecking detection. Fix: a reader thread that only `read()`s + enqueues, and a
> separate worker that drains the queue into the engine (`AudioCapture` reader→queue→worker), with a
> 1 s AudioRecord buffer for headroom. `stop()` **joins both threads** before releasing, so no
> callback fires after stop. This is the concrete realization of §8 (one capture thread, one
> inference worker) — treat it as mandatory, not optional.

## Context

Real use will face poorly-tuned / quiet mics (the test session was RMS ~0.02 vs
~0.1 studio). The root issue is **capture gain**, and capture-time gain control beats
post-hoc software normalization: software normalization amplifies the recorded noise
floor too, whereas capture-time AGC makes the ADC digitize a louder signal → better SNR.

## The principle

Most built-in "enhancement" is tuned for telephony/voice-chat. For *recitation*:
- **AGC (Automatic Gain Control)** — WANT IT. Directly fixes the quiet-mic problem.
- **Noise Suppression (NS)** — CAUTION. Tuned for conversational speech; can distort
  sustained vowels / **madd**. A/B test; don't assume it helps.
- **Acoustic Echo Cancellation (AEC)** — SKIP. No playback to cancel.

Goal: **AGC yes, NS cautious, AEC off** → i.e. don't use the all-in-one "voice
communication" mode; enable pieces selectively.

## Android

- **AudioSource (biggest decision):**
  - `VOICE_RECOGNITION` — ASR-tuned, light processing; **best default.**
  - `UNPROCESSED` (API 24+, gate on `PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED`) — raw;
    use if doing our own gain.
  - Avoid `VOICE_COMMUNICATION` (full telephony AEC+NS+AGC, too aggressive).
- **android.media.audiofx** (attach to AudioRecord session id; hardware-dependent, gate
  on `.isAvailable()`): `AutomaticGainControl` (enable), `NoiseSuppressor` (A/B test vs
  madd), `AcousticEchoCanceler` (skip).

## iOS

- **AVAudioSession mode:** `.measurement` minimizes system DSP (no AGC/NS — then rely on
  our own gain); `.voiceChat`/`.videoChat` enable Apple voice processing (usually too
  aggressive).
- **Voice Processing I/O** (`kAudioUnitSubType_VoiceProcessingIO` / `setVoiceProcessingEnabled`,
  iOS 13+) — gives Apple AGC; newer iOS lets you selectively bypass AGC/ducking stages.
- **`AVAudioSession.inputGain`** — only if `isInputGainSettable` (often false on iPhones);
  don't rely on it.

## Recommendation for the app

1. ASR-tuned path: `VOICE_RECOGNITION` (Android) / `.measurement` + own gain (iOS), with
   **AGC enabled**, **NS off or A/B-tested**.
2. Keep the **software RMS normalization** (already in the inference front-end) as a floor
   — consistent level into the model regardless of device AGC.
3. **Consistency matters:** whatever the app enables changes the signal distribution, and
   it varies wildly across OEM devices → the model must still be trained/augmented for
   that variation. Built-in enhancement reduces severity; augmentation handles the
   residual device-to-device variation. Complementary, not either/or.

## Caveats

- Effect availability is device-dependent — gate everything on availability checks.
- Behaviors shift across OS versions — verify on actual target devices.
