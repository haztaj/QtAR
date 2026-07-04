#!/usr/bin/env python3
"""
Live microphone ayah detection.

Recite into the mic; the script shows the running best-guess ayah and commits
when confident. Pipeline mirrors the design:

  mic -> Silero VAD gate -> 16 kHz log-mel -> Emformer+CTC -> greedy phonemes
       -> fuzzy matcher -> ranked surah:ayah -> persistence-based commit

Uses the PyTorch checkpoint directly (variable-length; no fixed ONNX window).

  python demo/live_detect.py
  python demo/live_detect.py --device 3 --checkpoint training/exp/best_mic.pt

Press Ctrl-C to quit. List mics with:  python demo/live_detect.py --list-devices
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "matcher"))

from data import logmel_16k, load_tokens, load_ayah_phonemes          # noqa: E402
from model import EmformerCTC                                         # noqa: E402
from phoneme_matcher import (PhonemeTrie, PhonemeMatcher, CommitTracker,  # noqa: E402
                             SequentialContext)

SR = 16000
VAD_BLOCK = 512                 # silero v5 wants 512-sample chunks @ 16 kHz (32 ms)
AYAH_TEXT = REPO / "data" / "manifests" / "ayah_text.json"

SURAH_NAMES = {
    78: "An-Naba", 79: "An-Nazi'at", 80: "Abasa", 81: "At-Takwir", 82: "Al-Infitar",
    83: "Al-Mutaffifin", 84: "Al-Inshiqaq", 85: "Al-Buruj", 86: "At-Tariq", 87: "Al-A'la",
    88: "Al-Ghashiyah", 89: "Al-Fajr", 90: "Al-Balad", 91: "Ash-Shams", 92: "Al-Layl",
    93: "Ad-Duha", 94: "Ash-Sharh", 95: "At-Tin", 96: "Al-Alaq", 97: "Al-Qadr",
    98: "Al-Bayyinah", 99: "Az-Zalzalah", 100: "Al-Adiyat", 101: "Al-Qari'ah",
    102: "At-Takathur", 103: "Al-Asr", 104: "Al-Humazah", 105: "Al-Fil", 106: "Quraysh",
    107: "Al-Ma'un", 108: "Al-Kawthar", 109: "Al-Kafirun", 110: "An-Nasr", 111: "Al-Masad",
    112: "Al-Ikhlas", 113: "Al-Falaq", 114: "An-Nas",
}


def name(key: str) -> str:
    s, a = key.split(":")
    return f"{SURAH_NAMES.get(int(s), 'S'+s)} {s}:{a}"


class SessionRecorder:
    """Records the full session audio + a timestamped log of finalized ayat, so a
    detection can be re-investigated later. Reset (overwritten) each run — one session
    on disk. See demo/analyze_session.py and demo/CLAUDE.md.

      <dir>/session.wav    full 16 kHz mono recording (includes pauses, for true times)
      <dir>/events.jsonl   one line per finalized ayah (index, sample range, detection)
      <dir>/meta.json      model + args + start time
    """

    def __init__(self, out_dir: Path, sr: int, meta: dict, enabled: bool = True):
        self.enabled = enabled
        self.sr = sr
        self.samples = 0
        self.seg_start = 0
        self.index = 0
        if not enabled:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in ("session.wav", "events.jsonl", "meta.json"):
            p = out_dir / f
            if p.exists():
                p.unlink()
        self.dir = out_dir
        self.wav = sf.SoundFile(out_dir / "session.wav", "w", sr, 1, subtype="PCM_16")
        self.events = open(out_dir / "events.jsonl", "w", encoding="utf-8")
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def feed(self, block: np.ndarray) -> None:
        if self.enabled:
            self.wav.write(block)
        self.samples += len(block)

    def mark_speech_start(self) -> None:
        self.seg_start = self.samples

    def record(self, **fields) -> int:
        self.index += 1
        if self.enabled:
            ev = {"index": self.index,
                  "start_s": round(self.seg_start / self.sr, 2),
                  "end_s": round(self.samples / self.sr, 2),
                  "start_sample": self.seg_start, "end_sample": self.samples,
                  **fields}
            self.events.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self.events.flush()
        self.seg_start = self.samples
        return self.index

    def close(self) -> None:
        if self.enabled:
            self.wav.close()
            self.events.close()


def greedy(log_probs: torch.Tensor, length: int, id2tok) -> list[str]:
    ids = log_probs[0, :length].argmax(-1).tolist()
    out, prev = [], -1
    for s in ids:
        if s != prev and s != 0:
            out.append(id2tok[s])
        prev = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="training/exp/best_mic.pt")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.15, help="commit margin")
    ap.add_argument("--persistence", type=int, default=5, help="commit persistence (K)")
    ap.add_argument("--revise", type=int, default=8, help="persistence to CHANGE a commit (hysteresis)")
    ap.add_argument("--context-bonus", type=float, default=0.22,
                    help="cost bonus for the expected next ayah (0 disables sequential context)")
    ap.add_argument("--context-window", type=int, default=2, help="how many next ayat get the prior")
    ap.add_argument("--surah-bonus", type=float, default=0.10,
                    help="bonus for staying in the current surah (resists jumping out)")
    ap.add_argument("--streak-bonus", type=float, default=0.05,
                    help="extra bonus per confirmed continuation (stickier as a sequence builds)")
    ap.add_argument("--reset-tail", type=float, default=0.3,
                    help="audio (s) kept after an ayah completes, to seed the next ayah")
    ap.add_argument("--complete-cost", type=float, default=0.30,
                    help="max terminal norm-cost to call an ayah complete (ayah-end signal)")
    ap.add_argument("--min-progress", type=float, default=0.2,
                    help="min fraction of an ayah matched before it's an early-detection candidate")
    ap.add_argument("--infer-every", type=float, default=0.4, help="re-run inference every N s")
    ap.add_argument("--min-speech", type=float, default=0.5, help="min audio before inferring")
    ap.add_argument("--vad-threshold", type=float, default=0.5)
    ap.add_argument("--norm-rms", type=float, default=0.1,
                    help="gain-normalize input to this RMS (handles quiet mics; 0 disables)")
    ap.add_argument("--session-dir", default=str(REPO / "demo" / "sessions"),
                    help="where to record audio + detections (reset each run)")
    ap.add_argument("--no-record", action="store_true", help="disable session recording")
    ap.add_argument("--mode", default="sliding", choices=["sliding", "stream", "buffer"],
                    help="sliding: fixed-window whole-ayah matching (continuous short ayat); "
                         "stream: prefix-anchored, early detection of ayat of ANY length "
                         "(incl. long ayat the window can't see); buffer: legacy growing-buffer")
    ap.add_argument("--window", type=float, default=4.0, help="sliding window width (s)")
    ap.add_argument("--hop", type=float, default=1.0, help="sliding/stream hop (s)")
    ap.add_argument("--window-cost", type=float, default=0.30, help="max edit-cost for a confident window")
    ap.add_argument("--commit-cost", type=float, default=0.75,
                    help="stream mode: max prefix-align cost to commit (loose garbage gate; "
                         "rank persistence is the real commit signal)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Arabic display on Windows
    except Exception:
        pass

    if args.list_devices:
        print(sd.query_devices())
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok2id = load_tokens()
    id2tok = {v: k for k, v in tok2id.items()}
    ckpt = torch.load(REPO / args.checkpoint, map_location=device)
    model = EmformerCTC(num_tokens=ckpt["vocab"]).to(device).eval()
    model.load_state_dict(ckpt["model"])
    ayah_ph = load_ayah_phonemes()
    trie = PhonemeTrie.from_ayah_phonemes(ayah_ph)
    ayah_text = json.loads(AYAH_TEXT.read_text(encoding="utf-8")) if AYAH_TEXT.exists() else {}

    def vad_active(win):                       # cheap energy gate (skip near-silence)
        return float(np.sqrt((win ** 2).mean())) > 0.005

    from silero_vad import load_silero_vad, VADIterator
    vad = VADIterator(load_silero_vad(), threshold=args.vad_threshold,
                      sampling_rate=SR, min_silence_duration_ms=500)

    print(f"model: {args.checkpoint} ({device})  |  commit T={args.threshold} K={args.persistence}")
    print(f"mic: {sd.query_devices(args.device, 'input')['name'] if args.device is not None else 'default'}")
    print("\nRecite a Juz-Amma ayah. Ctrl-C to quit.\n" + "-" * 64)

    seq = SequentialContext(list(trie.key_to_node.keys()),
                            bonus=args.context_bonus, window=args.context_window,
                            surah_bonus=args.surah_bonus, streak_bonus=args.streak_bonus)

    @torch.no_grad()
    def infer(buf: np.ndarray):
        """Returns (ranked, committed_key, top_key, detected_key, progress, phonemes).
        `detected` = committed if confident else the current top candidate; completion/
        advance keys off `detected` so a continuous recitation advances even when a
        shared-prefix ayah (e.g. 113:1 vs 114:1) blocks a formal commit."""
        wav = buf
        if args.norm_rms > 0:                             # lift quiet mic audio into the
            rms = float(np.sqrt((wav ** 2).mean()) + 1e-9)  # (loud) training distribution
            wav = np.clip(wav * (args.norm_rms / rms), -1.0, 1.0).astype(np.float32)
        feats = logmel_16k(torch.from_numpy(np.ascontiguousarray(wav))).unsqueeze(0).to(device)
        lengths = torch.tensor([feats.shape[1]], device=device)
        log_probs, out_lens = model(feats, lengths)
        phons = greedy(log_probs.cpu(), int(out_lens[0]), id2tok)
        m = PhonemeMatcher(trie, allow_restart=False)
        tracker = CommitTracker(args.threshold, args.persistence, args.revise)
        for p in phons:
            m.step(p)
            _, top_key, margin = seq.rerank(m, k=3, min_progress=args.min_progress)
            tracker.update(top_key, margin)
        ranked, top_key, _ = seq.rerank(m, k=3, min_progress=args.min_progress)
        committed = tracker.committed
        detected = committed or top_key
        prog = m.ayah_progress(detected, args.complete_cost) if detected else (0.0, None, False)
        return ranked, committed, top_key, detected, prog, phons

    @torch.no_grad()
    def decode_window(buf: np.ndarray):
        """Greedy phonemes for one audio window (sliding mode)."""
        wav = buf
        if args.norm_rms > 0:
            rms = float(np.sqrt((wav ** 2).mean()) + 1e-9)
            wav = np.clip(wav * (args.norm_rms / rms), -1.0, 1.0).astype(np.float32)
        feats = logmel_16k(torch.from_numpy(np.ascontiguousarray(wav))).unsqueeze(0).to(device)
        lp, ol = model(feats, torch.tensor([feats.shape[1]], device=device))
        return greedy(lp.cpu(), int(ol[0]), id2tok)

    rec = SessionRecorder(
        Path(args.session_dir), SR,
        meta={"checkpoint": args.checkpoint, "device": device,
              "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "args": {k: v for k, v in vars(args).items()}},
        enabled=not args.no_record)
    if rec.enabled:
        print(f"recording session -> {args.session_dir}")

    q: queue.Queue = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    # ---------------- stream mode (prefix-anchored; early detection, any ayah length) ----
    if args.mode == "stream":
        from streaming import StreamDetector
        det = StreamDetector(trie, seq, ayah_ph, persistence=3, jump_persistence=5,
                             min_progress=args.min_progress, commit_cost_max=args.commit_cost)
        H = int(args.hop * SR)
        MAXBUF = int(30 * SR)                 # hard cap (older audio slides out)
        SILENCE_RESET = 2.0                   # s of silence -> new passage: clear buffer + context
        verbs = {"detect": "DETECTED", "advance": "→ NEXT", "jump": "JUMP →"}
        print(f"mode: stream  hop={args.hop}s  (prefix-anchored — early detection incl. long ayat)")
        print("Recite ayat (any length; continuous is fine). Ctrl-C to quit.\n" + "-" * 64)
        buf = np.zeros(0, dtype=np.float32)
        total = last_proc = silent_hops = 0
        with sd.InputStream(samplerate=SR, channels=1, blocksize=VAD_BLOCK,
                            dtype="float32", device=args.device, callback=cb):
            try:
                while True:
                    block = q.get()
                    rec.feed(block)
                    total += len(block)
                    buf = np.concatenate([buf, block])[-MAXBUF:]
                    if total - last_proc < H:
                        continue
                    last_proc = total
                    if not vad_active(buf[-int(1.0 * SR):]):    # recent near-silence
                        silent_hops += 1
                        if silent_hops * args.hop >= SILENCE_RESET and len(buf):
                            buf = np.zeros(0, dtype=np.float32)  # a real pause -> fresh passage
                            det.reset()
                            silent_hops = 0
                        continue
                    silent_hops = 0
                    if len(buf) < int(args.min_speech * SR):
                        continue
                    st = det.feed(decode_window(buf))
                    ctx = f"*{seq.streak}" if seq.current else "  "
                    line = "  ".join(f"{name(k)}({c:.2f},{pr:.0%})" for k, c, pr in st["ranked"])
                    print(f"\r{ctx}[{len(buf)/SR:4.1f}s] {line:<62}", end="", flush=True)
                    ce = st["commit_event"]
                    if ce:
                        print(f"\r✓ {verbs[ce['event']]}  {name(ce['ayah']):<26}(cost {ce['cost']})",
                              flush=True)
                        if ayah_text.get(ce["ayah"]):
                            print(f"            {ayah_text[ce['ayah']]}")
                        rec.record(detected=ce["ayah"], committed=True, completed=False,
                                   expected=seq.current, streak=seq.streak,
                                   top3=[[k, round(c, 3), round(pr, 3)] for k, c, pr in st["ranked"]],
                                   phonemes="")
                    if st["refocus"]:              # new ayah started -> refocus decode on it
                        buf = buf[-int(st["refocus"] * SR):]
            except KeyboardInterrupt:
                print("\nbye.")
            finally:
                rec.close()
                if rec.enabled:
                    print(f"session saved: {rec.index} ayat -> {args.session_dir}")
        return

    # ---------------- sliding-window mode (continuous recitation, no pauses) -----------
    if args.mode == "sliding":
        from sliding import SlidingWindowSegmenter
        seg = SlidingWindowSegmenter(None, seq, ayah_ph, max_cost=args.window_cost)
        W, H = int(args.window * SR), int(args.hop * SR)
        print(f"mode: sliding  window={args.window}s hop={args.hop}s")
        print("Recite continuously (no pauses needed). Ctrl-C to quit.\n" + "-" * 64)
        rolling = np.zeros(0, dtype=np.float32)
        total = 0
        last_proc = 0
        prev_t = 0.0
        with sd.InputStream(samplerate=SR, channels=1, blocksize=VAD_BLOCK,
                            dtype="float32", device=args.device, callback=cb):
            try:
                while True:
                    block = q.get()
                    rec.feed(block)
                    total += len(block)
                    rolling = np.concatenate([rolling, block])[-2 * W:]
                    if total - last_proc < H or len(rolling) < int(0.5 * SR):
                        continue
                    last_proc = total
                    speech = vad_active(rolling[-W:])
                    if not speech:
                        continue
                    ev = seg.process(decode_window(rolling[-W:]), total / SR)
                    if ev:
                        k = ev["ayah"]
                        verb = {"detect": "DETECTED", "advance": "→ NEXT", "jump": "JUMP →"}[ev["event"]]
                        print(f"\r✓ {verb}  {name(k):<28} (cost {ev['cost']})", flush=True)
                        if ayah_text.get(k):
                            print(f"            {ayah_text[k]}")
                        rec.seg_start = int(prev_t * SR)
                        rec.record(detected=k, committed=ev["event"] != "jump", completed=True,
                                   expected=seq.current, streak=seq.streak,
                                   top3=[[k, ev["cost"], 1.0]], phonemes="")
                        prev_t = ev["t"]
            except KeyboardInterrupt:
                print("\nbye.")
            finally:
                rec.close()
                if rec.enabled:
                    print(f"session saved: {rec.index} ayat -> {args.session_dir}")
        return

    # ---------------- buffer mode (legacy growing-buffer + completion-reset) -----------
    buf = np.zeros(0, dtype=np.float32)
    in_speech = False
    last_infer_len = 0
    shown = None             # last committed key shown this utterance (revision detect)
    announced_done = None    # ayah whose completion we've already announced

    def next_name(key):
        i = seq._idx.get(key)
        return name(seq._order[i + 1]) if i is not None and i + 1 < len(seq._order) else "—"

    def make_event(ranked, detected, committed_flag, phons, completed):
        return {"detected": detected, "committed": committed_flag, "completed": completed,
                "expected": seq.current, "streak": seq.streak,
                "top3": [[k, round(c, 3), round(pr, 3)] for k, c, pr in ranked],
                "phonemes": " ".join(phons)}

    def show(buf):
        """Run one inference pass, update the display. Returns an event dict if this
        pass finished an ayah (so the loop can advance + record it), else None."""
        nonlocal shown, announced_done
        ranked, committed, top_key, detected, (prog, _cost, complete), phons = infer(buf)
        if not ranked:
            return None
        ctx = f"*{seq.streak}" if seq.current else "  "   # context active + streak
        line = "  ".join(f"{name(k)}({c:.2f},{pr:.0%})" for k, c, pr in ranked)
        bar = f" {prog:3.0%}"
        print(f"\r{ctx}[{len(buf)/SR:4.1f}s]{bar} {line:<62}", end="", flush=True)

        if detected and detected != shown:
            verb = "DETECTED" if shown is None else "REVISED →"
            shown = detected
            tag = "" if committed else " (tentative)"
            print(f"\r✓ {verb}  {name(detected)}{tag:<20}", flush=True)
            if ayah_text.get(detected):
                print(f"            {ayah_text[detected]}")

        # content-based ayah-end: advance when the TOP ayah finishes (commit not required)
        if detected and complete and announced_done != detected:
            announced_done = detected
            ev = make_event(ranked, detected, committed is not None, phons, completed=True)
            seq.set_current(detected)            # detection now expects detected+1 (streak++)
            print(f"\r● COMPLETE  {name(detected)}   →  up next: {next_name(detected)}",
                  flush=True)
            return ev
        return None

    with sd.InputStream(samplerate=SR, channels=1, blocksize=VAD_BLOCK,
                        dtype="float32", device=args.device, callback=cb):
        try:
            while True:
                block = q.get()
                rec.feed(block)                      # full session audio (incl. pauses)
                evt = vad(torch.from_numpy(block), return_seconds=True)
                if evt and "start" in evt:
                    in_speech, buf, last_infer_len = True, block.copy(), 0
                    shown = announced_done = None
                    rec.mark_speech_start()
                elif in_speech:
                    buf = np.concatenate([buf, block])

                # periodic live inference — keeps updating, may REVISE after committing
                if in_speech and len(buf) >= args.min_speech * SR \
                        and len(buf) - last_infer_len >= args.infer_every * SR:
                    last_infer_len = len(buf)
                    ev = show(buf)
                    if ev:
                        idx = rec.record(**ev)       # ayah finalized via completion
                        # advance NOW (don't wait for a pause); keep a short tail.
                        tail = int(args.reset_tail * SR)
                        buf = buf[-tail:] if len(buf) > tail else buf
                        last_infer_len = len(buf)
                        shown = announced_done = None

                # end of utterance: finalize anything not already auto-completed
                if evt and "end" in evt:
                    if in_speech and len(buf) >= args.min_speech * SR:
                        ranked, committed, _, detected, _, phons = infer(buf)
                        final = detected or (ranked[0][0] if ranked else None)
                        if final:
                            if committed is None:
                                print(f"\r~ best guess {name(final):<30}", flush=True)
                            rec.record(**make_event(ranked, final, committed is not None,
                                                    phons, completed=False))
                            seq.set_current(final)   # next utterance expects final+1
                            print(f"  (up next: {next_name(final)})")
                    in_speech = False
                    buf = np.zeros(0, dtype=np.float32)
                    print("-" * 64)
        except KeyboardInterrupt:
            print("\nbye.")
        finally:
            rec.close()
            if rec.enabled:
                print(f"session saved: {rec.index} ayat -> {args.session_dir}")


if __name__ == "__main__":
    main()
