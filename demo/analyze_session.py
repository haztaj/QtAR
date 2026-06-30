#!/usr/bin/env python3
"""
Investigate a recorded live session (demo/live_detect.py writes one).

  python demo/analyze_session.py                 # list every finalized ayah
  python demo/analyze_session.py 4               # deep-dive on the 4th ayah
  python demo/analyze_session.py 4 --play-out demo/sessions/aya4.wav

For a given ayah it re-extracts the exact audio segment from session.wav, re-runs the
acoustic model + matcher, and prints: the decoded phonemes, the candidate ranking with
and without the sequential context that was active at the time, and the Arabic text of
the top candidates — enough to see why a detection went the way it did.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "matcher"))

from data import logmel_16k, load_tokens, load_ayah_phonemes      # noqa: E402
from model import EmformerCTC                                     # noqa: E402
from phoneme_matcher import PhonemeTrie, PhonemeMatcher, SequentialContext  # noqa: E402

SR = 16000


def load_session(d: Path):
    events = [json.loads(l) for l in (d / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return events, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", nargs="?", type=int, help="ayah index to investigate (1-based)")
    ap.add_argument("--session-dir", default=str(REPO / "demo" / "sessions"))
    ap.add_argument("--checkpoint", default=None, help="override; else uses the session's")
    ap.add_argument("--play-out", default=None, help="write the segment WAV here")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    d = Path(args.session_dir)
    if not (d / "events.jsonl").exists():
        print(f"no session at {d} (run demo/live_detect.py first)")
        return
    events, meta = load_session(d)

    # --- list mode ---
    if args.index is None:
        print(f"session: {meta.get('started')}  model={meta.get('checkpoint')}  {len(events)} ayat\n")
        print(f"{'#':>3} {'time':>12}  {'detected':>10} {'cmt':>3} {'done':>4}  context  top-3")
        for e in events:
            t = f"{e['start_s']:.1f}-{e['end_s']:.1f}s"
            top = " ".join(k for k, *_ in e["top3"])
            print(f"{e['index']:>3} {t:>12}  {e['detected'] or '-':>10} "
                  f"{'Y' if e['committed'] else '.':>3} {'Y' if e['completed'] else '.':>4}  "
                  f"{str(e['expected'] or '-'):>7}  {top}")
        print("\nRun with an index to deep-dive, e.g.  python demo/analyze_session.py 4")
        return

    ev = next((e for e in events if e["index"] == args.index), None)
    if ev is None:
        print(f"no ayah #{args.index} (session has {len(events)})")
        return

    # --- deep-dive ---
    audio, sr = sf.read(d / "session.wav", dtype="float32")
    seg = audio[ev["start_sample"]:ev["end_sample"]]
    if args.play_out:
        sf.write(args.play_out, seg, sr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = args.checkpoint or meta.get("checkpoint", "training/exp/best_mic.pt")
    tok = load_tokens(); id2tok = {v: k for k, v in tok.items()}
    ck = torch.load(REPO / ckpt_path, map_location=device)
    model = EmformerCTC(num_tokens=ck["vocab"]).to(device).eval(); model.load_state_dict(ck["model"])
    ap_ph = load_ayah_phonemes(); trie = PhonemeTrie.from_ayah_phonemes(ap_ph)
    ayah_text = json.loads((REPO / "data" / "manifests" / "ayah_text.json").read_text(encoding="utf-8"))

    with torch.no_grad():
        feats = logmel_16k(torch.from_numpy(np.ascontiguousarray(seg))).unsqueeze(0).to(device)
        lp, ol = model(feats, torch.tensor([feats.shape[1]], device=device))
    ids = lp.cpu()[0, :int(ol[0])].argmax(-1).tolist()
    ph, prev = [], -1
    for s in ids:
        if s != prev and s != 0:
            ph.append(id2tok[s])
        prev = s

    print(f"=== ayah #{ev['index']}  ({ev['start_s']:.1f}-{ev['end_s']:.1f}s, "
          f"{(ev['end_s']-ev['start_s']):.1f}s) ===")
    print(f"recorded detection : {ev['detected']}  (committed={ev['committed']} "
          f"completed={ev['completed']}, context expected {ev['expected']}, streak {ev['streak']})")
    print(f"recorded top-3     : {ev['top3']}")
    print(f"decoded phonemes   : {' '.join(ph)}  (live: {ev['phonemes']})")

    def show_rank(label, current):
        seq = SequentialContext(list(trie.key_to_node.keys()))
        seq.set_current(current)
        m = PhonemeMatcher(trie, allow_restart=False); m.step_many(ph)
        ranked, _, _ = seq.rerank(m, k=5, min_progress=0.2)
        print(f"\n{label}:")
        for k, c, pr in ranked:
            print(f"   {k:>8}  cost={c:+.2f}  prog={pr:.0%}   {ayah_text.get(k,'')}")

    show_rank("ranking WITHOUT context", None)
    if ev["expected"]:
        show_rank(f"ranking WITH context (expected {ev['expected']})", ev["expected"])
    if args.play_out:
        print(f"\nsegment audio -> {args.play_out}")


if __name__ == "__main__":
    main()
