#!/usr/bin/env python3
"""
Offline regression for the live-detect matching modes.

Runs each preserved test-fixture recording (demo/test_fixtures/) through the stream/sliding
pipelines and asserts the committed ayah sequence matches the expected golden. Guards the
tricky cases so a future matcher/model change can't silently break them:

  - a single LONG ayah (78:40) that the sliding window can't see  -> stream detects it
  - a CONTINUOUS run of long ayat (78:38->39->40)                 -> stream commits each
  - a continuous run of short ayat (114:1->2->3)                  -> sliding advances through

The `.wav` files are gitignored (audio rule), so a missing fixture is SKIPPED, not failed —
this runs wherever the audio is present (the committed `*_events.jsonl` document each case).

  python demo/regression.py            # -> ALL PASS / FAILURES (exit 0 / 1)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "matcher"))
sys.path.insert(0, str(REPO / "demo"))

from data import logmel_16k, load_tokens, load_ayah_phonemes   # noqa: E402
from model import EmformerCTC                                  # noqa: E402
from phoneme_matcher import PhonemeTrie, SequentialContext     # noqa: E402
import sliding                                                 # noqa: E402
import streaming                                               # noqa: E402
import auto                                                    # noqa: E402

FIX = REPO / "demo" / "test_fixtures"
CKPT = REPO / "training" / "exp" / "best_mic.pt"
NORM_RMS = 0.1

# (wav filename, mode, expected committed ayah sequence). Goldens verified 2026-07-04.
CASES = [
    # auto (default) — one mode across every ayah length; this is what the demo runs.
    ("user_78_40_naba_long.wav",             "auto",    ["78:40"]),
    ("user_78_38to40_naba_continuous.wav",   "auto",    ["78:38", "78:39", "78:40"]),
    ("user_78_38to40_naba_continuous_b.wav", "auto",    ["78:38", "78:39", "78:40"]),
    ("user_85_12to16_buruj.wav",             "auto",    ["85:12", "85:13", "85:14", "85:15", "85:16"]),
    ("user_114_quietmic.wav",                "auto",    ["114:1", "114:2", "114:3"]),
    # mixed long+medium+short continuous: KNOWN PARTIAL (3/4) — 98:2 (medium) is in the gap;
    # the growing-buffer decode lag is the limit (streaming-export would fix). Guards against
    # regressing below this; bump the golden if 98:2 gets recovered.
    ("user_98_1to4_bayyinah_mixed.wav",      "auto",    ["98:1", "98:3", "98:4"]),
    # the underlying single modes (what auto merges): sliding = short, stream = long.
    ("user_78_40_naba_long.wav",             "stream",  ["78:40"]),
    ("user_78_38to40_naba_continuous.wav",   "stream",  ["78:38", "78:39", "78:40"]),
    ("user_85_12to16_buruj.wav",             "sliding", ["85:12", "85:13", "85:14", "85:15", "85:16"]),
    ("user_114_quietmic.wav",                "sliding", ["114:1", "114:2", "114:3"]),
]


def make_context(trie):
    return SequentialContext(list(trie.key_to_node.keys()), bonus=0.22, window=2,
                             surah_bonus=0.10, streak_bonus=0.05)


def run_case(wav, sr, decode, trie, ap, mode):
    if mode == "auto":
        ev = auto.run_offline(wav, sr, decode, trie, ap, window_s=4.0, hop_s=1.0)
    elif mode == "stream":
        ev = streaming.run_offline(wav, sr, decode, trie, make_context(trie), ap, hop_s=1.0)
    elif mode == "sliding":
        ev = sliding.run_offline(wav, sr, decode, ap, make_context(trie),
                                 window_s=4.0, hop_s=1.0, max_cost=0.30)
    else:
        raise ValueError(mode)
    return [e["ayah"] for e in ev]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokens()
    id2 = {v: k for k, v in tok.items()}
    ap = load_ayah_phonemes()
    trie = PhonemeTrie.from_ayah_phonemes(ap)
    ck = torch.load(CKPT, map_location=device)
    model = EmformerCTC(num_tokens=ck["vocab"]).to(device).eval()
    model.load_state_dict(ck["model"])

    @torch.no_grad()
    def decode(buf):
        w = buf
        rms = float(np.sqrt((w ** 2).mean()) + 1e-9)
        w = np.clip(w * (NORM_RMS / rms), -1.0, 1.0).astype(np.float32)
        feats = logmel_16k(torch.from_numpy(np.ascontiguousarray(w))).unsqueeze(0).to(device)
        lp, ol = model(feats, torch.tensor([feats.shape[1]], device=device))
        ids = lp[0, :int(ol[0])].argmax(-1).cpu().tolist()
        out, prev = [], -1
        for s in ids:
            if s != prev and s != 0:
                out.append(id2[s])
            prev = s
        return out

    print(f"model: {CKPT.name} ({device})\n" + "-" * 68)
    passed = failed = skipped = 0
    for fname, mode, expected in CASES:
        path = FIX / fname
        if not path.exists():
            print(f"  SKIP  {mode:<8} {fname:<38} (fixture .wav not present)")
            skipped += 1
            continue
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        got = run_case(wav, sr, decode, trie, ap, mode)
        ok = got == expected
        passed += ok
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {mode:<8} {fname:<38} "
              f"{got}" + ("" if ok else f"  != expected {expected}"))

    print("-" * 68)
    print(f"RESULT: {passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
