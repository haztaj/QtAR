#!/usr/bin/env python3
"""
Generate golden conformance fixtures from the Python reference, so the C++ SDK port can
be validated bit-for-bit-ish against it. See conformance/spec.md.

Produces (under conformance/):
  assets/mel_filterbank.bin   [n_freqs=201, n_mels=80] f32 — exact mel filters (use these
                              in C++ verbatim to eliminate a whole class of mismatch)
  assets/hann_window.bin      [400] f32 — exact analysis window
  assets/tokens.txt, assets/ayah_phonemes.json — Stage-2 lexicon (self-contained copy)
  assets/edit_cases.json      unit cases for the normalized edit distance
  fixtures/frontend/*.wav     16 kHz mono inputs
  golden/frontend/*.logmel.bin + shape — expected log-mel [T,80] f32
  fixtures/matcher/*.json      window phoneme sequences (decoupled from the model)
  golden/matcher/*.events.json expected sliding-window ayah events
  manifest.json               index + tolerances

Run: python conformance/generate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "matcher"))
sys.path.insert(0, str(REPO / "demo"))


def gen_highlight(CONF):
    """Golden for the Stage-3 HighlightController (deferral + centralized snapshots).

    Model-independent: drives the controller with committed-detection sequences and records
    the render-ready snapshot after each step. Covers every deferral path so the C++ port is
    pinned. Self-contained (needs only the ambiguity map) — safe to call without the model.
    Returns the manifest fragment.
    """
    from highlight_controller import HighlightController   # matcher/ is on sys.path

    (CONF / "assets").mkdir(parents=True, exist_ok=True)
    (CONF / "fixtures" / "highlight").mkdir(parents=True, exist_ok=True)
    (CONF / "golden" / "highlight").mkdir(parents=True, exist_ok=True)
    ambig = REPO / "data" / "lang" / "ambiguous_ayat.json"
    # self-contained copy so the C++ conformance_runner needs only the conformance dir
    (CONF / "assets" / "ambiguous_ayat.json").write_text(ambig.read_text(encoding="utf-8"), encoding="utf-8")
    hc = HighlightController(ambig)

    # Each scenario is a list of input steps: {"detect": key} or {"choose": key}.
    scenarios = {
        # baseline: unambiguous ayat highlight immediately.
        "unambiguous_run": [{"detect": "78:1"}, {"detect": "78:2"}, {"detect": "78:3"}],
        # predecessor pins the ambiguous ayah -> confirm now, no deferral.
        "resolve_predecessor": [{"detect": "83:21"}, {"detect": "83:22"}],
        # successor pins it -> defer (active stays None), then retro-confirm on the next ayah.
        "defer_then_successor": [{"detect": "83:23"}, {"detect": "83:24"}],
        # context can't help (99:8 ends its surah) -> needs_choice, resolved manually.
        "unresolvable_choice": [{"detect": "99:8"}, {"choose": "99:7"}],
        # deferred ayah whose expected successor never comes -> falls back to needs_choice.
        "defer_broken_sequence": [{"detect": "83:23"}, {"detect": "78:5"}],
    }

    entries = []
    for name, steps in scenarios.items():
        hc.reset()
        states = []
        for step in steps:
            if "detect" in step:
                snap = hc.detect(step["detect"])
            else:
                snap = hc.choose(step["choose"])
            states.append(snap.to_dict())
        (CONF / "fixtures" / "highlight" / f"{name}.json").write_text(
            json.dumps({"steps": steps}, ensure_ascii=False, indent=1), encoding="utf-8")
        (CONF / "golden" / "highlight" / f"{name}.states.json").write_text(
            json.dumps({"states": states}, ensure_ascii=False, indent=1), encoding="utf-8")
        entries.append({"name": name, "steps": f"fixtures/highlight/{name}.json",
                        "states": f"golden/highlight/{name}.states.json", "n_steps": len(steps)})
        print(f"  highlight/{name}: {len(steps)} steps -> "
              f"final active={states[-1]['active']} pending={states[-1]['pending']}")
    return entries

from data import (logmel_16k, _mel, SAMPLE_RATE, N_MELS, N_FFT, HOP, WIN,  # noqa: E402
                  FMIN, FMAX, LOG_FLOOR, NORM_RMS, load_tokens, load_ayah_phonemes)
from model import EmformerCTC                                              # noqa: E402
from phoneme_matcher import PhonemeTrie, SequentialContext                 # noqa: E402
from sliding import SlidingWindowSegmenter, _edit_norm                     # noqa: E402

CONF = REPO / "conformance"
CKPT = REPO / "training" / "exp" / "best_mic.pt"
TOL_LOGMEL = 1e-2          # max abs diff allowed on log-mel (generous; tighten if port is good)


def save_f32(arr: np.ndarray, path: Path):
    arr.astype("<f4").ravel().tofile(path)


def main():
    for d in ["assets", "fixtures/frontend", "fixtures/matcher", "golden/frontend", "golden/matcher"]:
        (CONF / d).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokens(); id2tok = {v: k for k, v in tok.items()}
    ap = load_ayah_phonemes()
    trie = PhonemeTrie.from_ayah_phonemes(ap)
    ck = torch.load(CKPT, map_location=device)
    model = EmformerCTC(num_tokens=ck["vocab"]).to(device).eval(); model.load_state_dict(ck["model"])

    manifest = {"sample_rate": SAMPLE_RATE, "n_mels": N_MELS, "n_fft": N_FFT, "hop": HOP,
                "win": WIN, "fmin": FMIN, "fmax": FMAX, "log_floor": LOG_FLOOR,
                "norm_rms": NORM_RMS, "mel_scale": "htk", "window": "hann_periodic",
                "stft": "center=True, pad_mode=reflect, power=2.0",
                "tolerances": {"logmel_max_abs": TOL_LOGMEL, "phonemes": "exact", "events": "exact"},
                "checkpoint": str(CKPT.relative_to(REPO)),
                "frontend": [], "matcher": [], "highlight": [], "edit_cases": "assets/edit_cases.json"}

    # --- 0) highlight golden (model-independent — do it first) ---
    manifest["highlight"] = gen_highlight(CONF)

    # --- exact constants the C++ should reuse verbatim ---
    mel = _mel(SAMPLE_RATE)
    fb = mel.mel_scale.fb.numpy()            # [n_freqs, n_mels] = [201, 80]
    win = mel.spectrogram.window.numpy()     # [400]
    save_f32(fb, CONF / "assets" / "mel_filterbank.bin")
    save_f32(win, CONF / "assets" / "hann_window.bin")
    manifest["mel_filterbank_shape"] = list(fb.shape)
    manifest["hann_window_shape"] = list(win.shape)
    # self-contained lexicon copy
    (CONF / "assets" / "tokens.txt").write_text((REPO / "data" / "lang" / "tokens.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (CONF / "assets" / "ayah_phonemes.json").write_text((REPO / "data" / "lang" / "ayah_phonemes.json").read_text(encoding="utf-8"), encoding="utf-8")

    # --- helpers ---
    def to16k(path):
        w, sr = sf.read(path, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        if sr != SAMPLE_RATE:
            import torchaudio
            w = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(torch.from_numpy(w)).numpy()
        return np.ascontiguousarray(w, dtype=np.float32)

    def decode(win_audio):
        feats = logmel_16k(torch.from_numpy(np.ascontiguousarray(win_audio))).unsqueeze(0).to(device)
        with torch.no_grad():
            lp, ol = model(feats, torch.tensor([feats.shape[1]], device=device))
        ids = lp.cpu()[0, :int(ol[0])].argmax(-1).tolist()
        out, prev = [], -1
        for s in ids:
            if s != prev and s != 0:
                out.append(id2tok[s])
            prev = s
        return out

    # --- 1) front-end fixtures (WAV -> log-mel) ---
    import pandas as pd
    df = pd.read_csv(REPO / "data" / "raw" / "audio" / "manifest.csv")
    fe_sources = []
    for sid, aid, nm in [(112, 1, "studio_ikhlas1_short"), (78, 1, "studio_naba1"),
                         (96, 1, "studio_alaq1_long")]:
        row = df.query(f"surah_id=={sid} and ayah_id=={aid}").iloc[0]
        fe_sources.append((nm, to16k(row["path"])))
    user_wav = CONF.parent / "demo" / "test_fixtures" / "user_114_quietmic.wav"
    if user_wav.exists():
        fe_sources.append(("user_114_quietmic", to16k(str(user_wav))[:int(6 * SAMPLE_RATE)]))
    # synthetic edge cases
    fe_sources.append(("silence_1s", np.zeros(SAMPLE_RATE, dtype=np.float32)))
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    fe_sources.append(("sine_440hz", (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)))

    for nm, wav in fe_sources:
        sf.write(CONF / "fixtures" / "frontend" / f"{nm}.wav", wav, SAMPLE_RATE, subtype="FLOAT")
        lm = logmel_16k(torch.from_numpy(wav)).numpy()       # [T, 80]
        save_f32(lm, CONF / "golden" / "frontend" / f"{nm}.logmel.bin")
        manifest["frontend"].append({"name": nm, "wav": f"fixtures/frontend/{nm}.wav",
                                     "logmel": f"golden/frontend/{nm}.logmel.bin",
                                     "logmel_shape": list(lm.shape)})

    # --- 2) matcher/segmenter fixtures (window phonemes -> ayah events) ---
    seg_cfg = dict(window_s=4.0, hop_s=1.0, max_cost=0.30,
                   context=dict(bonus=0.22, window=2, surah_bonus=0.10, streak_bonus=0.05))

    def make_matcher_fixture(name, windows):
        seq = SequentialContext(list(trie.key_to_node.keys()), **seg_cfg["context"])
        seg = SlidingWindowSegmenter(None, seq, ap, max_cost=seg_cfg["max_cost"])
        events = []
        for i, ph in enumerate(windows):
            ev = seg.process(ph, float(i))
            if ev:
                events.append(ev)
        (CONF / "fixtures" / "matcher" / f"{name}.json").write_text(
            json.dumps({"windows": windows, "config": seg_cfg}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        (CONF / "golden" / "matcher" / f"{name}.events.json").write_text(
            json.dumps({"events": events}, ensure_ascii=False, indent=1), encoding="utf-8")
        manifest["matcher"].append({"name": name, "windows": f"fixtures/matcher/{name}.json",
                                    "events": f"golden/matcher/{name}.events.json",
                                    "n_windows": len(windows), "n_events": len(events)})
        return events

    # 2a) real: the user's quiet-mic continuous 114 session -> per-window phonemes
    if user_wav.exists():
        audio = to16k(str(user_wav))
        W, H = int(4.0 * SAMPLE_RATE), int(1.0 * SAMPLE_RATE)
        windows, start = [], 0
        while start < len(audio):
            w = audio[start:start + W]
            windows.append(decode(w) if len(w) >= int(0.5 * SAMPLE_RATE) else [])
            start += H
        evs = make_matcher_fixture("user_114_session", windows)
        print(f"  matcher/user_114_session: {len(windows)} windows -> events "
              f"{[e['ayah'] for e in evs]}")

    # 2b) clean synthetic: G2P phonemes of 108:1,2,3 as a few windows each (deterministic)
    clean_windows = []
    for key in ["108:1", "108:1", "108:2", "108:2", "108:3", "108:3"]:
        clean_windows.append(ap[key])
    evs = make_matcher_fixture("clean_108_sequence", clean_windows)
    print(f"  matcher/clean_108_sequence: events {[e['ayah'] for e in evs]}")

    # --- 2c) inference golden: log-mel -> ONNX -> greedy phoneme ids (for test_inference) ---
    # Uses the fp32 export (desktop ORT can't run the int8 ConvInteger op on some versions;
    # on-device int8 is validated on the target ORT). Same-model comparison validates the
    # C++ ORT session + CTC greedy decode.
    onnx_fp32 = REPO / "export" / "onnx" / "model.onnx"
    if onnx_fp32.exists():
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_fp32), providers=["CPUExecutionProvider"])
        FT = sess.get_inputs()[0].shape[1]
        (CONF / "golden" / "inference").mkdir(parents=True, exist_ok=True)
        for fx in manifest["frontend"]:
            lm = np.fromfile(CONF / fx["logmel"], dtype="<f4").reshape(fx["logmel_shape"])
            v = min(lm.shape[0], FT)
            feats = np.zeros((1, FT, 80), dtype=np.float32); feats[0, :v] = lm[:v]
            lp, ol = sess.run(None, {"features": feats, "lengths": np.array([v], dtype=np.int64)})
            seq = lp[0, :int(ol[0])].argmax(-1).tolist()
            out, prev = [], -1
            for s in seq:
                if s != prev and s != 0: out.append(s)
                prev = s
            (CONF / "golden" / "inference" / f"{fx['name']}.phonemes.txt").write_text(
                " ".join(map(str, out)), encoding="utf-8")
        manifest["inference"] = {"model": "export/onnx/model.onnx (fp32)",
                                 "note": "ids per golden/inference/<name>.phonemes.txt; run test_inference"}
        print(f"  inference golden: {len(manifest['frontend'])} clips (from fp32 ONNX)")

    # --- 3) edit-distance unit cases ---
    edit_cases = []
    pairs = [(ap["114:2"], ap["114:2"]), (ap["114:2"], ap["88:10"]),
             (ap["112:1"], ap["112:2"]), (["m", "a", "l"], ["m", "a", "l", "i", "k"])]
    for a, b in pairs:
        edit_cases.append({"a": a, "b": b, "norm": round(_edit_norm(a, b), 6)})
    (CONF / "assets" / "edit_cases.json").write_text(json.dumps(edit_cases, ensure_ascii=False, indent=1), encoding="utf-8")

    (CONF / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote conformance fixtures -> {CONF}")
    print(f"  frontend: {len(manifest['frontend'])} clips, matcher: {len(manifest['matcher'])} sessions, "
          f"edit cases: {len(edit_cases)}")


if __name__ == "__main__":
    main()
