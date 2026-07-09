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

def gen_chain(CONF):
    """Golden for the unit-chain decoder (the research 'winning design': multi-scale
    matched-filter windows + 3-gram retrieval + infix scoring + blended selection ->
    successor votes + twin substitution -> 2-deep deferral assembly).

    Model-independent: streams are synthetic (ground-truth unit phonemes, deterministic
    times, seeded noise), so the golden regenerates anywhere. The C++ port must match
    emitted units (key + fire time) and the assembled chain EXACTLY.
    Returns the manifest fragment.
    """
    sys.path.insert(0, str(REPO / "research"))
    import random
    from chain_sliding import (decode_sliding, build_ngram_index, make_succ_full,
                               assemble)

    (CONF / "assets").mkdir(parents=True, exist_ok=True)
    (CONF / "fixtures" / "chain").mkdir(parents=True, exist_ok=True)
    (CONF / "golden" / "chain").mkdir(parents=True, exist_ok=True)

    ayah_ph = {k: v.split() for k, v in json.loads(
        (REPO / "data/lang/ayah_phonemes.json").read_text(encoding="utf-8")).items()}
    seg_raw = json.loads((REPO / "data/lang/segment_phonemes.json").read_text(encoding="utf-8"))
    refs = {k: v["phonemes"].split() for k, v in seg_raw.items()}
    segmented = {k.split("#")[0] for k in refs}
    refs.update({k: v for k, v in ayah_ph.items() if k not in segmented})
    # self-contained flat unit lexicon for the C++ port
    (CONF / "assets" / "unit_phonemes.json").write_text(
        json.dumps({k: " ".join(v) for k, v in refs.items()}, ensure_ascii=False, indent=0),
        encoding="utf-8")
    ref_lens = {k: len(v) for k, v in refs.items()}
    ngram_idx = build_ngram_index(refs)
    succ_full = make_succ_full(refs)
    vocab = sorted({p for v in refs.values() for p in v})

    PH_SEC = 0.26            # realistic decoded phoneme rate (~3.8 ph/s)
    GAP_SEC = 0.5            # inter-unit pause

    def units_of(ayah):
        n = max((int(k.split("#")[1]) for k in refs if k.startswith(ayah + "#")), default=0)
        return [f"{ayah}#{i:02d}" for i in range(1, n + 1)] if n else [ayah]

    def synth(ayat, sub=0.0, dele=0.0, seed=0, junk_after=None, junk_len=12):
        """Ground-truth phoneme stream for consecutive ayat, seeded noise; optional junk
        block after unit index `junk_after`. Also emits per-phoneme posterior `alts` (Phase 0):
        a substituted position keeps the CORRECT phoneme as a strong 2nd alternative, so the
        Phase-2 soft-scoring path (sub_min < 1) can recover it — this is what makes the
        soft_score_run fixture diverge from greedy."""
        rng = random.Random(seed)
        phons, times, alts = [], [], []
        t = 0.0
        for ui, u in enumerate([x for a in ayat for x in units_of(a)]):
            for p in refs[u]:
                if dele and rng.random() < dele:
                    continue
                if sub and rng.random() < sub:
                    s = rng.choice(vocab)
                    phons.append(s); alts.append([[s, 0.55], [p, 0.44]])   # correct = strong 2nd
                else:
                    phons.append(p); alts.append([[p, 0.9]])
                times.append(round(t, 4))
                t += PH_SEC
            t += GAP_SEC
            if junk_after is not None and ui == junk_after:
                for _ in range(junk_len):
                    j = rng.choice(vocab)
                    phons.append(j); alts.append([[j, 0.6]]); times.append(round(t, 4))
                    t += PH_SEC
                t += GAP_SEC
        return {"phonemes": phons, "times": times, "alts": alts}

    params = dict(window_s=10.0, hop_s=1.5, cost=0.30, votes_next=1, votes_jump=2)
    scenarios = {
        # clean multi-segment chaining across an ayah boundary
        "clean_seg_run": synth(["2:6", "2:7", "2:8"]),
        # decode-error robustness: substitutions + deletions
        "noisy_sub_del": synth(["2:30", "2:31"], sub=0.08, dele=0.08, seed=7),
        # exact-twin context resolution (104:4#01 'kalla' is a cross-surah twin)
        "twin_context_run": synth(["104:3", "104:4", "104:5"], sub=0.05, seed=3),
        # junk block between two true units: the 2-deep assembly must bridge it
        "junk_sandwich": synth(["3:5", "3:6", "3:7"], junk_after=1, seed=11),
        # context-gated EARLY detection: expected units fire from a >=50% prefix match
        "early_prefix_run": synth(["2:6", "2:7", "2:8"], sub=0.05, seed=5),
        # Phase-2 posterior-aware SCORING: heavy substitutions, but the correct phoneme is a
        # strong 2nd alt at each — soft scoring (sub_min 0.0) recovers units greedy rejects.
        "soft_score_run": synth(["2:6", "2:7", "2:8"], sub=0.25, seed=9),
    }
    extra_params = {"early_prefix_run": {"early_prefix": 0.5},
                    "soft_score_run": {"sub_min": 0.0}}

    entries = []
    for name, stream in scenarios.items():
        pp = dict(params, **extra_params.get(name, {}))
        emitted = decode_sliding(stream, ngram_idx, refs, pp["window_s"],
                                 pp["hop_s"], pp["cost"], pp["votes_next"],
                                 pp["votes_jump"], ref_lens=ref_lens,
                                 use_twin_sub=True, succ_fn=succ_full,
                                 early_prefix=pp.get("early_prefix"),
                                 sub_min=pp.get("sub_min", 1.0))
        # golden pins the emitted KEY SEQUENCE + assembled chain (exact).
        chain = assemble(emitted, succ_full)
        (CONF / "fixtures" / "chain" / f"{name}.json").write_text(
            json.dumps({"stream": stream, "params": pp}, ensure_ascii=False, indent=0),
            encoding="utf-8")
        (CONF / "golden" / "chain" / f"{name}.chain.json").write_text(
            json.dumps({"emitted": emitted, "assembled": chain},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        entries.append({"name": name, "stream": f"fixtures/chain/{name}.json",
                        "chain": f"golden/chain/{name}.chain.json",
                        "n_phonemes": len(stream["phonemes"]), "n_emitted": len(emitted)})
        print(f"  chain/{name}: {len(stream['phonemes'])} ph -> emitted {emitted} "
              f"-> chain {chain}")
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
    import argparse
    apar = argparse.ArgumentParser()
    apar.add_argument("--only", choices=["chain", "highlight"], default=None,
                      help="regenerate just one model-independent section (updates manifest in place)")
    args = apar.parse_args()
    if args.only:
        man_path = CONF / "manifest.json"
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        manifest[args.only] = gen_chain(CONF) if args.only == "chain" else gen_highlight(CONF)
        man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"updated manifest section: {args.only}")
        return

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

    # --- 0) highlight + chain golden (model-independent — do them first) ---
    manifest["highlight"] = gen_highlight(CONF)
    manifest["chain"] = gen_chain(CONF)

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
    # Silero VAD model (paused ayah-by-ayah boundary reset) — copied from the pip package so the
    # C++ core + Android build have it; .onnx is gitignored, so this reproduces it on a fresh tree.
    try:
        import silero_vad, shutil
        src = Path(silero_vad.__file__).parent / "data" / "silero_vad_16k_op15.onnx"
        shutil.copyfile(src, CONF / "assets" / "silero_vad.onnx")
    except Exception as e:  # noqa: BLE001 — optional; VAD degrades to off if absent
        print(f"  (silero VAD not copied: {e}; paused-recitation reset will be disabled)")

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
