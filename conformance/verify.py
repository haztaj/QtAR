#!/usr/bin/env python3
"""
Verify a candidate implementation (e.g. the C++ SDK port) against the golden fixtures.

The candidate runs its own front-end / matcher on the fixture inputs and writes outputs
in the SAME format + filenames as golden/ (see conformance/spec.md), into a directory;
then:

  python conformance/verify.py --candidate path/to/candidate_outputs

Front-end log-mel is compared within tolerance; phonemes/events are compared exactly.

Default (no --candidate) runs a SELF-CHECK: recomputes from the Python reference and
confirms it reproduces its own golden — sanity that the harness is deterministic and a
live, runnable example of the expected outputs.

  python conformance/verify.py            # self-check
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CONF = Path(__file__).resolve().parent


def load_f32(path: Path, shape):
    return np.fromfile(path, dtype="<f4").reshape(shape)


def check_frontend(man, get_logmel, tol):
    ok = True
    for fx in man["frontend"]:
        gold = load_f32(CONF / fx["logmel"], fx["logmel_shape"])
        cand = get_logmel(fx)
        if cand is None:
            print(f"  [frontend] {fx['name']:24} MISSING candidate output"); ok = False; continue
        if cand.shape != gold.shape:
            print(f"  [frontend] {fx['name']:24} SHAPE {cand.shape} != {tuple(gold.shape)}"); ok = False; continue
        d = float(np.abs(cand - gold).max())
        good = d <= tol
        ok &= good
        print(f"  [frontend] {fx['name']:24} max_abs_diff={d:.2e}  {'OK' if good else 'FAIL (tol %.0e)'%tol}")
    return ok


def check_matcher(man, get_events):
    ok = True
    for fx in man["matcher"]:
        gold = json.loads((CONF / fx["events"]).read_text(encoding="utf-8"))["events"]
        cand = get_events(fx)
        if cand is None:
            print(f"  [matcher]  {fx['name']:24} MISSING candidate output"); ok = False; continue
        gkeys = [(e["event"], e["ayah"]) for e in gold]
        ckeys = [(e["event"], e["ayah"]) for e in cand]
        good = gkeys == ckeys
        ok &= good
        print(f"  [matcher]  {fx['name']:24} {'OK' if good else 'FAIL'}  "
              f"gold={[a for _,a in gkeys]} cand={[a for _,a in ckeys]}")
    return ok


def check_highlight(man, get_states):
    """State-snapshot sequences are compared EXACTLY (the SDK output contract)."""
    ok = True
    for fx in man.get("highlight", []):
        gold = json.loads((CONF / fx["states"]).read_text(encoding="utf-8"))["states"]
        cand = get_states(fx)
        if cand is None:
            print(f"  [highlight] {fx['name']:24} MISSING candidate output"); ok = False; continue
        good = cand == gold
        ok &= good
        tail = gold[-1] if gold else {}
        print(f"  [highlight] {fx['name']:24} {'OK' if good else 'FAIL'}  "
              f"final(active={tail.get('active')}, pending={tail.get('pending')})")
    return ok


def self_check(man):
    """Recompute from the Python reference and compare to golden."""
    sys.path.insert(0, str(CONF.parent / "training"))
    sys.path.insert(0, str(CONF.parent / "matcher"))
    sys.path.insert(0, str(CONF.parent / "demo"))
    import soundfile as sf, torch
    from data import logmel_16k
    from phoneme_matcher import PhonemeTrie, SequentialContext
    from sliding import SlidingWindowSegmenter
    from highlight_controller import HighlightController
    ap = json.loads((CONF / "assets" / "ayah_phonemes.json").read_text(encoding="utf-8"))
    ap = {k: v.split() for k, v in ap.items()}
    trie = PhonemeTrie.from_ayah_phonemes(ap)

    def get_logmel(fx):
        w, _ = sf.read(CONF / fx["wav"], dtype="float32")
        return logmel_16k(torch.from_numpy(np.ascontiguousarray(w))).numpy()

    def get_events(fx):
        spec = json.loads((CONF / fx["windows"]).read_text(encoding="utf-8"))
        c = spec["config"]["context"]
        seq = SequentialContext(list(trie.key_to_node.keys()), **c)
        seg = SlidingWindowSegmenter(None, seq, ap, max_cost=spec["config"]["max_cost"])
        out = []
        for i, ph in enumerate(spec["windows"]):
            ev = seg.process(ph, float(i))
            if ev:
                out.append(ev)
        return out

    def get_states(fx):
        hc = HighlightController(CONF.parent / "data" / "lang" / "ambiguous_ayat.json")
        steps = json.loads((CONF / fx["steps"]).read_text(encoding="utf-8"))["steps"]
        out = []
        for s in steps:
            snap = hc.detect(s["detect"]) if "detect" in s else hc.choose(s["choose"])
            out.append(snap.to_dict())
        return out

    return get_logmel, get_events, get_states


def candidate_loaders(cand_dir: Path, man):
    def get_logmel(fx):
        p = cand_dir / Path(fx["logmel"]).name
        return load_f32(p, fx["logmel_shape"]) if p.exists() else None

    def get_events(fx):
        p = cand_dir / Path(fx["events"]).name
        return json.loads(p.read_text(encoding="utf-8")).get("events") if p.exists() else None

    def get_states(fx):
        p = cand_dir / Path(fx["states"]).name
        return json.loads(p.read_text(encoding="utf-8")).get("states") if p.exists() else None

    return get_logmel, get_events, get_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=None, help="dir of candidate outputs; omit for self-check")
    args = ap.parse_args()
    man = json.loads((CONF / "manifest.json").read_text(encoding="utf-8"))
    tol = man["tolerances"]["logmel_max_abs"]

    if args.candidate:
        print(f"Verifying candidate: {args.candidate}")
        gl, ge, gs = candidate_loaders(Path(args.candidate), man)
    else:
        print("SELF-CHECK (Python reference vs its own golden):")
        gl, ge, gs = self_check(man)

    ok = check_frontend(man, gl, tol)
    ok &= check_matcher(man, ge)
    ok &= check_highlight(man, gs)
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
