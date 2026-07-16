#!/usr/bin/env python3
"""
Calibration-probe prototype (de-risking in-app launch tuning).

Question: if the app asked the user to recite a KNOWN passage once at first launch, could it
auto-pick per-user `normRms` + `chainCost` that match the values we hand-tuned globally
(normRms 0.15 / chainCost 0.45)? And is there per-user headroom, or is the global already best?

Mechanism (faithful to what the app would do): for a passage whose truth ayat we KNOW, decode at
several normRms candidates and measure the focused decode cost of each TRUE ayah (infix match of
its reference phonemes against the decoded stream — the same "truth cost" used in the research
notes). Then:
    normRms*   = argmin over candidates of the MEDIAN true-ayah cost   (best front-end gain)
    chainCost* = upper quantile of the true-ayah costs at normRms* + margin (fire threshold that
                 lets the user's true units through, below the ~0.5 junk floor)

We have no Al-Fatiha enrollment clip, so each real labeled phone session stands in as the
"enrollment passage" (we know its truth ayat). Reports per-session picks + whether they cluster
at the shipped globals, and how much the per-session-best normRms beats a fixed 0.15.

Run: research/calib_probe.py   (GPU; ~1 min over the labeled sessions)
"""
import sys, json, statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "research"))

from data import load_wav_16k, normalize_rms, _mel, SAMPLE_RATE, LOG_FLOOR, load_tokens  # noqa: E402
from model import EmformerCTC  # noqa: E402
from chain_sliding import greedy_with_alts, _infix_norm  # noqa: E402

NORMRMS_CANDS = [0.05, 0.10, 0.15, 0.20, 0.25]
GLOBAL_NORMRMS, GLOBAL_COST = 0.15, 0.45
ENC_FRAME_SEC = 0.04
CKPT = REPO / "training/exp/best_full_tu.pt"


def logmel_norm(wav: torch.Tensor, target: float) -> torch.Tensor:
    """logmel_16k, but RMS-normalized to an arbitrary target (the front-end knob under test)."""
    wav = normalize_rms(wav, target=target)
    mel = _mel(SAMPLE_RATE)(wav)
    return torch.log(torch.clamp(mel, min=LOG_FLOOR)).transpose(0, 1).contiguous()


def expand_truth(s: str) -> list[str]:
    out = []
    for part in str(s).split():
        if ":" not in part:
            continue
        sur, rng = part.split(":")
        if "-" in rng:
            a, b = rng.split("-")
            out += [f"{sur}:{i}" for i in range(int(a), int(b) + 1)]
        else:
            out.append(f"{sur}:{rng}")
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=dev)
    model = EmformerCTC(num_tokens=ck["vocab"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    id2tok = {v: k for k, v in load_tokens().items()}
    refs = {k: v.split() for k, v in json.load(open(REPO / "data/lang/ayah_phonemes.json")).items()}

    labels = pd.read_csv(REPO / "data/raw/audio_bench/real/labels.csv")
    sessdir = REPO / "data/raw/audio_bench/real/sessions"
    print(f"model {CKPT.name} (val_PER {ck['val_per']:.4f}) on {dev} | "
          f"normRms candidates {NORMRMS_CANDS} | global ({GLOBAL_NORMRMS}, {GLOBAL_COST})\n")

    rows = []
    for _, r in labels.iterrows():
        wavp = sessdir / r["file"]
        if not wavp.exists():
            continue
        truth = [t for t in expand_truth(r["truth"]) if t in refs]
        if not truth:
            continue
        wav = load_wav_16k(str(wavp))
        raw_rms = float(wav.pow(2).mean().sqrt())
        percand = {}
        for cand in NORMRMS_CANDS:
            feats = logmel_norm(wav, cand).unsqueeze(0).to(dev)
            lens = torch.tensor([feats.shape[1]]).to(dev)
            with torch.no_grad():
                lp, ol = model(feats, lens)
            phons, _t, _a = greedy_with_alts(lp[0].float().cpu(), int(ol[0]), id2tok, ENC_FRAME_SEC)
            percand[cand] = [_infix_norm(refs[t], phons) for t in truth]
        med = {c: statistics.median(v) for c, v in percand.items()}
        best_nr = min(med, key=med.get)
        bc = percand[best_nr]
        gc = percand[GLOBAL_NORMRMS]
        rows.append(dict(
            file=r["file"], truth=r["truth"], n=len(truth), raw_rms=raw_rms, med=med,
            best_nr=best_nr, best_med=med[best_nr], global_med=med[GLOBAL_NORMRMS],
            p50=float(np.percentile(bc, 50)), p75=float(np.percentile(bc, 75)),
            p90=float(np.percentile(bc, 90)),
            cost_star=round(float(np.percentile(bc, 75)) + 0.05, 2),
            fire_global=sum(c <= GLOBAL_COST for c in gc), fire_best=sum(c <= GLOBAL_COST for c in bc),
        ))

    # ---- per-session table ----
    hdr = f"{'session':<26}{'truth':<11}{'rawRMS':>7}  " + "".join(f"nr{c:<5}" for c in NORMRMS_CANDS) + \
          f"{'nr*':>5}{'p50':>6}{'p75':>6}{'p90':>6}{'cost*':>7}{'fire@.45':>9}"
    print(hdr); print("-" * len(hdr))
    for x in rows:
        mark = lambda c: ("*" if c == x["best_nr"] else " ")  # noqa: E731
        meds = "".join(f"{x['med'][c]:.2f}{mark(c)}  " for c in NORMRMS_CANDS)
        print(f"{x['file']:<26}{x['truth']:<11}{x['raw_rms']:>7.3f}  {meds}"
              f"{x['best_nr']:>5}{x['p50']:>6.2f}{x['p75']:>6.2f}{x['p90']:>6.2f}"
              f"{x['cost_star']:>7.2f}{x['fire_best']}/{x['n']:<7}")

    # ---- aggregates: the de-risking answers ----
    print("\n=== calibration verdict ===")
    from collections import Counter
    nrc = Counter(x["best_nr"] for x in rows)
    print(f"best normRms* distribution: {dict(sorted(nrc.items()))}  "
          f"(global default {GLOBAL_NORMRMS})")
    at_global = sum(x["best_nr"] == GLOBAL_NORMRMS for x in rows)
    print(f"sessions whose best normRms == 0.15: {at_global}/{len(rows)}")
    gain = statistics.mean(x["global_med"] - x["best_med"] for x in rows)
    print(f"mean true-cost improvement of per-session normRms* vs fixed 0.15: {gain:+.3f} "
          f"(median true cost; lower=better)")
    allp75 = [x["p75"] for x in rows]; allp90 = [x["p90"] for x in rows]
    allstar = [x["cost_star"] for x in rows]
    print(f"true-cost p75 across sessions: min {min(allp75):.2f} / med {statistics.median(allp75):.2f} "
          f"/ max {max(allp75):.2f}   (global chainCost {GLOBAL_COST})")
    print(f"true-cost p90 across sessions: min {min(allp90):.2f} / med {statistics.median(allp90):.2f} "
          f"/ max {max(allp90):.2f}")
    print(f"auto-picked cost* (p75+0.05): min {min(allstar):.2f} / med {statistics.median(allstar):.2f} "
          f"/ max {max(allstar):.2f}")
    fg = sum(x["fire_global"] for x in rows); fb = sum(x["fire_best"] for x in rows)
    tot = sum(x["n"] for x in rows)
    print(f"true units firing @ cost 0.45: global-normRms {fg}/{tot}, best-normRms {fb}/{tot}")


if __name__ == "__main__":
    main()
