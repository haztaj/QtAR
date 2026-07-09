"""Build NOISE-AUGMENTED decode caches for the posterior-aware-matching eval.

The clean test decodes run ~10% PER, where retrieval is already saturated (Phase 1 was
neutral). To measure whether posterior-aware SCORING (Phase 2) helps in the DEPLOYMENT
regime, decode the test clips through the phase-2 phone-channel augmentation (mic band,
noise, codec) -> ~30% PER, WITH per-phoneme posteriors, using the deployed mic-adapted
model. Writes full_streams_test_noisy.pkl + unseg_streams_test_noisy.pkl.

Deterministic: each clip's augmentation is seeded by its recording_id, so the noisy cache
is reproducible and greedy vs posterior see the exact same audio.

  python research/build_noisy_cache.py
"""
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
for p in ("research", "training", "matcher"):
    sys.path.insert(0, str(REPO / p))

TEST_RECITERS = {"warsh_husary", "warsh_yassin", "yasser_ad_dussary"}
SEG_OUT = REPO / "data/raw/segments/full_streams_test_noisy.pkl"
UNSEG_OUT = REPO / "data/raw/segments/unseg_streams_test_noisy.pkl"
CKPT = REPO / "training/exp/best_s123_mic_clean.pt"
LINEAR_BUDGET, QUAD_BUDGET, ENC = 24000, 1.0e8, 0.04


def main():
    from data import load_wav_16k, logmel_16k, load_tokens
    from model import EmformerCTC
    from augment import build_waveform_augment
    from chain_sliding import greedy_with_alts
    from torch.nn.utils.rnn import pad_sequence

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = EmformerCTC(num_tokens=ck["vocab"]).to(device).eval()
    model.load_state_dict(ck["model"])
    id2tok = {v: k for k, v in load_tokens().items()}
    aug = build_waveform_augment(16000)     # phone-channel chain (mic band, noise, codec)

    manifest = pd.read_csv(REPO / "data/raw/audio/manifest.csv")
    manifest = manifest[manifest.reciter_id.isin(TEST_RECITERS)]
    spans = pd.read_csv(REPO / "data/raw/segments/segment_spans.csv")
    n_seg = spans.groupby("key")["seg_idx"].max().to_dict()
    seg_keys = set(n_seg)

    def augged(path, rid):
        w = load_wav_16k(path).numpy()
        random.seed(abs(hash(rid)) % (2**32)); np.random.seed(abs(hash(rid)) % (2**32))
        return torch.from_numpy(np.ascontiguousarray(aug(samples=w, sample_rate=16000)))

    def decode_rows(rows, seg):
        rows = sorted(rows, key=lambda r: r["duration"])
        out, i = [], 0
        while i < len(rows):
            batch = [rows[i]]; i += 1
            while i < len(rows):
                mx = max(r["duration"] for r in batch + [rows[i]]) * 100
                if (len(batch) + 1) * mx > LINEAR_BUDGET or (len(batch) + 1) * mx * mx > QUAD_BUDGET:
                    break
                batch.append(rows[i]); i += 1
            feats = [logmel_16k(augged(r["path"], r["recording_id"])) for r in batch]
            lens = torch.tensor([f.shape[0] for f in feats])
            padded = pad_sequence(feats, batch_first=True)
            with torch.no_grad():
                if device == "cuda":
                    torch.cuda.empty_cache()
                    with torch.amp.autocast("cuda"):
                        lp, ol = model(padded.to(device), lens.to(device))
                else:
                    lp, ol = model(padded.to(device), lens.to(device))
            lp = lp.float().cpu()
            for b, r in enumerate(batch):
                ph, tm, al = greedy_with_alts(lp[b], int(ol[b]), id2tok, ENC)
                key = f"{r['surah_id']}:{r['ayah_id']}"
                d = {"recording_id": r["recording_id"], "reciter": r["reciter_id"],
                     "key": key, "dur": r["duration"],
                     "phonemes": ph, "times": tm, "alts": al}
                if seg:
                    d["n_segments"] = n_seg[key]
                out.append(d)
            if len(out) % 400 < len(batch):
                print(f"  {'seg' if seg else 'unseg'} {len(out)}/{len(rows)}", flush=True)
        return out

    seg_rows = manifest[[f"{s}:{a}" in seg_keys for s, a in
                         zip(manifest.surah_id, manifest.ayah_id)]].to_dict("records")
    unseg_rows = manifest[[f"{s}:{a}" not in seg_keys for s, a in
                           zip(manifest.surah_id, manifest.ayah_id)]].to_dict("records")
    print(f"noisy decode: {len(seg_rows)} segmented + {len(unseg_rows)} unsegmented clips")

    SEG_OUT.write_bytes(pickle.dumps(decode_rows(seg_rows, True)))
    print(f"cached -> {SEG_OUT}")
    UNSEG_OUT.write_bytes(pickle.dumps(decode_rows(unseg_rows, False)))
    print(f"cached -> {UNSEG_OUT}")


if __name__ == "__main__":
    main()
