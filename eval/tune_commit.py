#!/usr/bin/env python3
"""
Commit-threshold sweep for the Stage-2 matcher.

The matcher commits an ayah when the top-1/runner-up norm-cost margin clears a
threshold. Too low -> early but wrong commits; too high -> late or no commit.

This runs the full pipeline ONCE per clip, recording the streaming margin
trajectory (per phoneme: top-1 key, margin), then evaluates a grid of thresholds
cheaply. For each threshold it reports, on the committed clips:
  - commit rate     : fraction of clips that ever commit
  - commit accuracy : of commits, fraction that are the correct ayah (precision)
  - false-commit    : 1 - commit accuracy
  - latency         : mean fraction of phonemes seen at commit time (lower=earlier)

  python eval/tune_commit.py --checkpoint training/exp/best_mic.pt --split test
  python eval/tune_commit.py --checkpoint training/exp/best_mic.pt \
         --manifest data/raw/phase2/retasy_test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "matcher"))

from data import AyahDataset, collate, load_tokens, load_ayah_phonemes   # noqa: E402
from model import EmformerCTC                                            # noqa: E402
from phoneme_matcher import PhonemeTrie, PhonemeMatcher                  # noqa: E402

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
PERSISTENCE = [3, 5, 8, 12, 16, 24]   # consecutive phonemes the leader must hold the margin
# Grid extended upward for the surah 1-3 expansion: Al-Baqarah's long shared prefixes
# ("ya ayyuha alladhina amanu...") hold a false margin far longer than any Juz-Amma ayah,
# so the old T=0.15/K=5 committed before divergence (59.7% false commits on the s1-3 test).
INF = float("inf")


def first_commit(traj, T, K):
    """traj: list of (top_key, margin). Commit when the SAME top_key holds margin>=T
    for K consecutive steps. Returns (commit_index, committed_key) or (None, None)."""
    run = 0
    for i, (key, margin) in enumerate(traj):
        prev_key = traj[i - 1][0] if i else None
        if margin >= T and key is not None and (run == 0 or key == prev_key):
            run += 1
        elif margin >= T and key is not None:
            run = 1   # leader changed but still above T -> restart the run
        else:
            run = 0
        if run >= K:
            return i, key
    return None, None


def greedy_phonemes(log_probs, length, id2tok):
    ids = log_probs[:length].argmax(dim=-1).tolist()
    out, prev = [], -1
    for s in ids:
        if s != prev and s != 0:
            out.append(id2tok[s])
        prev = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="training/exp/best_mic.pt")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--manifest", default=None, help="custom manifest (e.g. retasy_test.csv)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok2id = load_tokens()
    id2tok = {v: k for k, v in tok2id.items()}
    ckpt = torch.load(REPO / args.checkpoint, map_location=device)
    model = EmformerCTC(num_tokens=ckpt["vocab"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    trie = PhonemeTrie.from_ayah_phonemes(load_ayah_phonemes())

    if args.manifest:
        ds = AyahDataset(None, manifest_csv=args.manifest)
        src = args.manifest
    else:
        ds = AyahDataset(args.split)
        src = args.split
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    limit = args.limit or len(ds)
    print(f"{args.checkpoint} on {src}: recording margin trajectories over <= {limit} clips")

    # Phase 1: record per-clip streaming trajectory of (top_key, margin).
    trajectories = []   # (true_key, n_steps, [(top_key, margin), ...])
    n = 0
    with torch.no_grad():
        for batch in dl:
            feats = batch["features"].to(device)
            flens = batch["feature_lengths"].to(device)
            log_probs, out_lens = model(feats, flens)
            log_probs, out_lens = log_probs.cpu(), out_lens.cpu()
            for b in range(feats.size(0)):
                if n >= limit:
                    break
                true_key = "{}:{}".format(*batch["ayah_ids"][b])
                phons = greedy_phonemes(log_probs[b], int(out_lens[b]), id2tok)
                m = PhonemeMatcher(trie, allow_restart=False)
                traj = []
                for p in phons:
                    m.step(p)
                    top, margin = m.commit_margin()
                    traj.append((top.key if top else None, margin if margin != INF else 999.0))
                trajectories.append((true_key, len(phons), traj))
                n += 1
            if n >= limit:
                break

    # Phase 2: sweep (threshold, persistence) over recorded trajectories.
    print(f"\nEvaluated {n} clips. Commit = same top-1 holds margin>=T for K phonemes.\n")
    print(f"{'T':>5} {'K':>3} {'commit%':>8} {'commit_acc':>11} {'false%':>8} {'latency':>8}")
    for T in THRESHOLDS:
        for K in PERSISTENCE:
            correct, lat = [], []
            for true_key, steps, traj in trajectories:
                idx, key = first_commit(traj, T, K)
                if idx is not None:
                    correct.append(key == true_key)
                    lat.append((idx + 1) / max(1, steps))
            if correct:
                acc = sum(correct) / len(correct)
                print(f"{T:>5.2f} {K:>3} {len(correct)/n:>8.1%} {acc:>11.1%} "
                      f"{1-acc:>8.1%} {sum(lat)/len(lat):>8.0%}")
            else:
                print(f"{T:>5.2f} {K:>3} {0.0:>8.1%} {'-':>11} {'-':>8} {'-':>8}")
        print()

    print("Pick the (T,K) with acceptable false% at the lowest latency. Persistence (K) "
          "suppresses transient wrong leaders; latency is the cost.")


if __name__ == "__main__":
    main()
