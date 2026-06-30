#!/usr/bin/env python3
"""
End-to-end ayah-ID evaluation: Stage 1 (Emformer+CTC) -> Stage 2 (fuzzy matcher).

  audio -> log-mel -> model -> CTC greedy phoneme stream -> matcher -> ranked ayat

Reports top-1 / top-3 ayah-ID accuracy, mean time-to-detection (fraction of
phonemes until the true ayah first reaches top-1), and false-commit rate.

  python eval/evaluate.py --checkpoint training/exp/best.pt --split val --limit 300
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
from phoneme_matcher import PhonemeTrie, PhonemeMatcher, CommitTracker   # noqa: E402


def greedy_phonemes(log_probs: torch.Tensor, length: int, id2tok: dict[int, str]) -> list[str]:
    ids = log_probs[:length].argmax(dim=-1).tolist()
    out, prev = [], -1
    for s in ids:
        if s != prev and s != 0:
            out.append(id2tok[s])
        prev = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="training/exp/best.pt")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--retasy", action="store_true",
                    help="evaluate on the full RetaSy learner set instead of a clean split")
    ap.add_argument("--manifest", default=None,
                    help="evaluate on a custom manifest (e.g. phase-2 held-out retasy_test.csv)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--commit-margin", type=float, default=0.15,
                    help="norm-cost gap for a confident commit")
    ap.add_argument("--persistence", type=int, default=5,
                    help="consecutive phonemes the leader must hold the margin (tuned default)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok2id = load_tokens()
    id2tok = {v: k for k, v in tok2id.items()}

    ckpt = torch.load(REPO / args.checkpoint, map_location=device)
    model = EmformerCTC(num_tokens=ckpt["vocab"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch')}, "
          f"val_PER {ckpt.get('val_per'):.3f})")

    trie = PhonemeTrie.from_ayah_phonemes(load_ayah_phonemes())
    print(f"matcher trie: {trie.num_nodes()} nodes")

    if args.manifest:
        ds = AyahDataset(None, manifest_csv=args.manifest)
        print(f"custom-manifest eval: {len(ds)} clips from {args.manifest}")
    elif args.retasy:
        # Learner set: exclude clips whose audio doesn't match the labeled ayah.
        BAD = {"not_related_quran", "not_match_aya", "multiple_aya", "in_complete"}
        retasy_manifest = REPO / "data" / "raw" / "retasy_audio" / "manifest.csv"
        ds = AyahDataset(None, manifest_csv=retasy_manifest,
                         row_filter=lambda df: ~df["final_label"].isin(BAD))
        print(f"RetaSy learner eval: {len(ds)} clips (excluded {BAD})")
    else:
        ds = AyahDataset(args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=0)

    n = top1 = top3 = committed = false_commit = 0
    ttd_sum, ttd_cnt = 0.0, 0
    limit = args.limit or len(ds)

    with torch.no_grad():
        for batch in dl:
            feats = batch["features"].to(device)
            flens = batch["feature_lengths"].to(device)
            log_probs, out_lens = model(feats, flens)
            log_probs = log_probs.cpu()
            out_lens = out_lens.cpu()

            for b in range(feats.size(0)):
                if n >= limit:
                    break
                true_sid, true_aid = batch["ayah_ids"][b]
                true_key = f"{true_sid}:{true_aid}"
                phons = greedy_phonemes(log_probs[b], int(out_lens[b]), id2tok)

                m = PhonemeMatcher(trie, allow_restart=False)
                tracker = CommitTracker(threshold=args.commit_margin, persistence=args.persistence)
                first_top1 = None
                commit_key = None
                for i, p in enumerate(phons, 1):
                    cands = m.step(p)
                    if cands and cands[0].key == true_key and first_top1 is None:
                        first_top1 = i
                    if commit_key is None:
                        top, margin = m.commit_margin()
                        commit_key = tracker.update(top, margin)
                cands = m.candidates(k=3)
                keys = [c.key for c in cands]

                n += 1
                top1 += true_key == (keys[0] if keys else None)
                top3 += true_key in keys[:3]
                if phons and first_top1 is not None:
                    ttd_sum += first_top1 / len(phons)
                    ttd_cnt += 1

                if commit_key is not None:
                    committed += 1
                    false_commit += commit_key != true_key
            if n >= limit:
                break

    print(f"\nEvaluated {n} clips on {args.split}")
    print(f"  top-1 ayah accuracy : {top1/n:.1%}")
    print(f"  top-3 ayah accuracy : {top3/n:.1%}")
    if ttd_cnt:
        print(f"  mean time-to-detect : {ttd_sum/ttd_cnt:.0%} of phonemes "
              f"({ttd_cnt}/{n} ever reached top-1)")
    print(f"  commit rate         : {committed/n:.1%} "
          f"(margin>={args.commit_margin}, persistence={args.persistence})")
    if committed:
        print(f"  false-commit rate   : {false_commit/committed:.1%} of commits")


if __name__ == "__main__":
    main()
