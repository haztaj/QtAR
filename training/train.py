#!/usr/bin/env python3
"""
Train the streaming Emformer + CTC phoneme model.

  python training/train.py --smoke          # few steps, verify GPU/AMP path
  python training/train.py --epochs 60      # real run

Mixed precision + gradient accumulation (16 GB VRAM friendly).  Watches CTC
loss and greedy phoneme error rate (PER) on val; checkpoints best PER.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Reduce fragmentation OOMs with variable-length batches.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from data import AyahDataset, LengthBucketBatchSampler, collate, load_tokens
from model import EmformerCTC

REPO = Path(__file__).resolve().parent.parent
EXP_DIR = REPO / "training" / "exp"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    """CTC greedy decode: argmax -> collapse repeats -> drop blanks (id 0)."""
    ids = log_probs.argmax(dim=-1)  # [B, T]
    out = []
    for b in range(ids.size(0)):
        seq = ids[b, : lengths[b]].tolist()
        collapsed, prev = [], -1
        for s in seq:
            if s != prev and s != 0:
                collapsed.append(s)
            prev = s
        out.append(collapsed)
    return out


def edit_distance(a: list[int], b: list[int]) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]


def split_targets(flat: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    out, off = [], 0
    for n in lengths.tolist():
        out.append(flat[off:off + n].tolist())
        off += n
    return out


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def lr_at(step: int, warmup: int, total: int, peak: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


# ---------------------------------------------------------------------------

def evaluate(model, loader, ctc, device) -> tuple[float, float]:
    model.eval()
    tot_loss, n_batches = 0.0, 0
    tot_err, tot_len = 0, 0
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            flens = batch["feature_lengths"].to(device)
            with torch.amp.autocast("cuda"):
                log_probs, out_lens = model(feats, flens)
            loss = ctc(log_probs.transpose(0, 1).float(), batch["targets"],
                       out_lens.cpu(), batch["target_lengths"])
            tot_loss += loss.item()
            n_batches += 1
            preds = greedy_decode(log_probs.cpu(), out_lens.cpu())
            refs = split_targets(batch["targets"], batch["target_lengths"])
            for p, r in zip(preds, refs):
                tot_err += edit_distance(p, r)
                tot_len += len(r)
    return tot_loss / max(1, n_batches), tot_err / max(1, tot_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--frame-budget", type=int, default=24000,
                    help="max (batch_size * max_frames) per batch; bounds attention memory")
    ap.add_argument("--quad-budget", type=float, default=1.0e8,
                    help="max (batch_size * max_frames^2); bounds O(T^2) attention memory so "
                         "long-clip batches stay below the WDDM paging cliff (see data.py). "
                         "Lower it when other apps hold significant VRAM.")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--augment", action="store_true", help="phase-2 phone-channel augmentation (train only)")
    ap.add_argument("--noise-dir", default=None, help="optional background-noise corpus dir")
    ap.add_argument("--ir-dir", default=None, help="optional impulse-response corpus dir")
    ap.add_argument("--accum", type=int, default=2, help="grad accumulation steps")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--init-from", default=None, help="checkpoint to warm-start weights (fine-tune)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to FULLY resume (model+optimizer+scaler+step+epoch); "
                         "--epochs stays the TOTAL, training continues from the saved epoch")
    ap.add_argument("--epochs-per-run", type=int, default=0,
                    help="exit (code 75) after this many epochs — the supervisor restarts the "
                         "process to bound the audiomentations RAM leak (grows ~8.5 GB/epoch "
                         "in the MAIN process; worker respawn alone doesn't free it)")
    ap.add_argument("--tag", default="", help="suffix for checkpoint filenames (e.g. _aug)")
    ap.add_argument("--train-manifest", default=None,
                    help="custom training manifest (e.g. phase-2 combined clean+RetaSy); "
                         "uses all its rows instead of the main reciter split")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="few steps to verify path")
    args = ap.parse_args()

    # Line-buffer stdout so a redirected logfile updates live (monitorable mid-run).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device == "cpu":
        print("WARNING: no CUDA — training will be very slow", file=sys.stderr)

    torch.manual_seed(1337)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    vocab = len(load_tokens())
    if args.train_manifest:
        train_ds = AyahDataset(None, manifest_csv=args.train_manifest, max_seconds=args.max_seconds,
                               augment=args.augment, noise_dir=args.noise_dir, ir_dir=args.ir_dir)
    else:
        train_ds = AyahDataset("train", max_seconds=args.max_seconds, augment=args.augment,
                               noise_dir=args.noise_dir, ir_dir=args.ir_dir)
    val_ds = AyahDataset("val", max_seconds=args.max_seconds)  # eval always on clean
    print(f"train={len(train_ds)}  val={len(val_ds)}  vocab={vocab}")

    train_sampler = LengthBucketBatchSampler(train_ds.frame_lengths(), args.frame_budget,
                                             shuffle=True, quad_budget=args.quad_budget)
    val_sampler = LengthBucketBatchSampler(val_ds.frame_lengths(), args.frame_budget,
                                           shuffle=False, quad_budget=args.quad_budget)
    print(f"batches/epoch: train={len(train_sampler)} val={len(val_sampler)}")

    # Augmentation (audiomentations) leaks RAM across epochs; with persistent_workers
    # that accumulates until the OS OOM-kills the run (epoch time grew 80s->245s and
    # death ~epoch 14). Restarting workers each epoch frees it (small respawn cost).
    persist = args.num_workers > 0 and not args.augment
    train_dl = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate,
                          num_workers=args.num_workers, persistent_workers=persist)
    val_dl = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collate,
                        num_workers=args.num_workers, persistent_workers=persist)

    model = EmformerCTC(num_tokens=vocab).to(device)
    print(f"params: {model.num_params() / 1e6:.1f}M")
    if args.init_from:
        ck = torch.load(REPO / args.init_from, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"warm-started from {args.init_from} (epoch {ck.get('epoch')}, "
              f"val_PER {ck.get('val_per')})")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    total_steps = (len(train_dl) // args.accum) * args.epochs
    step, best_per = 0, float("inf")
    start_epoch = 1
    if args.resume:
        ck = torch.load(REPO / args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
        step = ck.get("step", 0)
        best_per = ck.get("best_per", float("inf"))
        start_epoch = ck.get("epoch", 0) + 1
        print(f"resumed from {args.resume} (epoch {ck.get('epoch')}, step {step}, "
              f"best_per {best_per:.3f}, opt={'yes' if 'opt' in ck else 'NO — weights only'})")

    epochs_done = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        run_loss, n_inf = 0.0, 0
        opt.zero_grad(set_to_none=True)

        for bi, batch in enumerate(train_dl):
            feats = batch["features"].to(device)
            flens = batch["feature_lengths"].to(device)
            with torch.amp.autocast("cuda"):
                log_probs, out_lens = model(feats, flens)
            loss = ctc(log_probs.transpose(0, 1).float(), batch["targets"],
                       out_lens.cpu(), batch["target_lengths"])
            n_inf += int(torch.isinf(loss).item()) if loss.numel() == 1 else 0
            scaler.scale(loss / args.accum).backward()

            if (bi + 1) % args.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, args.warmup, total_steps, args.lr)
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1

            run_loss += loss.item()
            if args.smoke and bi >= 6:
                print("smoke: forward/backward/opt path OK on", device)
                val_loss, val_per = evaluate(model, val_dl, ctc, device)
                print(f"smoke: val_loss={val_loss:.3f} val_per={val_per:.3f}")
                return

            if bi % 50 == 0 and device == "cuda":
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  ep{epoch} batch {bi:4d}/{len(train_dl)} "
                      f"loss {loss.item():5.2f} peakGPU {mem:4.1f}GB", flush=True)

        val_loss, val_per = evaluate(model, val_dl, ctc, device)
        if device == "cuda":
            torch.cuda.empty_cache()  # release reserved pool so it can't balloon
        dt = time.time() - t0
        cur_lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch:3d} | train_loss {run_loss/len(train_dl):6.3f} | "
              f"val_loss {val_loss:6.3f} | val_PER {val_per:6.3f} | "
              f"lr {cur_lr:.2e} | {dt:5.1f}s" + (f" | inf={n_inf}" if n_inf else ""))

        if val_per < best_per:
            best_per = val_per
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_per": val_per, "vocab": vocab}, EXP_DIR / f"best{args.tag}.pt")
            print(f"           new best PER {best_per:.3f} -> best{args.tag}.pt")
        # full state for exact resume (supervisor restarts bound the aug RAM leak)
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_per": val_per,
                    "vocab": vocab, "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "best_per": best_per}, EXP_DIR / f"last{args.tag}.pt")

        epochs_done += 1
        if args.epochs_per_run and epochs_done >= args.epochs_per_run and epoch < args.epochs:
            print(f"epochs-per-run reached ({epochs_done}) at epoch {epoch} — "
                  f"exiting for supervisor restart")
            sys.exit(75)

    print(f"done. best val PER {best_per:.3f}")


if __name__ == "__main__":
    main()
