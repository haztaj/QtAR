@echo off
rem Phase-3.1 RESTORE: short low-LR polish of best_s123_p3 on the pure phase-2 manifest
rem (per-ayah + RetaSy, heavy poor-mic augmentation) to recover the quiet-mic robustness
rem the continuous windows diluted (bench p3suf 141 vs anchor 145; losses concentrated in
rem the quiet-mic long-ayah family). Bet: suppression-resistance is sticky (15 p3 epochs);
rem verify with research/probe_suppression.py after. Selection by audio_bench.
rem Stop: schtasks /end /tn QtAR_p31_restore
cd /d C:\Users\hazem\projects\QtAR
set PYTHONIOENCODING=utf-8
C:\Users\hazem\AppData\Local\Programs\Python\Python313\python.exe training\train_supervisor.py ^
    --epochs 5 --tag _s123_p31 --train-manifest data/raw/phase2/combined_train.csv ^
    --epochs-per-run 1 --init-from training/exp/best_s123_p3.pt -- ^
    --lr 5e-5 --augment --frame-budget 24000 --num-workers 6 >> training\exp\p31_train.log 2>&1
