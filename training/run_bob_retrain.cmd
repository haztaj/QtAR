@echo off
rem Best-of-both retrain (cleaned RetaSy labels + junk-noise augmentation), leak-bounded
rem supervisor, exact resume from last_s123_bob.pt. Detached from any Claude session via
rem Task Scheduler (background bash jobs were killed twice mid-run, 2026-07-11).
rem Stop:   schtasks /end /tn QtAR_bob_retrain   (state survives; rerun this to resume)
cd /d C:\Users\hazem\projects\QtAR
set PYTHONIOENCODING=utf-8
C:\Users\hazem\AppData\Local\Programs\Python\Python313\python.exe training\train_supervisor.py ^
    --epochs 27 --tag _s123_bob --train-manifest data/raw/phase2/combined_train.csv ^
    --epochs-per-run 1 -- --lr 1.6e-4 --augment --noise-dir data/raw/phase2/junk_noise ^
    --frame-budget 24000 --num-workers 6 >> training\exp\bob_train.log 2>&1
