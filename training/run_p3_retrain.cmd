@echo off
rem Phase-3 concatenation training: fine-tune best_s123_mic on the MIXED manifest
rem (per-ayah clips + real continuous multi-ayah windows) — the repetition-suppression
rem root fix. Standard augmentation (NOT the junk-noise dir — that arm failed its gate).
rem Selection by audio_bench + probe_suppression, NOT val PER (taint-audit rule).
rem Stop: schtasks /end /tn QtAR_p3_retrain   (resumable; rerun this to continue)
cd /d C:\Users\hazem\projects\QtAR
set PYTHONIOENCODING=utf-8
C:\Users\hazem\AppData\Local\Programs\Python\Python313\python.exe training\train_supervisor.py ^
    --epochs 15 --tag _s123_p3 --train-manifest data/raw/phase3/combined_train.csv ^
    --epochs-per-run 1 --init-from training/exp/best_s123_mic.pt -- ^
    --lr 1e-4 --augment --frame-budget 24000 --num-workers 6 >> training\exp\p3_train.log 2>&1
