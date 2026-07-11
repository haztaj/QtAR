@echo off
rem Full continuous-corpus alignment sweep (phase-3 labeling). Idempotent — skips existing
rem per-surah CSVs, so rerun to resume. Detached via Task Scheduler (long background bash
rem jobs get killed — see training/run_bob_retrain.cmd, 2026-07-11).
rem Stop: schtasks /end /tn QtAR_align_continuous
cd /d C:\Users\hazem\projects\QtAR
set PYTHONIOENCODING=utf-8
C:\Users\hazem\AppData\Local\Programs\Python\Python313\python.exe data\align_continuous.py ^
    >> data\raw\continuous\alignments\align.log 2>&1
