@echo off
REM Long-running training script for Amadeus base model.
REM Run with: scripts\run_overnight.bat
REM This launches the training in the background and writes all output to a log file.

set HF_ENDPOINT=https://hf-mirror.com
set PYTHONIOENCODING=utf-8

set RUN_DIR=D:\Whoami\Amadeus\models\motion\overnight_run
set LOG=%RUN_DIR%\training_output.log
set DATA=D:\Whoami\Amadeus\data\preprocessed\hdtf_subset

if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

echo Starting overnight training at %DATE% %TIME% > "%LOG%"
"C:\Users\Sama608\.conda\envs\amadeus\python.exe" "D:\Whoami\Amadeus\scripts\train_base.py" ^
    --data_dir "%DATA%" ^
    --output_dir "%RUN_DIR%" ^
    --num_epochs 200 ^
    --warmup_steps 200 ^
    --early_stopping_patience 50 ^
    --ema_decay 0.999 ^
    --val_split 0.1 >> "%LOG%" 2>&1

echo Training finished at %DATE% %TIME% >> "%LOG%"
echo Exit code: %ERRORLEVEL% >> "%LOG%"
