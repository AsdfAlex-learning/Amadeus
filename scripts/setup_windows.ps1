# Amadeus — Windows Setup Script
# ================================
# Run from repo root: .\scripts\setup_windows.ps1
#
# Steps:
#   1. Create conda environment (Python 3.11.5)
#   2. Install conda packages (portaudio)
#   3. Install pip packages (torch, transformers, live2d-py, etc.)
#   4. Pre-download encoder weights (optional)
#   5. Verify installation

param(
    [switch]$SkipEnv,
    [switch]$SkipDownloads
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Amadeus — Windows Environment Setup" -ForegroundColor Cyan
Write-Host "  Python 3.11.5 | RTX 4060 Ti 8GB" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not $SkipEnv) {
    Write-Host "`n[1/4] Creating conda environment..." -ForegroundColor Yellow
    conda env create -f environment.yml
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Environment creation failed. Trying step-by-step approach..." -ForegroundColor Red
        conda create -n amadeus python=3.11.5 pip=23.2.1 setuptools=68.0.0 -y
        conda activate amadeus
        conda install -c conda-forge portaudio -y
        pip install -r requirements-lock.txt
    }
}

Write-Host "`n[2/4] Verifying core packages..." -ForegroundColor Yellow
conda run -n amadeus python -c @"
import sys
packages = ['torch', 'transformers', 'PySide6', 'live2d', 'numpy', 'pyaudio', 'cv2', 'yaml', 'loguru', 'soundfile', 'pyttsx3']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  OK  {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'  MISS {pkg}')
if missing:
    print(f'Missing: {missing}')
    sys.exit(1)
else:
    print('All core packages OK')
"@

Write-Host "`n[3/4] Checking device detection..." -ForegroundColor Yellow
conda run -n amadeus python -c @"
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem // (1024**3)} GB')
print(f'  MPS available: {torch.backends.mps.is_available()}')
"@

if (-not $SkipDownloads) {
    Write-Host "`n[4/4] Downloading encoder weights (optional)..." -ForegroundColor Yellow
    Write-Host "  Run manually: conda run -n amadeus python scripts/download_models.py"
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Setup complete! Next steps:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  1. conda activate amadeus"
Write-Host "  2. python scripts/download_models.py"
Write-Host "  3. python demo.py"
Write-Host "  4. python -m src.main  (full app)"