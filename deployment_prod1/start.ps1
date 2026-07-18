# ============================================================
# start.ps1  —  EUR/USD H4  v18.14  Live Deployment
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  =====================================================" -ForegroundColor Cyan
Write-Host "    EUR/USD H4 -- AI Trading Dashboard  v18.14" -ForegroundColor Cyan
Write-Host "    IC Markets Demo  |  Account: 52869629" -ForegroundColor Cyan
Write-Host "    Dashboard --> http://localhost:5000" -ForegroundColor Cyan
Write-Host "  =====================================================" -ForegroundColor Cyan
Write-Host ""

# ── Conda init ──────────────────────────────────────────────
$condaHook = "C:\Users\Al Rehman Laptop\anaconda3\shell\condabin\conda-hook.ps1"
if (Test-Path $condaHook) {
    & $condaHook
    conda activate ml_ds_env
} else {
    Write-Warning "Conda hook not found at $condaHook — using system Python"
}

# ── Move to project root ─────────────────────────────────────
Set-Location $PSScriptRoot

# ── Install / upgrade requirements ──────────────────────────
Write-Host "[*] Checking requirements..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet

# ── Auto-restart loop ────────────────────────────────────────
while ($true) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')]  Starting live dashboard..." -ForegroundColor Green
    python app\live_dashboard.py
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')]  Process ended — restarting in 5 s..." -ForegroundColor Red
    Start-Sleep -Seconds 5
}
