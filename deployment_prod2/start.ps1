# ============================================================
# start.ps1  --  EUR/USD H4  v18.14  PowerShell Launcher
# Alternative to run_live.bat
# Usage from VS Code terminal:
#   cd "C:\...\deployment_prod"
#   .\start.ps1
# ============================================================
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ROOT

Write-Host ""
Write-Host "  =====================================================" -ForegroundColor Cyan
Write-Host "    EUR/USD H4  --  AI Dashboard  v18.14" -ForegroundColor Cyan
Write-Host "    IC Markets Demo  |  Account: 52869629" -ForegroundColor Cyan
Write-Host "    Dashboard: http://localhost:8501" -ForegroundColor Cyan
Write-Host "  =====================================================" -ForegroundColor Cyan
Write-Host "  ROOT: $ROOT" -ForegroundColor Gray
Write-Host ""

if (-not (Test-Path "$ROOT\app.py")) {
    Write-Host "  ERROR: app.py not found at $ROOT" -ForegroundColor Red
    Write-Host "  Extract the ZIP first!" -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}

$condaHook = "C:\Users\Al Rehman Laptop\anaconda3\shell\condabin\conda-hook.ps1"
if (Test-Path $condaHook) {
    & $condaHook
    conda activate ml_ds_env
}

python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) { pip install streamlit --quiet }

while ($true) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')]  Starting Streamlit..." -ForegroundColor Green
    streamlit run "$ROOT\app.py" `
        --server.port 8501 `
        --server.headless false `
        --browser.gatherUsageStats false `
        --theme.base light
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')]  Stopped. Restarting in 5s..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
