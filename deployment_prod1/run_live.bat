@echo off
:: ============================================================
:: EUR/USD H4  —  v18.14  Live Deployment Runner
:: IC Markets Demo  |  MT5 #52869629
:: ============================================================

title EUR/USD AI Dashboard v18.14

:: ── Conda / Python env ─────────────────────────────────────
call "C:\Users\Al Rehman Laptop\anaconda3\Scripts\activate.bat"
call conda activate ml_ds_env

:: ── Change to project root ──────────────────────────────────
cd /d "%~dp0"

echo.
echo  =====================================================
echo    EUR/USD H4  --  AI Trading Dashboard  v18.14
echo    IC Markets Demo  ^|  Account: 52869629
echo    Dashboard : http://localhost:5000
echo  =====================================================
echo.

:loop

echo [%date% %time%]  Starting dashboard...

python app\live_dashboard.py

echo.
echo [%date% %time%]  Dashboard stopped — restarting in 5 s...
timeout /t 5 /nobreak

goto loop
