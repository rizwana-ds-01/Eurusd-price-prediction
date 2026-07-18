@echo off
title EUR/USD AI Dashboard v18.14
cd /d "%~dp0"

:: Activate conda
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" 2>nul
call conda activate ml_ds_env 2>nul || call conda activate ml_clean 2>nul

:: Create folders
if not exist "models" mkdir models
if not exist "data"   mkdir data
if not exist "logs"   mkdir logs

echo.
echo  =====================================================
echo   EUR/USD H4 -- AI Dashboard v18.14
echo   IC Markets Demo  ^|  Account: 52869629
echo   Open browser: http://localhost:5000
echo  =====================================================
echo.

python live_dashboard.py
pause
