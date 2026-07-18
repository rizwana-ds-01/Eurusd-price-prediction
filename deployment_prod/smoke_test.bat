@echo off
title v18.13 Smoke Test
cd /d "C:\Users\Zabih Ullah Mujahid\OneDrive\Desktop\Regime\deployment_prod"

echo ============================================================
echo  v18.13 SMOKE TEST (no MT5 needed)
echo  Tests: RANGE trades, BUY/SELL balance, 1trade/bar
echo ============================================================
echo.

D:\conda_envs\forex_prod\python.exe demo_runner.py --mode test

echo.
echo ============================================================
echo  If smoke test passed:
echo    1. Open MT5 and login to ICM
echo    2. Double-click START_ALL.bat
echo ============================================================
echo.
pause