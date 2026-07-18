@echo off
title v18.13 Trading Bot - EURUSD H4
cd /d "C:\Users\Zabih Ullah Mujahid\OneDrive\Desktop\Regime\deployment_prod"

echo ============================================================
echo  v18.13 EUR/USD H4 Trading Bot
echo  Score: 0.50xML + 0.30xRegime + 0.20xHMM
echo  RANGE enabled (thr=0.52)  ^|  BUY+SELL balanced
echo  DIR_THR=0.55  ^|  1 trade/bar  ^|  direction lock
echo  Price refresh: 1s  ^|  ML: 10s  ^|  Trade: H4 close
echo ============================================================
echo.
echo BEFORE STARTING:
echo   [1] Open MetaTrader 5
echo   [2] Login to ICM demo inside MT5
echo   [3] Wait for GREEN dot bottom-right of MT5
echo.
echo Starting in 5 seconds... (Ctrl+C to cancel)
timeout /t 5 >nul

D:\conda_envs\forex_prod\python.exe demo_runner.py --mode demo

echo.
echo Bot stopped. Press any key to close.
pause