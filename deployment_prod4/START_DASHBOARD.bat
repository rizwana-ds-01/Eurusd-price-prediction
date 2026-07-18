@echo off
title EUR/USD v18.20 — ICM Live Dashboard
cd /d "%~dp0"

echo ============================================================
echo  EUR/USD H4 — v18.20 Professional Dashboard
echo  ICM Demo ^| MT5 Live ^| White Theme ^| 7 Chart Tabs
echo ============================================================
echo.
echo  MT5 Login  : 52869629
echo  Server     : ICMarketsSC-Demo
echo  Symbol     : EURUSD H4
echo.
echo  BEFORE STARTING:
echo    [1] Open MetaTrader 5
echo    [2] Login to ICM demo account
echo    [3] Wait for green dot bottom-right of MT5
echo    [4] Then press any key below
echo.
pause

echo.
echo  Starting dashboard...
echo  Dashboard: http://localhost:8501
echo  MT5 connects automatically in ~15 seconds
echo.

:: Try conda env first, then system python
if exist "D:\conda_envs\forex_prod\python.exe" (
    D:\conda_envs\forex_prod\python.exe -m streamlit run streamlit_app.py ^
        --server.port 8501 ^
        --server.headless true ^
        --browser.gatherUsageStats false
) else if exist "%USERPROFILE%\anaconda3\envs\ml_ds_env\python.exe" (
    call conda activate ml_ds_env 2>nul
    python -m streamlit run streamlit_app.py ^
        --server.port 8501 ^
        --server.headless true ^
        --browser.gatherUsageStats false
) else if exist "%USERPROFILE%\anaconda3\envs\ml_clean\python.exe" (
    call conda activate ml_clean 2>nul
    python -m streamlit run streamlit_app.py ^
        --server.port 8501 ^
        --server.headless true ^
        --browser.gatherUsageStats false
) else (
    python -m streamlit run streamlit_app.py ^
        --server.port 8501 ^
        --server.headless true ^
        --browser.gatherUsageStats false
)

echo.
echo Dashboard stopped.
pause
