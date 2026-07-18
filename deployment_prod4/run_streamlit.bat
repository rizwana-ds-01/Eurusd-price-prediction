@echo off
title EUR/USD AI Streamlit Dashboard v18.15
cd /d "%~dp0"

:: Activate conda
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" 2>nul
call conda activate ml_ds_env 2>nul || call conda activate ml_clean 2>nul

:: Create folders
if not exist "models" mkdir models
if not exist "data"   mkdir data
if not exist "logs"   mkdir logs

:: Install streamlit + plotly if missing
pip show streamlit >nul 2>&1 || pip install streamlit plotly --quiet
pip show plotly    >nul 2>&1 || pip install plotly --quiet

echo.
echo  =====================================================
echo   EUR/USD H4 -- AI Dashboard v18.15 (Streamlit)
echo   Dashboard: http://localhost:8501
echo  =====================================================
echo.
echo Starting...
echo.

python -m streamlit run streamlit_app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

pause
