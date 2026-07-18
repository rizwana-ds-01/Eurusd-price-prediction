@echo off
:: ============================================================
:: EUR/USD H4  v18.14  --  Streamlit Dashboard Launcher
:: IC Markets Demo  |  Account: 52869629
::
:: HOW TO USE:
::   1. Extract ZIP to Desktop (right-click -> Extract All)
::   2. Double-click THIS file
::   3. Browser opens at http://localhost:8501
:: ============================================================

title EUR/USD AI Dashboard v18.14

:: Always cd to folder where THIS .bat lives (critical!)
cd /d "%~dp0"

echo.
echo  =====================================================
echo    EUR/USD H4  --  AI Dashboard  v18.14
echo    IC Markets Demo  ^|  Account: 52869629
echo    Dashboard: http://localhost:8501
echo  =====================================================
echo.
echo  Folder: %CD%
echo.

:: Verify app.py exists
if not exist "%~dp0app.py" (
    echo  ERROR: app.py not found!
    echo  Please EXTRACT the zip before running.
    echo  Do NOT run from inside the zip file.
    pause
    exit /b 1
)

:: Verify models folder has key file
if not exist "%~dp0models\B_scaled_v6_clean.pkl" (
    echo  ERROR: models\B_scaled_v6_clean.pkl not found!
    echo  Make sure models folder has all .pkl files.
    pause
    exit /b 1
)

:: Activate conda
call "C:\Users\Al Rehman Laptop\anaconda3\Scripts\activate.bat" 2>nul
call conda activate ml_ds_env 2>nul

:: Install streamlit if missing
python -c "import streamlit" 2>nul || pip install streamlit --quiet

echo  [%TIME%]  Starting Streamlit...
echo.

:loop
streamlit run "%~dp0app.py" ^
    --server.port 8501 ^
    --server.headless false ^
    --browser.gatherUsageStats false ^
    --theme.base light

echo.
echo  [%TIME%]  Stopped. Restarting in 5 s...
ping -n 6 127.0.0.1 >nul
goto loop
