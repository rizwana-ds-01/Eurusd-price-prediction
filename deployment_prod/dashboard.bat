@echo off
title v18.13 Pro Dashboard
cd /d "C:\Users\Zabih Ullah Mujahid\OneDrive\Desktop\Regime\deployment_prod"

echo ============================================================
echo  v18.13 Professional Trading Dashboard
echo  Opening: http://localhost:8501
echo  Light theme  ^|  BUY+SELL signals  ^|  Live UDL chart
echo ============================================================
timeout /t 3 >nul
start "" "http://localhost:8501"
D:\conda_envs\forex_prod\python.exe -m streamlit run streamlit_app.py --server.headless true

echo.
echo Dashboard stopped. Press any key to close.
pause