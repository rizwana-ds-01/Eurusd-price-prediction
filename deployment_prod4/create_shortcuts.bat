@echo off
title Creating Desktop Shortcuts - v18.13
set FOLDER=C:\Users\Al Rehman Laptop\OneDrive\Desktop\Regime\deployment_prod4
set DESKTOP=C:\Users\Al Rehman Laptop\Desktop

echo Creating shortcuts on Desktop...
echo.

:: ── Start Trading System (Bot + Dashboard) ──────────────────────────────
(
echo Set oWS = WScript.CreateObject^("WScript.Shell"^)
echo sLinkFile = "%DESKTOP%\Start Trading System.lnk"
echo Set oLink = oWS.CreateShortcut^(sLinkFile^)
echo oLink.TargetPath = "%FOLDER%\START_ALL.bat"
echo oLink.WorkingDirectory = "%FOLDER%"
echo oLink.Description = "v18.13 One-Click: Bot + Dashboard"
echo oLink.WindowStyle = 1
echo oLink.Save
) > "%TEMP%\mk_s1.vbs"
cscript //nologo "%TEMP%\mk_s1.vbs"
if exist "%DESKTOP%\Start Trading System.lnk" (
    echo   [OK] Start Trading System
) else (
    echo   [FAIL] Start Trading System
)

:: ── Trading Bot only ─────────────────────────────────────────────────────
(
echo Set oWS = WScript.CreateObject^("WScript.Shell"^)
echo sLinkFile = "%DESKTOP%\Trading Bot.lnk"
echo Set oLink = oWS.CreateShortcut^(sLinkFile^)
echo oLink.TargetPath = "%FOLDER%\demo_trading.bat"
echo oLink.WorkingDirectory = "%FOLDER%"
echo oLink.Description = "v18.13 Bot only"
echo oLink.WindowStyle = 1
echo oLink.Save
) > "%TEMP%\mk_s2.vbs"
cscript //nologo "%TEMP%\mk_s2.vbs"
if exist "%DESKTOP%\Trading Bot.lnk" (
    echo   [OK] Trading Bot
) else (
    echo   [FAIL] Trading Bot
)

:: ── Dashboard only ───────────────────────────────────────────────────────
(
echo Set oWS = WScript.CreateObject^("WScript.Shell"^)
echo sLinkFile = "%DESKTOP%\Trading Dashboard.lnk"
echo Set oLink = oWS.CreateShortcut^(sLinkFile^)
echo oLink.TargetPath = "%FOLDER%\dashboard.bat"
echo oLink.WorkingDirectory = "%FOLDER%"
echo oLink.Description = "v18.13 Dashboard only"
echo oLink.WindowStyle = 1
echo oLink.Save
) > "%TEMP%\mk_s3.vbs"
cscript //nologo "%TEMP%\mk_s3.vbs"
if exist "%DESKTOP%\Trading Dashboard.lnk" (
    echo   [OK] Trading Dashboard
) else (
    echo   [FAIL] Trading Dashboard
)

:: ── Smoke Test ───────────────────────────────────────────────────────────
(
echo Set oWS = WScript.CreateObject^("WScript.Shell"^)
echo sLinkFile = "%DESKTOP%\Smoke Test.lnk"
echo Set oLink = oWS.CreateShortcut^(sLinkFile^)
echo oLink.TargetPath = "%FOLDER%\smoke_test.bat"
echo oLink.WorkingDirectory = "%FOLDER%"
echo oLink.Description = "v18.13 Smoke Test (no MT5 needed)"
echo oLink.WindowStyle = 1
echo oLink.Save
) > "%TEMP%\mk_s4.vbs"
cscript //nologo "%TEMP%\mk_s4.vbs"
if exist "%DESKTOP%\Smoke Test.lnk" (
    echo   [OK] Smoke Test
) else (
    echo   [FAIL] Smoke Test
)

del "%TEMP%\mk_s1.vbs" 2>nul
del "%TEMP%\mk_s2.vbs" 2>nul
del "%TEMP%\mk_s3.vbs" 2>nul
del "%TEMP%\mk_s4.vbs" 2>nul

echo.
echo ============================================================
echo  DONE! Shortcuts on Desktop:
echo.
echo  "Start Trading System"  ONE CLICK starts everything
echo  "Trading Bot"           bot only (no dashboard)
echo  "Trading Dashboard"     dashboard only
echo  "Smoke Test"            verify system without MT5
echo.
echo  DAILY WORKFLOW:
echo    1. Open MT5 and login to ICM demo
echo    2. Double-click "Start Trading System"
echo    3. Dashboard opens at http://localhost:8501
echo    4. ML score updates every 10 seconds
echo    5. Trades execute on H4 bar close (every 4h)
echo    6. Both BUY and SELL signals now active
echo ============================================================
echo.
pause