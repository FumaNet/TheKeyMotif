@echo off
rem ===================================================================
rem  Runs the remaining identity-sweep thresholds (100, 95, 90, 80).
rem
rem  Double-click this file, or run it from a terminal. It must NOT be
rem  started from inside Claude Code: a background shell there is killed
rem  when the session ends and is reaped under memory pressure, which is
rem  what ended three previous attempts at the 99-fold 100% threshold.
rem
rem  Safe to re-run. Any threshold whose .pkl already exists is skipped,
rem  so an interrupted run resumes where it stopped.
rem
rem  To run a subset or change the order, pass the parameter through:
rem      RUN_SWEEP.bat -Thresholds 0.80,0.95,0.90,1.00
rem ===================================================================
setlocal
cd /d "%~dp0"

rem Start each run with a fresh status file, keeping the previous one.
if exist "results\sweep_status.txt" move /y "results\sweep_status.txt" "results\sweep_status_prev.txt" >nul 2>&1

echo Starting sweep. This takes roughly 3.5 hours for all four thresholds.
echo Progress is appended to results\sweep_status.txt
echo You can close nothing and leave this window open; it will say DONE below.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0sweep_overnight.ps1" %*

echo.
echo ===== DONE (exit code %ERRORLEVEL%) =====
echo Summary table: results\sweep_summary.md
echo Run log:       results\sweep_status.txt
echo.
pause
