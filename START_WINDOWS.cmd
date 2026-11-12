@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 run_dashboard.py
) else (
    python run_dashboard.py
)
if errorlevel 1 echo If Python is unavailable, open OPEN_DASHBOARD.html directly.
pause
