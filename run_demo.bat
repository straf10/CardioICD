@echo off
rem One-click start of the CardioICD demo on Windows (opens the browser when ready).
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe app\run_demo.py %*
) else (
    python app\run_demo.py %*
)
if errorlevel 1 pause
