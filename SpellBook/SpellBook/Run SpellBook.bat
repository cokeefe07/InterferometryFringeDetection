@echo off
REM One-click launcher: creates a local Python environment on first run,
REM installs the requirements, then starts SpellBook.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run: creating Python environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Could not create the environment. Is Python 3.9+ installed and on PATH?
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Installing requirements failed.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" SpellBook.py
if errorlevel 1 pause
