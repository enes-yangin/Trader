@echo off
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python 3.10+ not found. Install it from https://www.python.org/downloads/
    exit /b 1
)

python bootstrap.py %*
