@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  CLIMATE-XFER RAG Demo — Assignment 3
::  AI and Large Models | Masters Program 2025-2027
::  Run this file to install dependencies and launch the app
:: ============================================================

set "DASHBOARD_DIR=D:\Masters Program 2025-2027\Semester 1_2\Artificial Intelligence and Large  Models\Assignmnets\Assignment 3"

cd /d "%DASHBOARD_DIR%"
if errorlevel 1 (
    echo ERROR: Could not navigate to dashboard folder.
    echo Expected: %DASHBOARD_DIR%
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   CLIMATE-XFER RAG Demo ^| Assignment 3
echo   Retrieval-Augmented Generation with CLIMATE-XFER PDF
echo ============================================================
echo.

:: ── Check Python ────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH.
    echo Please install Python 3.9+ from https://www.python.org
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo Python found: %PY_VER%
echo.

:: ── Install dependencies ─────────────────────────────────────
echo Installing required packages (first run may take a few minutes)...
echo Note: sentence-transformers will download the MiniLM model (~90 MB).
echo.
pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    echo.
    echo ERROR: Package installation failed.
    echo Try running manually:  pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo Packages installed successfully.
echo.

:: ── Launch dashboard ─────────────────────────────────────────
echo ============================================================
echo   Launching CLIMATE-XFER RAG Demo...
echo   URL:  http://localhost:8502
echo.
echo   The browser should open automatically.
echo   If not, paste the URL above into your browser.
echo.
echo   Press Ctrl+C in this window to stop the server.
echo ============================================================
echo.

streamlit run climate_xfer_rag.py ^
    --server.port 8502 ^
    --server.headless false ^
    --browser.gatherUsageStats false

pause
