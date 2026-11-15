@echo off
title Hospital Appointment No-Show Prediction Launcher
echo ======================================================================
echo    Starting Hospital Appointment No-Show Prediction Platform
echo ======================================================================
echo.

cd /d "%~dp0"

echo [1/2] Starting FastAPI Backend on port 8000...
start "Hospital No-Show API" cmd /k "python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 2 >nul

echo [2/2] Starting Streamlit Frontend on port 8501...
start "Hospital No-Show Web App" cmd /k "python -m streamlit run frontend/app.py --server.port 8501"

echo.
echo ======================================================================
echo    Services Started Successfully!
echo.
echo    - Streamlit UI:        http://localhost:8501
echo    - FastAPI API:         http://127.0.0.1:8000
echo    - Swagger Docs:        http://127.0.0.1:8000/docs
echo ======================================================================
echo.
pause
